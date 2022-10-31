from multiprocessing import reduction
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class MDCA(torch.nn.Module):
    def __init__(self):
        super(MDCA,self).__init__()

    def forward(self , output, target):
        output = torch.softmax(output, dim=1)
        # [batch, classes]
        loss = torch.tensor(0.0).cuda()
        batch, classes = output.shape
        for c in range(classes):
            avg_count = (target == c).float().mean()
            avg_conf = torch.mean(output[:,c])
            loss += torch.abs(avg_conf - avg_count)
        denom = classes
        loss /= denom
        return loss

class LogitMarginL1(nn.Module):
    """Add marginal penalty to logits:
        CE + alpha * max(0, max(l^n) - l^n - margin)

    Args:
        margin (float, optional): The margin value. Defaults to 10.
        alpha (float, optional): The balancing weight. Defaults to 0.1.
        ignore_index (int, optional):
            Specifies a target value that is ignored
            during training. Defaults to -100.

        The following args are related to balancing weight (alpha) scheduling.
        Note all the results presented in our paper are obtained without the scheduling strategy.
        So it's fine to ignore if you don't want to try it.

        schedule (str, optional):
            Different stragety to schedule the balancing weight alpha or not:
            "" | add | multiply | step. Defaults to "" (no scheduling).
            To activate schedule, you should call function
            `schedula_alpha` every epoch in your training code.
        mu (float, optional): scheduling weight. Defaults to 0.
        max_alpha (float, optional): Defaults to 100.0.
        step_size (int, optional): The step size for updating alpha. Defaults to 100.
    """
    def __init__(self,
                 margin: float = 10,
                 alpha: float = 0.1,
                 ignore_index: int = 0,
                 schedule: str = "",
                 mu: float = 0,
                 max_alpha: float = 100.0,
                 step_size: int = 100,
                 **kwargs):
        super().__init__()
        assert schedule in ("", "add", "multiply", "step")
        self.margin = margin
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.mu = mu
        self.schedule = schedule
        self.max_alpha = max_alpha
        self.step_size = step_size

        self.cross_entropy = nn.CrossEntropyLoss()


    def schedule_alpha(self, epoch):
        """Should be called in the training pipeline if you want to se schedule alpha
        """
        if self.schedule == "add":
            self.alpha = min(self.alpha + self.mu, self.max_alpha)
        elif self.schedule == "multiply":
            self.alpha = min(self.alpha * self.mu, self.max_alpha)
        elif self.schedule == "step":
            if (epoch + 1) % self.step_size == 0:
                self.alpha = min(self.alpha * self.mu, self.max_alpha)

    def get_diff(self, inputs):
        max_values = inputs.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        diff = max_values - inputs
        return diff

    def forward(self, inputs, targets, *args):
        inputs, targets  = inputs.view(-1, inputs.shape[-1]), targets.contiguous().view(-1)
        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_ce = self.cross_entropy(inputs, targets)
        # get logit distance
        diff = self.get_diff(inputs)
        # linear penalty where logit distances are larger than the margin
        loss_margin = F.relu(diff-self.margin).mean()
        loss = loss_ce + self.alpha * loss_margin

        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self,
                 ignore_index=0,
                 **kwargs):
        super().__init__()
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    def forward(self, preds, target, *args):
        loss = self.loss_func(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
        return loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self,
                 ignore_index: int,
                 alpha: float,
                 normalize_length: bool = True,
                 **kwargs):
        """Construct an LabelSmoothingLoss object."""
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.padding_idx = ignore_index
        self.confidence = 1.0 - alpha
        self.smoothing = alpha
        self.normalize_length = normalize_length

    def forward(self, preds: torch.Tensor, target: torch.Tensor, *args) -> torch.Tensor:
        batch_size = preds.size(0)
        preds = preds.view(-1, preds.size(-1))
        target = target.view(-1)
        # use zeros_like instead of torch.no_grad() for true_dist,
        # since no_grad() can not be exported by JIT
        true_dist = torch.zeros_like(preds)
        true_dist.fill_(self.smoothing / (preds.size(-1) - 1))
        ignore = target == self.padding_idx  # (B,)
        total = len(target) - ignore.sum().item()
        target = target.masked_fill(ignore, 0)  # avoid -1 index
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(preds, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom

class CaslsChineseAttnLoss(nn.Module):
    def __init__(self, ignore_index, matric, alpha=0.0, normalize_length=True, device=None, **kwargs):
        super(CaslsChineseAttnLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smoothing = alpha
        self.matric = matric
        self.device = device
        self.kldiv = nn.KLDivLoss(reduction="none")
        self.normalize_length = normalize_length
    def forward(self, inputs, targets, _, labels):
        inputs, targets  = inputs.view(-1, inputs.shape[-1]), targets.contiguous().view(-1)
        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]
        nclass = inputs.shape[-1]

        i = 0
        need_smoothed = []
        for j in torch.nonzero(targets==1).squeeze():
            forth = torch.ones_like(targets[i:j+1]) * (targets.shape[-1] - 1)
            forth[1:] = targets[i:j]
            need_smoothed.append(torch.tensor(self.matric[forth.tolist(), targets[i:j+1].tolist()]))
            i = j + 1
        need_smoothed = torch.cat(need_smoothed, 0)

        length = torch.IntTensor(list(map(len, labels))) + 1
        smoothing = 1-torch.pow(1-self.smoothing, 1/length).view(-1,1)
        smooth_list = []
        for sm, l in zip(smoothing, length):
            sm = sm.repeat(l)
            smooth_list.append(sm)
        smoothing = torch.cat(smooth_list) * need_smoothed

        weight = torch.ones_like(inputs) * (smoothing / (nclass-1)).view(-1,1).to(self.device)
        src = (1. - weight.sum(dim=1))
        weight = weight.scatter(-1, targets.view(-1,1), src.view(-1,1))
        kl = self.kldiv(torch.log_softmax(inputs, dim=1), weight)

        return kl.sum()/length.sum()

class PairWiseWeightSmoothLoss(nn.Module):
    def __init__(self, 
                ignore_index, 
                matric, 
                alpha=0.0, 
                normalize_length=True, 
                device=None, 
                **kwargs):
        super(PairWiseWeightSmoothLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.padding_idx = ignore_index
        self.confidence = 1.0 - alpha
        self.smoothing = alpha
        self.matric = torch.tensor(matric, dtype=torch.float32)[:-1, :-1, :-1]
        self.normalize_length = normalize_length
        self.device = device
    def forward(self, input, target, _, labels):
        length = torch.FloatTensor(list(map(len, labels))) + 1.0
        batch_size, max_time_len, _ = input.shape

        forth_target = torch.zeros_like(target)
        forth_target[:, 1:] = target[:, :-1]
        forth_target = forth_target.view(-1)
        target = target.contiguous().view(-1)
        ignore = target == self.padding_idx
        total = (ignore == True).sum().item()
        input = input.view(-1, input.shape[-1])
        log_prob = F.log_softmax(input, dim=-1)

        smoothing = (1 - torch.pow((1-self.smoothing), 1.0 / length)).unsqueeze(1).repeat(1, max_time_len).view(-1)
        weight = (smoothing.unsqueeze(1) * self.matric[forth_target.tolist(), target.tolist(), :]).to(self.device)
        src = (1. -  weight.sum(dim=1))
        src = src.unsqueeze(-1).repeat(1, weight.size(1)) 
        weight.scatter_(-1, target.unsqueeze(1), src)
        weight = weight.type_as(log_prob)

        denom = total if self.normalize_length else batch_size
        loss = (-weight * log_prob).masked_fill(ignore.unsqueeze(1), 0).sum() / denom

        return loss

class EntropyRegularAttn(nn.Module):
    def __init__(self, ignore_index, alpha=0.05, **kwargs):
        super(EntropyRegularAttn, self).__init__()
        self.beta = alpha
        self.ignore_index = ignore_index
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, *args): 
        inputs, targets  = inputs.view(-1, inputs.shape[-1]), targets.contiguous().view(-1)
        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]
        ce_loss = self.ce_loss(inputs, targets)
        entrop_all = -(inputs.softmax(-1)*inputs.log_softmax(-1)).sum() / inputs.shape[0]
        loss = ce_loss - self.beta * entrop_all
        return loss

class GraduatedLabelSmoothingAttn(nn.Module):
    def __init__(self, 
                ignore_index=0,
                alpha=0.0,
                 normalize_length=True,
                **kwargs):
        super(GraduatedLabelSmoothingAttn, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.padding_idx = ignore_index
        self.confidence = 1.0 - alpha
        self.smoothing = alpha
        self.normalize_length = normalize_length
    def forward(self, input, target, *args):
        self.size = input.size(-1)
        batch_size = input.size(0)
        input = input.view(-1, input.shape[-1])
        target = target.view(-1)
        
        pred_probability, _ = torch.softmax(input, dim=1).max(1)
        smoothing = self.smoothing * torch.ones_like(input)
        smoothing[pred_probability >= 0.7, :] = 3 * self.smoothing
        smoothing[pred_probability <= 0.3, :] = 0.0
        true_dist = smoothing / (self.size - 1)
        confidence = 1 - true_dist.sum(-1)
        ignore = target == self.padding_idx  # (B,)
        total = len(target) - ignore.sum().item()
        #target = target.masked_fill(ignore, 0)  0
        true_dist.scatter_(1, target.unsqueeze(1), confidence.unsqueeze(1))
        kl = self.criterion(torch.log_softmax(input, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom