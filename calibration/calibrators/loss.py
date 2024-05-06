import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self,
                 ignore_index,
                 **kwargs):
        super().__init__()
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, preds: torch.Tensor, target: torch.Tensor, *args):
        loss = self.loss_func(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
        return loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self,
                 ignore_index,
                 opt,
                 normalize_length = True,
                 **kwargs):
        """Construct an LabelSmoothingLoss object."""
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.padding_idx = ignore_index
        self.confidence = 1.0 - opt.beta
        self.smoothing = opt.beta
        self.normalize_length = normalize_length

    def forward(self, preds: torch.Tensor, target: torch.Tensor, *args) -> torch.Tensor:
        batch_size = preds.size(0)
        preds = preds.view(-1, preds.size(-1))
        target = target.view(-1)
        true_dist = torch.zeros_like(preds)
        true_dist.fill_(self.smoothing / (preds.size(-1) - 1))
        ignore = target == self.padding_idx  # (B,)
        total = len(target) - ignore.sum().item()
        target = target.masked_fill(ignore, 0)  # avoid -1 index
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(preds, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom


class PairWiseWeightSmoothLoss(nn.Module):
    def __init__(self, 
                ignore_index, 
                opt, 
                normalize_length=True, 
                **kwargs):
        super(PairWiseWeightSmoothLoss, self).__init__()
        self.padding_idx = ignore_index
        self.confidence = 1.0 - opt.beta
        self.smoothing = opt.beta 
        self.normalize_length = normalize_length
        
        self.criterion = nn.KLDivLoss(reduction="none")
        self.matric = torch.tensor(np.load('confusion_matrices/Attn_confusion_matrices.npy'))

    def forward(self, input, target, length, *arg):
        batch_size, max_time_len, _ = input.shape
        forth_target = torch.zeros_like(target)
        forth_target[:, 1:] = target[:, :-1]
        forth_target = forth_target.view(-1)
        target = target.contiguous().view(-1)
        ignore = target == self.padding_idx
        total = (ignore == True).sum().item()
        input = input.view(-1, input.shape[-1])
        log_prob = F.log_softmax(input, dim=-1)

        self.matric = self.matric.to(input.device)
        smoothing = (1 - torch.pow((1-self.smoothing), 1.0 / length)).unsqueeze(1).repeat(1, max_time_len).view(-1)
        weight = (smoothing.unsqueeze(1) * self.matric[forth_target.tolist(), target.tolist(), :])
        src = (1. -  weight.sum(dim=1))
        src = src.unsqueeze(-1).repeat(1, weight.size(1)) 
        weight.scatter_(-1, target.unsqueeze(1), src)
        weight = weight.type_as(log_prob)

        denom = total if self.normalize_length else batch_size
        loss = (-weight * log_prob).masked_fill(ignore.unsqueeze(1), 0).sum() / denom
        return loss