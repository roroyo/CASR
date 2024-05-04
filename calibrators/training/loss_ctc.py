from multiprocessing import reduction
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dtw import dtw

class MDCA(torch.nn.Module):
    def __init__(self):
        super(MDCA,self).__init__()

    def forward(self, output, target):
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

class CTCClassficationAndMDCA(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=2.0, ignore_index=0, **kwargs):
        super(CTCClassficationAndMDCA, self).__init__()
        self.beta = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.classification_loss = CTC_Loss()
        self.MDCA = MDCA()

    def forward(self, preds, text, preds_size, length, *args):
        preds = preds.log_softmax(2).permute(1, 0, 2)
        loss_cls = self.classification_loss(preds, text, preds_size, length)

        inputs, targets  = preds.view(-1, preds.shape[-1]), text.contiguous().view(-1)
        loss_cal = self.MDCA(inputs, targets)
        return loss_cls + self.beta * loss_cal


class CTC_Loss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_func = torch.nn.CTCLoss(zero_infinity=True)
    def forward(self, preds, text, preds_size, length, *args):
        preds = preds.log_softmax(2).permute(1, 0, 2)
        loss = self.loss_func(preds, text, preds_size, length)
        return loss

class CTCEntropyRegular(nn.Module):
    def __init__(self, alpha=0.1, **kwargs):
        super(CTCEntropyRegular, self).__init__()
        self.beta = alpha
        self.ctc = nn.CTCLoss(zero_infinity=True)
    def forward(self, preds, text, preds_size, length, *args):
        preds_cls = preds.log_softmax(2).permute(1, 0, 2)
        ctc_loss = self.ctc(preds_cls, text, preds_size, length)

        total_input = preds.view(-1, preds.shape[-1])
        entrop_all = -(total_input.softmax(1)*total_input.log_softmax(1)).sum() / total_input.shape[0]
        loss = ctc_loss - self.beta*entrop_all
        return loss

class CTCLabelSmoothLoss(nn.Module):
    def __init__(self, alpha=0.0, **kwargs):
        super(CTCLabelSmoothLoss, self).__init__()
        self.smoothing = alpha
        self.ctc = nn.CTCLoss(zero_infinity=True)
        self.kldiv = nn.KLDivLoss()

    def forward(self, preds, text, preds_size, length, *args):
        preds = preds.log_softmax(2).permute(1, 0, 2)
        ctc_loss = self.ctc(preds, text, preds_size, length)
        kl_inp = preds.transpose(0, 1)
        uni_distri = torch.full_like(kl_inp, 1 / preds.shape[-1])
        kldiv_loss = self.kldiv(kl_inp, uni_distri)
        loss = (1 - self.smoothing) * ctc_loss + self.smoothing * kldiv_loss
        return loss

class CTCLogitMarginL1(nn.Module):
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
        self.ctc = nn.CTCLoss(zero_infinity=True)

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

    def forward(self, input, target, input_length, target_length, *args):
        input_ctc = input.log_softmax(2).permute(1, 0, 2)
        loss_ctc = self.ctc(input_ctc, target, input_length, target_length)
        
        input = input.view(-1, input.shape[-1])
        diff = self.get_diff(input)

        loss_margin = F.relu(diff-self.margin).mean()
        loss = loss_ctc + self.alpha * loss_margin
        return loss

class CTCGraduatedLabelSmoothing(nn.Module):
  def __init__(self, 
                alpha=0.0,
                sequence_normalize=True,
                **kwargs):
    super(CTCGraduatedLabelSmoothing, self).__init__()
    self.criterion = nn.KLDivLoss(reduction="none")
    self.confidence = 1.0 - alpha
    self.smoothing = alpha
    self.ctc = nn.CTCLoss(zero_infinity=True)
    self.normalize_length = sequence_normalize
  def forward(self, preds, text, preds_size, length, *args):
    preds_cls = preds.log_softmax(2).permute(1, 0, 2)
    ctc_loss = self.ctc(preds_cls, text, preds_size, length)

    size = preds.size(-1)
    preds = preds.view(-1, size)
    pred_probability, _ = torch.softmax(preds, dim=1).max(1)
    smoothing = self.smoothing * torch.ones_like(preds[:, 0])
    smoothing[pred_probability >= 0.7] = 3*self.smoothing
    smoothing[pred_probability <= 0.3] = 0.0
    smoothing.unsqueeze_(1)
    uni_distri = torch.full_like(preds, 1 / size)
    kl = self.criterion(torch.log_softmax(preds, dim=1), uni_distri)
    return (1 - smoothing).mean() * ctc_loss + (smoothing * kl).mean()


class SequenceSmoothLossCtc_v10(nn.Module):
    '''
        只把vis和semantic的str拿出来, 然后 loss = (1 - self.alpha) * loss_master + self.alpha * loss_smooth,
        没有用到conf, 而且每个str的损失是一致对待
    '''
    def __init__(self, 
                 converter,
                 semantic,
                 alpha=0.0,
                 **kwargs
                ):
        super().__init__()
        semantic[''] = [['', '', '', '', '', ], [0.2, 0.2, 0.2, 0.2, 0.2]]
        self.converter = converter
        self.semantic = semantic
        self.alpha = alpha
        self.loss_func = torch.nn.CTCLoss(zero_infinity=True)
    def forward(self, preds, text, preds_size, length, visual, labels, *args):
        time_length, size = preds.shape[1:]

        preds_master = preds.log_softmax(2).permute(1, 0, 2)
        loss_master = self.loss_func(preds_master, text, preds_size, length)

        smoothing_list = [visual[idx]['str'] + self.semantic[label][0] for idx, label in enumerate(labels)]
        text, length = zip(*[self.converter.encode(texts) for texts in smoothing_list])

        text, length = torch.cat(text, dim=0), torch.cat(length, dim=0)
        preds = preds.unsqueeze(1).repeat(1, text.shape[0] // preds.shape[0] , 1, 1).view(-1, time_length, size)
        preds_size = torch.IntTensor([time_length] * preds.size(0))

        preds_smooth = preds.log_softmax(2).permute(1, 0, 2)
        loss_smooth = self.loss_func(preds_smooth, text, preds_size, length)
        loss = (1 - self.alpha) * loss_master + self.alpha * loss_smooth
        return loss

class SequenceSmoothLossCTC_PAMI(nn.Module):
    def __init__(self, converter, alpha=0.0, smooth_tail=0.01, **kwargs):
        super().__init__()
        self.converter = converter
        self.alpha = alpha
        self.smooth_tail = smooth_tail
        self.loss_func = torch.nn.CTCLoss(zero_infinity=True, reduction='none')
    def forward(self, preds, text, preds_size, length, visual, labels, *args):
        _, T, C = preds.shape
        preds_master = preds.log_softmax(2).permute(1, 0, 2)
        loss_master = self.loss_func(preds_master, text, preds_size, length)
        conf_list = torch.exp(-loss_master).detach().clone()
        loss_master = (loss_master / length).mean()
        
        smoothing_list = [visual[idx]['str'] for idx, label in enumerate(labels)]
        text, length = zip(*[self.converter.encode(texts) for texts in smoothing_list])
        text, length = torch.cat(text, dim=0), torch.cat(length, dim=0)

        cor_num = text.shape[0] // preds.shape[0]
        preds = preds.unsqueeze(1).repeat(1, cor_num, 1, 1).view(-1, T, C)
        preds_size = torch.IntTensor([T] * preds.shape[0])
        preds_smooth = preds.log_softmax(2).permute(1, 0, 2)
        loss_smooth = self.loss_func(preds_smooth, text, preds_size, length)

        ranking = (self.smooth_tail + (1.0 - self.smooth_tail) * torch.pow((1 - conf_list), 2)).unsqueeze(1).repeat(1, cor_num).view(-1)
        length = torch.clamp(length, min=1)
        loss = loss_master + self.alpha * (ranking * loss_smooth / length).mean()
        return loss

class SequenceSmoothLossCTC_PAM_wo_adapt(nn.Module):
    def __init__(self, converter, alpha=0.0, smooth_tail=0.01, **kwargs):
        super().__init__()
        self.converter = converter
        self.alpha = alpha
        self.smooth_tail = smooth_tail
        self.loss_func = torch.nn.CTCLoss(zero_infinity=True, reduction='none')
    def forward(self, preds, text, preds_size, length, visual, labels, *args):
        _, T, C = preds.shape
        preds_master = preds.log_softmax(2).permute(1, 0, 2)
        loss_master = self.loss_func(preds_master, text, preds_size, length)
        conf_list = torch.exp(-loss_master).detach().clone()
        loss_master = (loss_master / length).mean()
        
        smoothing_list = [visual[idx]['str'] for idx, label in enumerate(labels)]
        text, length = zip(*[self.converter.encode(texts) for texts in smoothing_list])
        text, length = torch.cat(text, dim=0), torch.cat(length, dim=0)

        cor_num = text.shape[0] // preds.shape[0]
        preds = preds.unsqueeze(1).repeat(1, cor_num, 1, 1).view(-1, T, C)
        preds_size = torch.IntTensor([T] * preds.shape[0])
        preds_smooth = preds.log_softmax(2).permute(1, 0, 2)
        loss_smooth = self.loss_func(preds_smooth, text, preds_size, length)

        # ranking = (self.smooth_tail + (1.0 - self.smooth_tail) * torch.pow((1 - conf_list), 2)).unsqueeze(1).repeat(1, cor_num).view(-1)
        length = torch.clamp(length, min=1)
        loss = loss_master + self.alpha * (loss_smooth / length).mean()
        return loss