from lib2to3.pytree import convert
from multiprocessing import reduction
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from tools.ctc_utils import ctc_prefix_beam_search, get_op_seq

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

class CTCCASLS_v3(nn.Module):
    def __init__(self, matric, converter, blank=0, alpha=0.05, **kwargs):
        super(CTCCASLS_v3, self).__init__()
        self.smoothing = alpha
        self.matric = np.zeros((38,38,38))
        self.matric[0] = np.eye(38)
        self.matric[:,0,0] = 1
        self.matric[1:,1:,1:] = matric
        self.smooth_matrix = matric
        self.ctc = nn.CTCLoss(reduction='mean', blank=blank, zero_infinity=True)
        self.kldiv = nn.KLDivLoss(size_average=True, reduce=False)
        self.converter = converter

    def forward(self, input, target, input_length, target_length, _, labels,*args):
        '''
        input: B T C
        target: T S
        input_length: T
        target_length: T
        '''
        _, preds_index = input.max(2)
        nclass = input.shape[-1]
        ctc_input = input.log_softmax(2).permute(1, 0, 2)
        ctc_loss = self.ctc(ctc_input, target, input_length, target_length)

        kl_inp = input
        B, T, C = kl_inp.size()  # 96 26 37
        # step_matric = np.zeros((B, T, C))

        kldiv_loss = 0
        
        preds_str,input_pos = self.converter.decode(preds_index.data, input_length.data)

        for i in range(B):
            op_str = get_op_seq(preds_str[i], labels[i])
            label, pred_str, op_str = list(map(list, [labels[i], preds_str[i], op_str]))

            sample_input = kl_inp[i, :, :]  # 26 37(ignore the separation blank class)
            sample_input = sample_input.log_softmax(1)  # softmax  -> linear transformation?
            # i_sum = sample_input.sum(1).unsqueeze(1).expand(26,37)
            # sample_input = sample_input / i_sum

            # sample_length = target_length[i]
            sample_pos = input_pos[i]  # target_length (for selecting the probability of the corresponding pos
            selected_pred = sample_input[sample_pos]
            GT_target = target[i, :target_length[i]].data.cpu().numpy().astype(int).tolist()

            # pred_align = selected_pred
            pred_align = torch.zeros([len(op_str), (C-1)])


            align_index = []
            m = 0

            for j, op in enumerate(op_str):
                if op == '#':
                    align_index.append(selected_pred[None, m, :])
                    m = m + 1
                elif op == 's':
                    align_index.append(selected_pred[None, m, :])
                    m = m + 1
                elif op == 'i':
                    align_index.append(torch.full([1, C], 1 / nclass).log_softmax(1).cuda())
                elif op == 'd':
                    align_index.append(selected_pred[None, m, :])
                    GT_target.insert(j,0)
                    m = m + 1
            try:
                pred_align = torch.cat(align_index, 0)
            except:
                continue
            
            forth_target = [C-1] * len(GT_target)
            forth_target[1:] = GT_target[:-1]

            if len(pred_align):
                # for j in range(T):
                smoothing = 1 - math.pow((1 - self.smoothing), 1 / len(pred_align))
                step_matric = self.matric[forth_target, GT_target]

                SLS_distri = torch.from_numpy(step_matric).float().cuda() #(3,38)
                eps = SLS_distri.new_ones(SLS_distri.size()).float().cuda() * (1e-10)
                kldiv_loss += smoothing * (SLS_distri.sum(1) * self.kldiv((pred_align), (SLS_distri + eps)).mean(1)).mean()

        loss = ctc_loss + kldiv_loss
        # loss = (1 - self.smoothing) * ctc_loss + self.smoothing * kldiv_loss

        return loss

class CaslsChineseCTCLoss(nn.Module):
    def __init__(self, matric, converter, blank=0, alpha=0.05,**kwargs):
        super(CaslsChineseCTCLoss, self).__init__()
        self.smoothing = alpha
        self.nclass = len(converter.character) + 1
        self.matric = np.full((self.nclass, self.nclass), 0).astype(int)
        self.matric[1:, 1:] = matric
        self.matric = torch.tensor(self.matric).cuda()
        self.ctc = nn.CTCLoss(reduction='mean', blank=blank, zero_infinity=True)
        self.kldiv = nn.KLDivLoss(reduction='none')
        self.converter = converter

    def forward(self, input, target, input_length, target_length, _, labels,*args):
        _, preds_index = input.max(2)
        preds_str, input_pos = self.converter.decode(preds_index.data, input_length.data)
        ctc_input = input.log_softmax(2).permute(1, 0, 2)
        ctc_loss = self.ctc(ctc_input, target, input_length, target_length)
        batch_size, _, nclass = input.size()  

        kldiv_loss = 0
        
        for i in range(batch_size):
            op_str = get_op_seq(preds_str[i], labels[i])
            op_str = list(op_str)
            sample_input = input[i, :, :].log_softmax(1)
            #sample_target = input_index[i]
            sample_pos = input_pos[i]

            selected_pred = sample_input[sample_pos]

            GT_target = target[i, :target_length[i]].tolist()

            m = 0
            align_index = []

            for j, op in enumerate(op_str):
                if op == '#':
                    align_index.append(selected_pred[None, m, :])
                    m = m + 1
                elif op == 's':
                    align_index.append(selected_pred[None, m, :])
                    m = m + 1
                elif op == 'i':
                    align_index.append(torch.full([1, nclass], 1 / self.nclass).log_softmax(1).cuda())
                elif op == 'd':
                    align_index.append(selected_pred[None, m, :])
                    GT_target.insert(j,0)
                    m = m + 1

            if len(align_index) == 0: continue
            pred_align = torch.cat(align_index, 0)

            forth_target = [nclass - 1] * len(GT_target)
            forth_target[1:] = GT_target[:-1]
            need_smoothed = self.matric[forth_target, GT_target].view(-1, 1)
            smoothing = (1 - torch.pow((1 - self.smoothing), 1 / target_length[i])).view(-1, 1)
            weight = torch.full_like(pred_align, 1 / nclass)
            kldiv_loss += (need_smoothed * smoothing * self.kldiv(pred_align, weight)).mean()


        loss = ctc_loss + kldiv_loss
        return loss