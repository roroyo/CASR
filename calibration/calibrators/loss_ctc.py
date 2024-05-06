import math
from multiprocessing import reduction
import os


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from calibration.get_confusion_matrices import get_op_seq
from utils import CTCLabelConverter

class CTC_Loss(nn.Module):
    def __init__(self, 
                 **kwargs):
        super().__init__()
        self.loss_func = torch.nn.CTCLoss(zero_infinity=True)
    def forward(self, preds, text, preds_size, length, *args):
        preds = preds.log_softmax(2).permute(1, 0, 2)
        loss = self.loss_func(preds, text, preds_size, length)
        return loss


class CTCLabelSmoothLoss(nn.Module):
    def __init__(self, 
                 opt, 
                 **kwargs):
        super(CTCLabelSmoothLoss, self).__init__()
        self.smoothing = opt.beta
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


class CTCCASLS_v3(nn.Module):
    def __init__(self, 
                 opt, 
                 **kwargs):
        super(CTCCASLS_v3, self).__init__()
        self.smoothing = opt.beta
        self.matric = np.zeros((38,38,38))
        self.matric[0] = np.eye(38)
        self.matric[:,0,0] = 1
        self.matric[1:,1:,1:] = np.load('confusion_matrices/CTC_confusion_matrices.npy')[1:, 1:, 1:]

        self.converter = CTCLabelConverter(opt.character)
        self.ctc = nn.CTCLoss(reduction='mean', blank=0, zero_infinity=True)
        self.kldiv = nn.KLDivLoss(size_average=True, reduce=False)
        
    def forward(self, input, target, input_length, target_length, labels, *args):
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
        
        preds_str, input_pos = self.decode(preds_index.data, input_length.data)

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
                if op == '-':
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
    
    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        positions = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            position = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.converter.character[t[i]])
                    position.append(i)
            text = ''.join(char_list)

            texts.append(text)
            positions.append(position)
        return texts, positions