
import math
import os
import time
import string
import argparse
import re

import numpy as np
from typing import Tuple, List
from collections import defaultdict

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

computer_ctc_conf = torch.nn.CTCLoss(reduction='none', blank=0, zero_infinity=True)

def log_add(args: List[int]) -> float:
    """
    Stable log add
    """
    if all(a == -float('inf') for a in args):
        return -float('inf')
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp

def calculate_conf(ctc_probs, hyps, beam_size, batch_size=1):
    preds_size = torch.IntTensor([ctc_probs.size(0)] * batch_size * beam_size)
    length_for_loss = torch.IntTensor([len(hyp[0]) for hyp in hyps])
    ctc_probs = ctc_probs.unsqueeze(1).repeat(1, batch_size * beam_size, 1)
    text_for_loss = torch.zeros((batch_size * beam_size, 50))
    for idx, hyp in enumerate(hyps):
        text_for_loss[idx, :len(hyp[0])] = torch.IntTensor(hyp[0])
    loss_ctc = computer_ctc_conf(ctc_probs, text_for_loss, preds_size, length_for_loss)
    scores = torch.exp(-loss_ctc).tolist()
    return scores

def ctc_prefix_beam_search(
    preds: torch.Tensor,
    beam_size: int,
):
    batch_size = preds.shape[0]
    maxlen = preds.size(1)

    ctc_probs = torch.log_softmax(preds, dim=-1)  
    ctc_probs = ctc_probs.squeeze(0)
    cur_hyps = [(tuple(), (0.0, -float('inf')))]

    for t in range(0, maxlen):
        logp = ctc_probs[t]  
        next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
        top_k_logp, top_k_index = logp.topk(beam_size)  
        for s in top_k_index:
            s = s.item()
            ps = logp[s].item()
            for prefix, (pb, pnb) in cur_hyps:
                last = prefix[-1] if len(prefix) > 0 else None
                if s == 0:  
                    n_pb, n_pnb = next_hyps[prefix]
                    n_pb = log_add([n_pb, pb + ps, pnb + ps])
                    next_hyps[prefix] = (n_pb, n_pnb)
                elif s == last:
                    n_pb, n_pnb = next_hyps[prefix]
                    n_pnb = log_add([n_pnb, pnb + ps])
                    next_hyps[prefix] = (n_pb, n_pnb)

                    n_prefix = prefix + (s, )
                    n_pb, n_pnb = next_hyps[n_prefix]
                    n_pnb = log_add([n_pnb, pb + ps])
                    next_hyps[n_prefix] = (n_pb, n_pnb)
                else:
                    n_prefix = prefix + (s, )
                    n_pb, n_pnb = next_hyps[n_prefix]
                    n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                    next_hyps[n_prefix] = (n_pb, n_pnb)

        next_hyps = sorted(next_hyps.items(), key=lambda x: log_add(list(x[1])), reverse=True)
        cur_hyps = next_hyps[:beam_size]

    hyps = [[y[0], math.exp(log_add([y[1][0], y[1][1]]))] for y in cur_hyps]
    scores = calculate_conf(ctc_probs, hyps, beam_size, batch_size)

    return hyps, scores




    # scores = []
    # for hyp in hyps:
    #     preds_size = torch.IntTensor([ctc_probs.size(0)] * batch_size)
    #     length_for_loss = torch.tensor((len(hyps[0][0]),))
    #     text_for_loss = torch.zeros((batch_size, 50))
    #     text_for_loss[0, :len(hyps[0][0])] = torch.IntTensor(hyps[0][0])
    #     loss_ctc = computer_ctc_conf(ctc_probs.unsqueeze(1), text_for_loss, preds_size, length_for_loss)
    #     score = torch.exp(-loss_ctc)
    #     scores.append(score)