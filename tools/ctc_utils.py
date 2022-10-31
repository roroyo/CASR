
import math
import os
import time
import string
import argparse
import re
import operator

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

def _edit_dist_init(len1, len2):
    lev = []
    for i in range(len1):
        lev.append([0] * len2)  # initialize 2D array to zero
    for i in range(len1):
        lev[i][0] = i  # column 0: 0,1,2,3,4,...
    for j in range(len2):
        lev[0][j] = j  # row 0: 0,1,2,3,4,...
    return lev

def _edit_dist_step(lev, i, j, s1, s2, substitution_cost=1, transpositions=False):
    c1 = s1[i - 1]
    c2 = s2[j - 1]

    # skipping a character in s1
    a = lev[i - 1][j] + 1
    # skipping a character in s2
    b = lev[i][j - 1] + 1
    # substitution
    c = lev[i - 1][j - 1] + (substitution_cost if c1 != c2 else 0)

    # transposition
    d = c + 1  # never picked by default
    if transpositions and i > 1 and j > 1:
        if s1[i - 2] == c2 and s2[j - 2] == c1:
            d = lev[i - 2][j - 2] + 1

    # pick the cheapest
    lev[i][j] = min(a, b, c, d)

def edit_distance(s1, s2, substitution_cost=1, transpositions=False):
    # set up a 2-D array
    len1 = len(s1)
    len2 = len(s2)
    lev = _edit_dist_init(len1 + 1, len2 + 1)

    # iterate over the array
    for i in range(len1):
        for j in range(len2):
            _edit_dist_step(
                lev,
                i + 1,
                j + 1,
                s1,
                s2,
                substitution_cost=substitution_cost,
                transpositions=transpositions,
            )
    align_list = _edit_dist_backtrace(lev)
    return lev[len1][len2], align_list

def _edit_dist_backtrace(lev):
    i, j = len(lev) - 1, len(lev[0]) - 1
    alignment = [(i, j)]

    while (i, j) != (0, 0):
        directions = [
            (i - 1, j),  # skip s1
            (i, j - 1),  # skip s2
            (i - 1, j - 1),  # substitution
        ]

        direction_costs = (
            (lev[i][j] if (i >= 0 and j >= 0) else float("inf"), (i, j))
            for i, j in directions
        )
        _, (i, j) = min(direction_costs, key=operator.itemgetter(0))

        alignment.append((i, j))
    return list(reversed(alignment))


def get_op_seq(pred, gt):
    _, align_list = edit_distance(pred, gt, 2, True)
    edit_flag = []
    for k in range(1, len(align_list)):
        pre_i, pre_j = align_list[k - 1]
        i, j = align_list[k]

        if i == pre_i:  ## skip gt char, p need being inserted
            edit_flag.append('i')
        elif j == pre_j:  ## skip p char, p need being deleted
            edit_flag.append('d')
        else:
            if pred[i - 1] != gt[j - 1]:  ## subsitution
                edit_flag.append('s')
            else:  ## correct
                edit_flag.append('#')
    return ''.join(edit_flag)


    # scores = []
    # for hyp in hyps:
    #     preds_size = torch.IntTensor([ctc_probs.size(0)] * batch_size)
    #     length_for_loss = torch.tensor((len(hyps[0][0]),))
    #     text_for_loss = torch.zeros((batch_size, 50))
    #     text_for_loss[0, :len(hyps[0][0])] = torch.IntTensor(hyps[0][0])
    #     loss_ctc = computer_ctc_conf(ctc_probs.unsqueeze(1), text_for_loss, preds_size, length_for_loss)
    #     score = torch.exp(-loss_ctc)
    #     scores.append(score)