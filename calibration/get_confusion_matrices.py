import json
import argparse
import operator
import os
import numpy as np

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
                edit_flag.append('-')
    return ''.join(edit_flag)

def get_metrics(data, classes, Th):
    matric = np.zeros((len(classes), len(classes), len(classes)), dtype=np.float16)
    hot = np.zeros((len(classes), len(classes), len(classes)))
    for d in data:
        gt, pred = d[2], d[3]
        op_str = get_op_seq(pred, gt)
        gt, pred, op_str = list(map(list, [gt, pred, op_str]))

        fore_p = '∏' # denotes blank token
        for i in range(len(op_str)):
            op = op_str[i]
            if op == 's':
                gg = gt.pop(0)
                pp = pred.pop(0)
            elif op == '-':
                gg = gt.pop(0)
                pp = pred.pop(0)
            elif op == 'd':
                gg = '∏'
                pp = pred.pop(0)
            elif op == 'i':
                gg = gt.pop(0)
                pp = '∏'
            matric[classes.index(fore_p), classes.index(gg), classes.index(pp)] += 1
            if op != 'd':
                fore_p = gg

    for i in range(len(classes)):
        for j in range(len(classes)):
            matric[i, j, :] = matric[i, j, :] / (matric[i, j, :].sum() + 10e-6)
            temp = np.zeros_like(matric[0, 0, :])
            if matric[i, j, j] > 1 - Th:
                temp[j] = 1.0
                matric[i, j, :] = temp

    return matric

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--support_set_res_path', type=str, required=True, help='path to evaluation result of support set')
    parser.add_argument('--saved_path', type=str, required=True, help="path to confusion matrices")
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--threshold', type=float, default=0.05, help='probability threshold of the error-prone classes')
    opt = parser.parse_args()
    os.makedirs(f'{opt.saved_path}', exist_ok=True)

    with open(opt.support_set_res_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if opt.sensitive:
        with open('data_lmdb_release/charset_94.txt','r') as f:
            classes = sorted(f.read())
    else:
        with open('data_lmdb_release/charset_36.txt','r') as f:
            classes = sorted(f.read())

    if 'CTC' in opt.Prediction:
        classes = ['[CTCblank]'] + classes
    else:
        classes = ['[GO]', '[s]'] + classes
    
    matric = get_metrics(data, classes, Th=opt.threshold)
    np.save(f'{opt.saved_path}/{opt.Prediction}_confusion_matrices.npy', matric)


