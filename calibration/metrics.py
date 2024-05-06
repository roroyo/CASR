import os
import json
import argparse
import matplotlib
import numpy as np 
import pandas as pd
from zhconv import convert
from matplotlib import pyplot as plt
matplotlib.use('Agg')
# from utils import equal, equal_ed
# from tools.correctness_factory import correct_factory
# from correctness_factory import correct_factory

def full2half(ustring):
    rstring = ''
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif inside_code >= 65281 and inside_code <= 65374:
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

def equal(pred, gt):
    pred = convert(full2half(pred.lower().replace(' ','')), 'zh-hans')
    gt = convert(full2half(gt.lower().replace(' ','')), 'zh-hans')
    if pred == gt:
        return True
    else:
        return False


def ACE(data, bin_num, correct_fn=None, vis=False, save_pth='reliability_diagrams', prefix='', testing=False):
    '''
    data: [
        [
            word_confidence, 
            [char_conf1, char_conf2,...,char_conf3],
            pred_str,
            gt_str,
        ], 
        ...
    ]
    bin_num: Num of Bin to calculate ECE
    '''
    N = len(data)
    n_per_bin = N//bin_num

    correct_bin = [0]*bin_num
    width_bin = [0]*bin_num
    min_p_bin = [0]*bin_num
    prob_bin = [0]*bin_num
    total_bin = [0]*bin_num

    data = sorted(data, key=lambda x: x[0])
    Brier = 0
    # NLL = 0
    # for n in range(N):
    #     if data[n][2] == data[n][3]:
    #         P_log = -np.log(data[n][1][:-1])
    #         NLL += np.sum(P_log)

    for i in range(bin_num):
        if i != bin_num-1:
            ds = data[i*n_per_bin:(i+1)*n_per_bin]
        else:
            ds = data[i*n_per_bin:]

        if correct_fn is None:
            ds_correct = np.array(list(map(lambda x: int(x[2]==x[3].lower()), ds)))
            correct_bin[i] = np.sum(ds_correct)
        else:
            ds_correct = np.array(list(map(lambda x: int(equal(x[2], x[3])), ds)))
            correct_bin[i] = np.sum(ds_correct)

        ds_conf = np.array(list(map(lambda x: x[0], ds)))
        Brier += np.sum(np.power(ds_correct-ds_conf, 2))
        
        min_p_bin[i] = ds[0][0]
        width_bin[i] = ds[-1][0] - ds[0][0]
        prob_bin[i] = sum(list(map(lambda x: x[0], ds)))
        total_bin[i] = len(ds)

    correct_bin = np.array(correct_bin)
    prob_bin = np.array(prob_bin)
    total_bin = np.array(total_bin)
    width_bin = np.array(width_bin)

    acc_bin = correct_bin/total_bin
    conf_bin = prob_bin/total_bin
    # Brier = Brier/bin_num

    ticks = np.linspace(0, 0.90, 10)
    ticks = np.append(ticks, np.linspace(0.90,0.99, 10))
    ticks = np.append(ticks, np.linspace(0.99,0.999, 10))
    ticks = np.append(ticks, np.linspace(0.999,1, 10))

    if vis:
        plt.figure(0, clear=True)
        plt.plot(conf_bin, conf_bin, 'r-')
        plt.plot(conf_bin, acc_bin, 'b-')
        plt.bar(conf_bin, acc_bin, 0.005*width_bin*100)
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.grid(True)
    
    record_str = '|     score     | total |correct|  acc  |\n'
    record_str +='|---------------|-------|-------|-------|\n'
    for i,(s, t, c, a) in enumerate(zip(min_p_bin, total_bin, correct_bin, acc_bin)):
        if i!=bin_num-1:
            record_str += f'|{s:0.5f}~{min_p_bin[i+1]:0.5f}|{int(t):7d}|{int(c):7d}|{a:0.5f}|\n'
        else:
            record_str += f'|{s:0.5f}~{1.:0.5f}|{int(t):7d}|{int(c):7d}|{a:0.5f}|\n'
    
    CE = np.abs(conf_bin-acc_bin)
    ECE = np.sum(CE*total_bin/np.sum(total_bin))
    ACC = np.sum(correct_bin)/np.sum(total_bin)
    pzhCE = conf_bin-acc_bin
    pzhECE = np.sum(pzhCE*total_bin/np.sum(total_bin))
    Brier = Brier/np.sum(total_bin)
    record_str += f'ACC:{ACC:0.4f}\n'
    record_str += f'ECE:{ECE:0.4f}\n'
    record_str += f'BrierScore:{Brier:0.4f}\n'
    # record_str += f'NLL:{NLL:0.4f}\n'

    
    conf_str = ''
    dist_str = ''
    flag_str = ''
    for conf, dist, i in zip(conf_bin.tolist(), pzhCE.tolist(), pzhCE > 0):
        conf_str += f'|{conf:.5f}'
        dist_str += f'|{dist:.5f}'
        if i:
            flag_str += '+'
        else:
            flag_str += '-'

    if testing: print(record_str)
    if vis:
        #print(record_str)
        os.makedirs(f'{save_pth}', exist_ok=True)
        plt.text(0,1,f'ECE:{ECE:0.4f}/ACC:{ACC:0.4f}/BrierScore:{Brier:0.4f}')
        plt.savefig(os.path.join(save_pth, f'{prefix}bin{bin_num}_ECE{ECE:0.5f}.jpg'))

    return ECE, ACC, Brier, pzhECE, (flag_str, f'{conf_str}\n{dist_str}')


def ECE(data, bin_num, correct_fn=None):
    bin_boundaries = np.linspace(0, 1, bin_num + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    confidences, _, predictions, labels = list(map(np.array, zip(*data)))
    accuracies = np.array(list(map(lambda x: int(x[0] == x[1]), zip(predictions, labels))))

    ece = np.zeros(1)
    acc_bin = []
    conf_bin = []
    bin_score = np.array([])
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower.item()) * (confidences <= bin_upper.item())
        prop_in_bin = in_bin.mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_score = np.append(bin_score, np.abs(avg_confidence_in_bin - accuracy_in_bin))
            acc_bin.append(accuracy_in_bin)
            conf_bin.append(avg_confidence_in_bin)

    return ece.item(), np.max(bin_score)

def load_json(path):
    with open(path,'r') as f:
        data = json.load(f)
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_dir', type=str, default='results/train-macsmn/IIIT5K-SVT-SVTP-IC13-IC15-CUTE-alignment', help='Where to load json file')
    parser.add_argument('--rd_output_dir', type=str, default='./Reliability_Diagram')
    parser.add_argument('--save_path', type=str, default='results.xlsx')
    opt = parser.parse_args()

    writer = pd.ExcelWriter(os.path.join(opt.json_dir, opt.save_path))
    sheets = dict()
    for curDir, dirs, files in os.walk(opt.json_dir):
        files = sorted([file for file in files if file.endswith('json')])
        if files:
            content = []
            for file in files:
                data = load_json(os.path.join(curDir, file))
                print(len(data))
                ace, acc, brier, _, _ = ACE(data, 15, vis=True, save_pth=curDir, prefix=file.split('.')[0]+'_')
                ece, mce = ECE(data, 15)
                content.append(100 * np.array([acc, ece, ace, mce, brier]))
            sheets["_".join(curDir.split('/'))] = (content, files)
    for sheet_name, (content, files) in sheets.items():
        df = pd.DataFrame(np.around(content, 4), index = files, columns=['Acc', 'ECE', 'ACE', 'MCE', 'BS'])
        df.to_excel(writer, sheet_name=sheet_name)
    writer.save()
    writer.close()
