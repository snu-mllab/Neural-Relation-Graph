import numpy as np
import torch


# This fixes label collapsing in ImageNet label set
# reference: When does dough become a bagel? Analyzing the remaining mistakes on ImageNet
# https://arxiv.org/abs/2205.04596
def def_label_collapse():
    label_dict = {i: [i] for i in range(1000)}
    for lab in range(1000):
        if lab in [249, 250]:
            label_dict[lab] = [lab, 248]
        elif lab in [836, 837]:
            label_dict[lab] = [836, 837]
        elif lab in [385, 386]:
            label_dict[lab] = [lab, 101]
        elif lab == 504:
            label_dict[lab] = [504, 968]
        elif lab in [638, 639]:
            label_dict[lab] = [638, 639]
        elif lab in [657, 744]:
            label_dict[lab] = [657, 744]
        elif lab in [620, 681]:
            label_dict[lab] = [620, 681]
        elif lab in [664, 782]:
            label_dict[lab] = [664, 782]
        elif lab == 482:
            label_dict[lab] = [482, 848]
        elif lab in [356, 357, 358, 359]:
            label_dict[lab] = [356, 357, 358, 359]
        elif lab == 435:
            label_dict[lab] = [435, 876]
    return label_dict


def str_to_int(str_arr):
    if len(str_arr) > 2:
        str_arr = [int(s.strip()) for s in str_arr[1:-1].split(',')]
        return str_arr
    else:
        return []


def load_label_fixed():
    label_dict = def_label_collapse()
    label = torch.load('./val_label.pt')

    label_clps = [label_dict[a] for a in label['orig']]
    label_clps_r = label['real']
    for i in range(len(label_clps_r)):
        label_fix = []
        for a in label_clps_r[i]:
            label_fix += label_dict[a]
        label_fix = list(set(label_fix))
        label_clps_r[i] = label_fix

    return label_clps, label_clps_r


def valid_labels(labels):
    val_list = []
    for lab in labels:
        if len(lab) > 0:
            val_list.append(True)
        else:
            val_list.append(False)

    val_list = np.array(val_list)

    return val_list


def accuracy(logits, labels, preds=None, topk=1):
    if preds is None:
        _, preds = torch.topk(logits, k=topk, dim=-1)
    valid = valid_labels(labels)

    cor_list = []
    for i, lab in enumerate(labels):
        if len(lab) > 0:
            cor = any([p in lab for p in preds[i][:topk]])
        else:
            cor = None
        cor_list.append(cor)

    cor_list = np.array(cor_list)
    acc = cor_list[valid].sum() / valid.sum()

    return acc, cor_list


def compare_label(lab1, lab2):
    clean = False  # label unchanged
    inc = False  # multi-label (including original label)
    diff = False  # disjoint labels
    none = False  # unknown labels

    if (len(lab1) == 0) or (len(lab2) == 0):
        none = True
    else:
        # because original imagenet only allows for single label
        # only three categories are possible (even after considering label collapsing)
        diff = True
        for a in lab1:
            if a in lab2:
                diff = False

        if not diff:
            if len(lab1) == len(lab2):
                clean = True
            else:
                inc = True

    return clean, inc, diff, none
