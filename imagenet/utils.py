# original code: https://github.com/eladhoffer/convNet.pytorch/blob/master/preprocess.py

import torch
import numpy as np
try:
    from torchvision.transforms.functional import InterpolationMode
    has_interpolation_mode = True
except ImportError:
    has_interpolation_mode = False
from PIL import Image

_pil_interpolation_to_str = {
    Image.NEAREST: 'nearest',
    Image.BILINEAR: 'bilinear',
    Image.BICUBIC: 'bicubic',
    Image.BOX: 'box',
    Image.HAMMING: 'hamming',
    Image.LANCZOS: 'lanczos',
}
_str_to_pil_interpolation = {b: a for a, b in _pil_interpolation_to_str.items()}

if has_interpolation_mode:
    _torch_interpolation_to_str = {
        InterpolationMode.NEAREST: 'nearest',
        InterpolationMode.BILINEAR: 'bilinear',
        InterpolationMode.BICUBIC: 'bicubic',
        InterpolationMode.BOX: 'box',
        InterpolationMode.HAMMING: 'hamming',
        InterpolationMode.LANCZOS: 'lanczos',
    }
    _str_to_torch_interpolation = {b: a for a, b in _torch_interpolation_to_str.items()}
else:
    _torch_interpolation_to_str = {}


def str_to_pil_interp(mode_str):
    return _str_to_pil_interpolation[mode_str]


def str_to_interp_mode(mode_str):
    if has_interpolation_mode:
        return _str_to_torch_interpolation[mode_str]
    else:
        return _str_to_pil_interpolation[mode_str]


def interp_mode_to_str(mode):
    if has_interpolation_mode:
        return _torch_interpolation_to_str[mode]
    else:
        return _pil_interpolation_to_str[mode]


def l2(x, y):
    d = (x**2).sum(-1).unsqueeze(1) + (
        y**2).sum(-1).unsqueeze(0) - 2 * torch.matmul(x, y.transpose(1, 0))
    d = torch.sqrt(torch.clamp(d, min=0.))
    return d


def accuracy(prob, target, pred=None, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    if pred is None:
        _, pred = prob.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    acc = {}
    cor = {}
    for k in topk:
        correct_k = correct[:k].sum(0) > 0
        cor[k] = correct_k.tolist()
        acc[k] = correct_k.sum().float().mul_(100.0 / batch_size).item()

    return acc, cor


def round_list(arr, c=3):
    # also work for nested list
    arr = np.array(arr)
    arr = np.round(arr, c)

    return arr.tolist()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
