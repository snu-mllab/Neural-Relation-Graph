""" Real labels evaluator for ImageNet
Paper: `Are we done with ImageNet?` - https://arxiv.org/abs/2006.07159
Based on Numpy example at https://github.com/google-research/reassessed-imagenet
Hacked together by / Copyright 2020 Ross Wightman
"""
import os
import json
import numpy as np


def round_list(arr, c=3):
    arr = np.array(arr)
    arr = np.round(arr, c)

    return arr.tolist()


class RealLabelsImagenet:
    def __init__(self, filenames, real_json='real.json', topk=(1, 5)):
        with open(real_json) as real_labels:
            real_labels = json.load(real_labels)
            real_labels = {
                f'ILSVRC2012_val_{i + 1:08d}.JPEG': labels
                for i, labels in enumerate(real_labels)
            }
        self.real_labels = real_labels
        self.filenames = filenames
        assert len(self.filenames) == len(self.real_labels)

        self.ans = []
        for filename in self.filenames:
            filename = os.path.basename(filename)
            ans = self.real_labels[filename]
            self.ans.append(ans)

        self.sample_idx = 0
        self.nval_sample = 0
        self.topk = topk
        self.maxk = max(self.topk)
        self.is_correct = {k: [] for k in topk}

        self.prob = []

    def add_result(self, prob, pred_batch=None):
        if pred_batch is None:
            _, pred_batch = prob.topk(self.maxk, 1, True, True)

        pred_batch = pred_batch.cpu().numpy()
        for i, pred in enumerate(pred_batch):
            ans = self.ans[i]
            if ans:
                for k in self.topk:
                    self.is_correct[k].append(any([p in ans for p in pred[:k]]))
                self.nval_sample += 1
            else:
                for k in self.topk:
                    self.is_correct[k].append(None)

            self.prob.append(round_list(prob[i][ans].tolist()))
            self.sample_idx += 1

    def get_accuracy(self, k=None):
        if k is None:
            acc = {}
            for k in self.topk:
                is_correct_valid = np.array(self.is_correct[k])
                is_correct_valid = is_correct_valid[is_correct_valid != None]
                acc[k] = float(np.mean(is_correct_valid)) * 100
            return acc
        else:
            is_correct_valid = np.array(self.is_correct[k])
            is_correct_valid = is_correct_valid[is_correct_valid != None]
            return float(np.mean(is_correct_valid)) * 100


if __name__ == '__main__':
    from torchvision.datasets import ImageFolder
    import torch
    import numpy as np
    from imagenet.data import accuracy

    valdir = os.path.join('/ssd_data/imagenet', 'val')
    valset = ImageFolder(valdir, None)

    filenames = [name for name, _ in valset.samples]
    real_labels = RealLabelsImagenet(filenames, real_json='real.json', topk=(1, 5))

    labels = {"orig": valset.targets, "real": real_labels.ans}
    torch.save(labels, './val_label.pt')
