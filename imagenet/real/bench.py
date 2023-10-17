import os
import torch
import numpy as np
from imagenet.data import load_label_fixed, compare_label


class LoadBench():
    def __init__(self):
        self.ans_clps, self.ans_clps_r = load_label_fixed()
        self.n = len(self.ans_clps)

    def load_clean(self):
        """Compare original imagenet validation set and ReaL.

        Returns:
            clean: unchanged label
            diff: disjoint labels
            none: no label assigned in ReaL
        """
        clean_l = np.array([False] * self.n)
        diff_l = np.array([False] * self.n)
        none_l = np.array([False] * self.n)

        for i in range(self.n):
            clean, inc, diff, none = compare_label(self.ans_clps[i], self.ans_clps_r[i])
            if (len(self.ans_clps[i]) == 1) and clean:
                clean_l[i] = True
            diff_l[i] = diff
            none_l[i] = none

        return clean_l, diff_l, none_l


if __name__ == '__main__':
    bench = LoadBench()
    clean, diff, none = bench.load_clean()
    print("# clean label: ", sum(clean))
    print("# diff label: ", sum(diff))
    print("# none label: ", sum(none))

    torch.save({
        'clean': clean,
        'diff': diff,
        'none': none,
    }, './val_category.pt')
