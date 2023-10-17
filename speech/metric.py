import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.nn import KLDivLoss
from math import ceil
from sklearn import metrics


class LabelNoise():
    """ Calculate scores for label noise detection
    """
    def __init__(self):
        self.feat = None  # Feature of target data [N, D]
        self.prob = None  # Probability vector of target data [N, C]

        self.targets = None  # Noisy label [N,] (torch.long)
        self.noise = None   # Index of noisy label [N,] (torch.bool)

        self.nclass = None  
        self.path = './'

        self.baselines = {}

    def _sub_sample(self, hop):
        self.feat = self.feat[::hop]
        self.targets = self.targets[::hop]
        self.prob = self.prob[::hop]
        self.noise = self.noise[::hop]

        for k, v in self.baselines.items():
            self.baselines[k] = v[::hop]

        self.hop = hop
        print(f"Sumsample data with hop {hop}")

    def cal_tracin(self):
        # Should use unnormalized feature
        scores = []
        nc = 10000
        step = ceil(len(self.feat) / nc)
        for i in range(step):
            feat_norm = (self.feat[nc * i:nc * (i + 1)]**2).sum(-1)

            target = F.one_hot(self.targets[nc * i:nc * (i + 1)], num_classes=self.nclass).float()
            err = target - self.prob[nc * i:nc * (i + 1)]
            err_norm = (err**2).sum(-1)

            s = feat_norm * err_norm
            scores.append(s)

        scores = torch.cat(scores)
        assert len(scores) == len(self.feat)

        self.baselines['tracin'] = scores

    def cal_margin(self):
        # Higher scores means that the label is more likely to be noisy
        ans_p = torch.gather(self.prob, 1, self.targets[:, None]).squeeze()
        prob_k, _ = self.prob.topk(2, 1, True, True)
        margin = ans_p - prob_k[:, 0]
        margin[margin == 0] = (ans_p - prob_k[:, 1])[margin == 0]

        self.baselines['loss'] = -torch.log(ans_p + 1e-6)
        self.baselines['margin'] = -margin
        entropy = -(self.prob * torch.log(self.prob + 1e-6)).sum(-1)
        self.baselines['entropy'] = entropy
        self.baselines['least-confidence'] = -prob_k[:, 0]
        self.baselines['cwe'] = -ans_p / entropy

    def cal_jsd(self):
        scores = []
        nc = 10000
        step = ceil(len(self.feat) / nc)

        kld = KLDivLoss(reduction='none')
        for i in range(step):
            prob = self.prob[nc * i:nc * (i + 1)]
            target = F.one_hot(self.targets[nc * i:nc * (i + 1)], num_classes=self.nclass).float()

            M = 0.5 * (prob + target) + 1e-6
            s = 0.5 * (kld(torch.log(prob + 1e-6), M) + kld(torch.log(target + 1e-6), M))

            s = s.sum(-1)
            scores.append(s)

        scores = torch.cat(scores)
        assert len(scores) == len(self.feat)
        self.baselines['jsd'] = scores

    def cal_baselines(self, save=False):
        """ Calculate histograms for baselines

        Args:
            data (_type_): _description_
            args (_type_): _description_
        """
        for k in sorted(self.baselines.keys()):
            cal_auc_ap(self.noise, self.baselines[k], name=k)

        if save:
            path = os.path.join(self.path, f'score_baseline.pt')
            torch.save(self.baselines, path)

        print()


def cal_auc_ap(target, score, th=None, name="", verbose=True):
    """ AUROC, AP, TNR95 for label noise detection

    Args:
        target (torch tensor [N,]): An array indicates whether a label is noisy (1) or not (0).
        score (torch tensor [N,]): A score array where higher score means that the label is more likely to be noisy.
        th (float, optional): A threshold value for calculating F1 score. Defaults to None.
        name (str, optional): Header tag.
    """
    target = target.cpu().numpy()
    score = score.cpu().numpy()

    fpr, tpr, _ = metrics.roc_curve(target, score)
    roc = metrics.auc(fpr, tpr)
    ap = metrics.average_precision_score(target, score)

    idx = (tpr < 0.95).sum()
    tnr95 = 1 - fpr[idx]

    text = f"{name:16s}| ROC, AP, TNR95: {roc:.3f}, {ap:.3f}, {tnr95:.3f}"

    if th is not None:
        pred = -score < th
        acc = (pred & target).sum() / pred.sum()
        recall = (pred & target).sum() / (target).sum()
        f1 = 2 / (1 / acc + 1 / recall)

        text += f", F1: {f1:.3f} (acc: {acc:.3f}, recall: {recall:.3f}, pred: {pred.sum()})"

    if verbose:
        print(text)
    
    return roc, ap, tnr95


def hist(target, score, title='', bins=50, log=True):
    """Generating histogram for label reliability score and PR curve

    Args:
        target (torch tensor [N,]): An array indicates whether a label is noisy (1) or not (0).
        score (torch tensor [N,]): A score array where higher score means that the label is more likely to be noisy.
        title (str, optional): Title for the figure. Defaults to ''.
        bins (int, optional): Number of bins for histogram. Defaults to 50.
        log (bool, optional): Log scale or not. Defaults to True.
    """

    score = score.cpu().numpy()
    target = target.cpu().numpy()
    precision, recall, _ = metrics.precision_recall_curve(target, score)

    ncol = 4
    fig, axes = plt.subplots(1, ncol, figsize=(ncol * 3, 3))
    axes[0].hist(-score, bins=bins, log=log)
    axes[1].hist(-score[~target], bins=bins, log=log)
    axes[2].hist(-score[target], bins=bins, log=log)

    axes[0].set_title('Full')
    axes[1].set_title('clean')
    axes[2].set_title('noisy')

    axes[3].plot(recall[:-50], precision[:-50])
    axes[3].grid()
    axes[3].set_xlabel("recall")
    axes[3].set_ylabel("precision")
    axes[3].set_ylim([0, 1.05])
    axes[3].set_title('PR curve')

    fig.tight_layout(pad=2.0)
    os.makedirs('./images', exist_ok=True)
    plt.savefig(f'./images/{title}.png')
    plt.close()
