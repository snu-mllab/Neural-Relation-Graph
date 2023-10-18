"""
The original code is created by Jang-Hyun Kim.
GitHub Repository: https://github.com/snu-mllab/Neural-Relation-Graph
"""
import os
import torch
from relation import get_relation
from metric import LabelNoise, cal_auc_ap, hist
from feature import normalize


class LoadData(LabelNoise):
    """ Load inputs for relation graph: features, probs, labels, noisy set index
    """
    def __init__(self, args):
        super().__init__()
        name = f'{args.task_name}/{args.model_name}/epoch_{args.epoch}'
        self.path = os.path.join(args.cache_dir, name)

        if args.task_name == 'sst2':
            self.nclass = 2
        elif args.task_name == 'mnli':
            self.nclass = 3

        self._load_feat(args)
        self._load_noisy_label(args)

        # Calculate scores for baselines
        self.cal_tracin()
        self.cal_margin()

        # normalize feature
        self.feat = normalize(self.feat)

        # Subsample training set
        if args.hop > 1:
            self._sub_sample(args.hop)

        print(f"feature: {list(self.feat.shape)}, "
              f"# noisy label: {self.noise.sum()} ({self.noise.sum()/len(self.feat)*100:.1f}%)\n")

    def _load_feat(self, args):
        """Load data features and prediction probabilities
        
        Output:
            self.feat (torch.tensor [N, D]): features of data 
            self.prob (torch.tensor [N, C]): probability vectors of data
        """
        feat = torch.load(f"{self.path}/feat_train.pt")
        if args.dtype == "float16":
            feat['feat_cls'] = feat['feat_cls'].half()
            feat['logit'] = feat['logit'].half()

        self.feat = feat['feat_cls'].cuda()
        self.prob = torch.softmax(feat['logit'].cuda(), dim=-1)
        print(f"Load feature from {self.path}")

    def _load_noisy_label(self, args):
        """Load noisy label and its index
        
        Output:
            self.targets (torch.long [N,]): noisy label 
            self.noise (torch.bool [N,]): index of noisy label
        """
        noise_ratio = args.model_name.split('noise')[-1]
        idx = torch.load(
            os.path.join(args.cache_dir, f'{args.task_name}/target{noise_ratio}_large_4.pt'))

        self.targets = torch.tensor(idx['targets']).cuda()
        self.noise = torch.tensor([False] * len(self.targets))
        self.noise[idx['idx_chg']] = True


def get_init_score(data, args, save=None):
    """ Calculate initial label reliability score: Sum_j r(i,j)
    
    Args:
        data : instance of LoadData
    """
    path = os.path.join(data.path, f'graph_{args.kernel}_pow{args.pow}.pt')
    if save is None:
        save = (args.hop == 1) and (args.kernel == 'cos_p')

    if os.path.isfile(path) and save:
        graph = torch.load(path)
    else:
        graph = get_relation(data.feat,
                             data.feat,
                             data.targets,
                             data.targets,
                             data.prob,
                             data.prob,
                             kernel_type=args.kernel,
                             pow=args.pow,
                             chunk=args.chunk,
                             verbose=args.verbose)
        if save:
            torch.save(graph, path)

    score = graph['score'].float()
    return score


def del_edge(data, noise, args):
    """ Calculate sum of relation values regarding noisy subset: Sum_{j\in N} r(i,j)
    
    Args:
        data : instance of LoadData
        noise : Estimated noisy subset
    """
    if noise.sum() > 0:
        graph = get_relation(data.feat,
                             data.feat[noise],
                             data.targets,
                             data.targets[noise],
                             data.prob,
                             data.prob[noise],
                             kernel_type=args.kernel,
                             pow=args.pow,
                             chunk=args.chunk,
                             verbose=args.verbose)

        score_del = graph['score'].float()
    else:
        score_del = 0

    return score_del


def eval_score(score, data, save_hist=False):
    """ Calculate evaluation scores for neural relation graph

    Args:
        score ([N,]): label reliability score
        data : instance of LoadData
    """
    score = score / score.abs().max()
    cal_auc_ap(data.noise, -score, name=f'relation')
    if save_hist:
        hist(data.noise, -score, title=f'score_pow{args.pow}')


if __name__ == '__main__':
    # Evaluate neural relation graph for label noise detection
    from argument import args

    data = LoadData(args)
    data.cal_baselines()

    n_iter = 2
    score_orig = get_init_score(data, args, save=False)
    score = score_orig.clone()
    for t in range(1, n_iter):
        noise = (score / score.abs().max() < -args.reg)
        score_del = del_edge(data, noise, args)

        score = score_orig - 2 * score_del

    eval_score(score, data)
    del data
