"""
The original code is created by Jang-Hyun Kim.
GitHub Repository: https://github.com/snu-mllab/Neural-Relation-Graph
"""
import os
import torch
from imagenet.data import load_data
from imagenet.train import validate_model
from feature import load_model_classifier, normalize, cal_prob
from metric import LabelNoise


class LoadData(LabelNoise):
    """ Load inputs for relation graph: features, probs, labels, noisy set index
    """
    def __init__(self, args):
        super().__init__()
        self.path = args.cache_dir

        self.nclass = 1000
        self._load_feat(args)

        # Load index of noisy ImageNet validation sample (Beyer et al. Are we done with imagenet?, 2020)
        cat = torch.load("./imagenet/real/val_category.pt")
        self.noise = torch.tensor(~cat['clean'])

        # Calculate scores for baselines
        self.cal_tracin()
        self.cal_margin()

        # normalize feature
        self.feat = normalize(self.feat)

        print(f"feature: {list(self.feat.shape)}, "
              f"# noisy label: {self.noise.sum()} ({self.noise.sum()/len(self.feat)*100:.1f}%)\n")

    def _load_feat(self, args):
        """Load validation data features and prediction probabilities
        
        Output:
            self.feat (torch.tensor [N, D]): features of data 
            self.prob (torch.tensor [N, C]): probability vectors of data
        """
        model, classifier, transform = load_model_classifier(args)
        _, valset = load_data('imagenet', model, transform)
        self.targets = torch.tensor(valset.targets).cuda()

        tag = f"_{args.epoch}" if args.epoch is not None else ""
        file = f"{self.path}/features/feat_val{tag}.pt"

        if not os.path.isfile(file):
            _, self.feat = validate_model(args, model, valset)
            if args.save:
                torch.save(self.feat.cpu(), file)
        else:
            self.feat = torch.load(file)

        if args.dtype == "float16":
            self.feat = self.feat.half()

        # calculate probability vectors
        with torch.no_grad():
            self.feat = self.feat.cuda()
            self.prob = cal_prob(classifier, self.feat)

        print(f"Load feature from {self.path}")


def eval_score(score, data):
    """ Calculate evaluation scores for neural relation graph

    Args:
        score ([N,]): label reliability score
        data : instance of LoadData
    """
    score = score[:50000]
    score = score / score.abs().max()
    cal_auc_ap(data.noise, -score, name=f'relation')


if __name__ == '__main__':
    # Evaluate neural relation graph for label noise detection
    from argument import args
    from detect import cal_auc_ap, get_init_score, del_edge

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
