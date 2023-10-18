"""
The original code is created by Jang-Hyun Kim.
GitHub Repository: https://github.com/snu-mllab/Neural-Relation-Graph
"""
import torch
from imagenet.data import load_data
from feature import load_model_classifier, normalize, cal_prob, load_feat, load_feat_ood
from metric import cal_auc_ap


class LoadDataOOD():
    """ Load inputs for relation graph: features, probs (validation and OOD sets)
    """
    def __init__(self, args):
        self.nclass = 1000

        # Load features for validation set and OOD sets
        model, self.classifier, transform = load_model_classifier(args)
        self._load_feat(args, model, transform)

        # calculate baseline scores
        self.cal_baselines()

        if args.hop > 1:
            self.subsample_train_data(args.hop)

        self.feat = normalize(self.feat)
        self.feat_val = normalize(self.feat_val)
        print(f"Total feature: {self.feat_val.shape}")

    def _load_feat(self, args, model, transform):
        """Load data features and prediction probabilities
        
        Output:
            self.feat (torch.tensor [N, D]): features of training data 
            self.prob (torch.tensor [N, C]): probability vectors of training data
            self.feat_val (torch.tensor [N_val+N_ood, D]): features of validation/OOD data 
            self.prob_val (torch.tensor [N_val+N_ood, C]): probability vectors of validation/OOD data 
        """
        trainset, valset = load_data('imagenet', model, transform)
        self.targets = torch.tensor(trainset.targets).cuda()
        self.targets_val = None

        self.feat, self.feat_val = load_feat(args, model, trainset, valset)
        feat_ood = load_feat_ood(args, model, transform)
        self.feat_val = torch.cat([self.feat_val, feat_ood])

        with torch.no_grad():
            self.prob = cal_prob(self.classifier, self.feat)
            self.logit_val = cal_prob(self.classifier, self.feat_val, return_logit=True)
            self.prob_val = torch.softmax(self.logit_val, dim=-1)

    def subsample_train_data(self, hop):
        n = len(self.feat)
        indices = torch.arange(n)[::hop]

        self.feat = self.feat[indices]
        self.prob = self.prob[indices]

    def cal_baselines(self, temp=1.0):
        """ Calculate baseline scores. Larger score means more likely to be OOD.
        """
        self.max_logit = -self.logit_val.max(-1)[0]
        self.max_prob = -self.prob_val.max(-1)[0]
        self.energy = -torch.logsumexp(self.logit_val / temp, dim=1)
        self.cal_kl_matching()
        self.cal_maha()
        self.cal_react()

    def cal_kl_matching(self):
        nclass = 1000
        pred = self.prob.max(-1)[1]
        kl_scores = []
        for c in range(nclass):
            cond = pred == c
            dist = self.prob[cond].mean(0)
            kl = (self.prob_val *
                  (torch.log(self.prob_val + 1e-6) - torch.log(dist + 1e-6))).sum(-1)
            kl_scores.append(kl)
        kl_scores = torch.stack(kl_scores, dim=-1)
        self.kl = kl_scores.min(-1)[0]

    def cal_maha(self):
        nclass = 1000

        maha_scores = []
        for c in range(nclass):
            cond = self.targets == c
            feat_c = self.feat[cond]
            n, d = feat_c.shape

            mu = feat_c.mean(0, keepdim=True)
            dist = feat_c - mu  # n x d
            cov = torch.matmul(dist.transpose(1, 0), dist) / n
            cov = cov + 1e-3 * torch.eye(d).cuda()
            invcov = torch.linalg.inv(cov).to(dtype=feat_c.dtype)  # d x d

            dist = self.feat_val - mu
            a = torch.matmul(dist, invcov)
            score = (a * dist).sum(-1)
            maha_scores.append(score)

        maha_scores = torch.stack(maha_scores, dim=-1)
        self.maha = maha_scores.min(-1)[0]

    def cal_react(self):
        feat_in = self.feat[::100].reshape(-1)
        n_p = int(len(feat_in) * 0.9)
        thres = torch.sort(feat_in)[0][n_p]

        feat_clamp = torch.clamp(self.feat_val, max=thres)
        feat_clamp = self.feat_val
        logit_clamp = cal_prob(self.classifier, feat_clamp, return_logit=True)
        self.react = -torch.logsumexp(logit_clamp, dim=1)


def baselines(data):
    print()
    d = [
        ("MSP", data.max_prob),
        ("Max Logit", data.max_logit),
        ("Mahalanobis", data.maha),
        ("Energy", data.energy),
        ("ReAct", data.react),
        ("KL-Matching", data.kl),
    ]
    for name, value in d:
        print(f"* {name}")
        cal_ood_score(value)


def cal_ood_score(score, n_val=50000):
    ood_idx = torch.tensor([False] * len(score))
    ood_idx[n_val:] = True

    cal_auc_ap(ood_idx, score, name="all")

    names = ['places', 'sun', 'inat', 'dtd']
    for i in range(4):
        # Each ood data except dtd has 10000 samples
        score_ = torch.cat([score[:n_val], score[n_val + i * 10000:n_val + (i + 1) * 10000]])
        ood_idx = torch.tensor([False] * len(score_))
        ood_idx[n_val:] = True

        cal_auc_ap(ood_idx, score_, name=names[i])
    print()


if __name__ == '__main__':
    from argument import args
    from relation import get_relation

    if args.pow != 1:
        print("**Caution** Temperature is not 1, which is not optimal for OOD detection.\n")

    data = LoadDataOOD(args)
    baselines(data)

    # KNN OOD detection
    print(f"* KNN")
    k = int(1000 / args.hop)
    knn = get_relation(data.feat_val,
                       data.feat,
                       None,
                       None,
                       data.prob_val,
                       data.prob,
                       topk=k,
                       kernel_type="cos",
                       pow=args.pow,
                       chunk=args.chunk,
                       verbose=args.verbose)
    score = 1 / (knn[f'top_val'][:, -1].float() + 1e-6)
    cal_ood_score(score)

    # OOD detection with relation graph
    print(f"* Relation graph on {len(data.feat)} training data with pow {args.pow}")
    graph = get_relation(data.feat_val,
                         data.feat,
                         None,
                         None,
                         data.prob_val,
                         data.prob,
                         kernel_type=args.kernel,
                         pow=args.pow,
                         chunk=args.chunk,
                         verbose=args.verbose)
    score = 1 / (graph['score'].float() + 1e-6)
    cal_ood_score(score)

    del data
