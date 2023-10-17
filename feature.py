import os
import torch
from math import ceil
from imagenet.train import validate_model
from imagenet.data import load_data
from models.load import load_model


def normalize(feat, nc=50000):
    with torch.no_grad():
        split = ceil(len(feat) / nc)
        for i in range(split):
            feat_ = feat[i * nc:(i + 1) * nc]
            feat[i * nc:(i + 1) * nc] = feat_ / torch.sqrt((feat_**2).sum(-1)).reshape(-1, 1)

    return feat


def load_feat(args, model, trainset, valset):
    tag = f"_{args.epoch}" if args.epoch is not None else ""

    path_val = f"{args.cache_dir}/features/feat_val{tag}.pt"
    if not os.path.isfile(path_val):
        model.cuda()
        results, feat_val = validate_model(args, model, valset)
        torch.save(results, f"{args.cache_dir}/features/result_val{tag}.pt")
        torch.save(feat_val.cpu(), path_val)
    else:
        feat_val = torch.load(path_val)

    path_train = f"{args.cache_dir}/features/feat_train{tag}.pt"
    if not os.path.isfile(path_train):
        model.cuda()
        results, feat_train = validate_model(args, model, trainset)
        torch.save(results, f"{args.cache_dir}/features/result_train{tag}.pt")
        torch.save(feat_train.cpu(), path_train)
    else:
        feat_train = torch.load(path_train)

    if args.dtype == "float16":
        feat_train = feat_train.half()
        feat_val = feat_val.half()

    feat_train, feat_val = feat_train.cuda(), feat_val.cuda()
    mem = torch.cuda.memory_allocated() / 1024.**2
    print(f"Features: {mem:.0f}MB, train: {list(feat_train.shape)}, val: {list(feat_val.shape)}")
    return feat_train, feat_val


def load_feat_ood(args, model, transform):
    tag = f"_{args.epoch}" if args.epoch is not None else ""

    ood_list = ['places', 'sun', 'inat', 'dtd']
    path_ood = f"{args.cache_dir}/features/feat_ood{tag}.pt"

    if not os.path.isfile(path_ood):
        model.cuda()
        feat_ood = []
        for args.dataset in ood_list:
            _, valset = load_data(args.dataset, model, transform)
            _, feat = validate_model(args, model, valset)
            feat_ood.append(feat)
            print(f"{args.dataset}: ", feat.shape)

        feat_ood = torch.cat(feat_ood, dim=0)
        torch.save(feat_ood.cpu(), path_ood)
    else:
        feat_ood = torch.load(path_ood)

    if args.dtype == "float16":
        feat_ood = feat_ood.half()

    feat_ood = feat_ood.cuda()
    print(f"Load ood features: ", feat_ood.shape)
    return feat_ood


def load_model_classifier(args):
    model, transform = load_model(args, verbose=True)
    model.eval()
    if args.name[:6] == 'resnet':
        classifier = model.fc.cuda()
    elif args.name[:8] == 'convnext':
        classifier = model.head.fc.cuda()
    else:
        classifier = model.head.cuda()

    if args.dtype == "float16":
        model = model.half()
        classifier = classifier.half()

    return model, classifier, transform


def cal_prob(classifier, feat, n_b=50000, return_logit=False):
    prob = []
    n = len(feat)
    with torch.no_grad():
        for i in range((n - 1) // n_b + 1):
            prob_ = classifier(feat[n_b * i:n_b * (i + 1)])
            if return_logit == False:
                prob_ = torch.softmax(prob_, dim=-1)
            prob.append(prob_)

    prob = torch.cat(prob)
    return prob
