import torch
from math import ceil


def normalize(feat, nc=50000):
    """Normalize feature

    Args:
        feat (torch tensor, [N,d])
        nc (int, optional): Maximum number of tensor size for each iteration. Defaults to 50000.
    """
    with torch.no_grad():
        split = ceil(len(feat) / nc)
        for i in range(split):
            feat_ = feat[i * nc:(i + 1) * nc]
            feat[i * nc:(i + 1) * nc] = feat_ / torch.sqrt((feat_**2).sum(-1)).reshape(-1, 1)

    return feat


if __name__ == "__main__":
    # For feature extraction, please refer to ./model/src/feat.py
    pass