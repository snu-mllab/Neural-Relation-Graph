import torch
import time
from math import ceil


def kernel(feat, feat_t, label, label_t, prob, prob_t, split=2, kernel_type='cos_p'):
    """Kernel function (assume feature is normalized)
    """
    size = ceil(len(feat_t) / split)
    rel_full = []
    for i in range(split):
        feat_t_ = feat_t[i * size:(i + 1) * size]
        if label_t is not None:
            label_t_ = label_t[i * size:(i + 1) * size]

        with torch.no_grad():
            if kernel_type[:3] == 'cos':
                dot = torch.matmul(feat, feat_t_.transpose(1, 0))
                dot = torch.clamp(dot, min=0.)

                if kernel_type == 'cos_p':
                    prob_t_ = prob_t[i * size:(i + 1) * size]
                    sim = torch.matmul(prob, prob_t_.transpose(1, 0))
                    dot *= sim

            if label is not None:
                coef = 2 * (label.unsqueeze(1) == label_t_.unsqueeze(0)).float() - 1.
            else:
                coef = 1.
            rel = coef * dot

        rel_full.append(rel)

    rel_full = torch.cat(rel_full, dim=-1)
    return rel_full


def _init():
    graph = {}
    keys = ['low', 'low_val', 'top', 'top_val', 'score']
    for k in keys:
        graph[k] = []

    return graph, keys


def get_relation(feat,
                 feat_t,
                 label,
                 label_t,
                 prob,
                 prob_t,
                 pow=1,
                 kernel_type='cos_p',
                 chunk=50,
                 topk=0,
                 thres=0.03,
                 verbose=True):
    """Get relation values (top-k and summation)
    
    Args:
        feat (torch.Tensor [N,D]): features of the source data
        feat_t (torch.Tensor [N',D]): features of the target data
        label (torch.Tensor [N,]): label of source data
        label_t (torch.Tensor [N,]): label of target data
        prob (torch.Tensor [N,C]): probabilty vectors of the source data
        prob_t (torch.Tensor [N',C]): probabilty vectors of the target data
        pow (int): Temperature of kernel function
        kernel_type (str): Type of kernel function
        chunk (int): batch size of kernel calculation (trade off between memory and speed)
        topk (int): topk retrieval for the relation graph. Defaults to 0.
        thres (float): cut off value for small relation graph edges. Defaults to 0.03.
        verbose (bool, optional): Print progress. Defaults to True.

    Returns:
        graph: statistics of relation graph
    """

    n = feat.shape[0]
    n_chunk = ceil(n / chunk)

    graph, keys = _init()
    s = time.time()
    for i in range(n_chunk):
        feat_ = feat[i * chunk:(i + 1) * chunk]
        prob_ = None
        label_ = None
        if prob is not None:
            prob_ = prob[i * chunk:(i + 1) * chunk]
        if label is not None:
            label_ = label[i * chunk:(i + 1) * chunk]

        rel = kernel(feat_, feat_t, label_, label_t, prob_, prob_t, kernel_type=kernel_type)

        if verbose:
            mem = torch.cuda.memory_allocated() / 1024.**2
            eta = (time.time() - s) * (n_chunk - i) / (i + 1)
            print(f"[{eta:.0f}s] {i + 1}/{n_chunk}, {mem:.0f}MB", end='\r')

        if topk > 0:
            for largest in [False, True]:
                val, idx = torch.topk(rel, k=topk + 1, dim=1, largest=largest)

                k = 'top' if largest else "low"
                graph[f'{k}'].append(idx.cpu())
                graph[f'{k}_val'].append(val.cpu())

        mask = (rel.abs() > thres)
        rel_mask = mask * rel
        edge_sum = (rel_mask.sign() * (rel_mask.abs()**pow)).sum(-1)

        graph['score'].append(edge_sum.cpu())

    for k in keys:
        if len(graph[k]) > 0:
            graph[k] = torch.cat(graph[k], dim=0)

    return graph
