import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .utils import AverageMeter, accuracy, round_list


def make_result_dict():
    results = {}
    results['logit'] = []
    results['logit_ans'] = []
    results['pred'] = []
    results['prob'] = []
    results['ans'] = []
    results['ans_p'] = []
    results['cor1'] = []
    results['cor5'] = []

    return results


def update_results(results, output, target, topk=5):
    prob = F.softmax(output, dim=1)
    prob_top, pred = prob.topk(topk, 1, True, True)
    logit_top, _ = output.topk(topk, 1, True, True)
    acc, cor = accuracy(prob, target, pred, topk=(1, topk))

    results['logit'] += round_list(logit_top.tolist())
    results['logit_ans'] += round_list(torch.gather(output, 1, target[:, None]).squeeze().tolist())

    results['pred'] += pred.tolist()
    results['prob'] += round_list(prob_top.tolist())

    results['ans'] += target.tolist()
    results['ans_p'] += round_list(torch.gather(prob, 1, target[:, None]).squeeze().tolist())

    results['cor1'] += cor[1]
    results['cor5'] += cor[topk]

    acc1, acc5 = acc[1], acc[topk]
    return acc1, acc5


def forward_with_feat(x, model):
    """ The model forward codes should be modified to return features
    """
    if (model.name[:3] == 'mae') or (model.name[:4] == 'beit'):
        feat = model.forward_features(x)
        output = model.head(feat)
    elif model.name[:6] == 'resnet':
        feat = model.forward_features(x)
        if model.drop_rate:
            feat = F.dropout(feat, p=float(model.drop_rate), training=model.training)
        output = model.fc(feat)
    elif model.name[:8] == 'convnext':
        feat = model.forward_features(x)
        feat = model.forward_head(feat, pre_logits=True)
        output = model.head.fc(feat)

    return output, feat


def validate_feat(val_loader, model, topk=10, print_freq=10, verbose=True):
    top1, batch_time = AverageMeter(), AverageMeter()
    results = make_result_dict()
    features = []

    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.cuda(), target.cuda()
            output, feat = forward_with_feat(input, model)

            acc1, acc5 = update_results(results, output, target, topk=topk)
            features.append(feat.cpu())

            n = input.size(0)
            top1.update(acc1, n)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0 and verbose == True:
                mem = torch.cuda.max_memory_allocated() / 1024.**2
                print('Validation: [{}/{}]\t'
                      'Time {batch_time.val:.2f} ({batch_time.avg:.2f})  '
                      'Top 1-acc {top1.val:.2f} ({top1.avg:.2f})  '
                      'Mem {mem:.0f}MB'.format(i,
                                               len(val_loader),
                                               batch_time=batch_time,
                                               top1=top1,
                                               mem=mem))

    results_np = {}
    for k, i in results.items():
        results_np[k] = np.array(i)

    features = torch.cat(features)
    return results_np, features


def validate_model(args, model, valset):
    model.cuda()

    while True:
        try:
            val_loader = DataLoader(valset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers,
                                    pin_memory=False)
            results = validate_feat(val_loader, model, print_freq=args.print_freq)
            break
        except KeyboardInterrupt:
            break
        except:
            args.batch_size //= 2
            args.print_freq *= 2
            print("Batch size is reduced! ", args.batch_size)
            if args.batch_size == 0: break

    return results


if __name__ == "__main__":
    from argument import args, DIR_RESULT
    from models.load import load_model
    from imagenet.data import load_data
    import pandas as pd
    import torch
    import os

    model, transform = load_model(args.name, verbose=True)
    model.cuda()

    print(f"Files will be saved at {args.fname} folder")

    _, valset, mask, real = load_data(args.dataset, args.data_dir, model, transform)
    results = validate_model(args, model, valset, mask=mask, real=real, return_feat=True)
    df = pd.DataFrame(data=results)
    df.to_csv(
        os.path.join(DIR_RESULT,
                     f"results/validation5/{args.dataset}_{args.fname}_tta{args.multicrop}.csv"))
