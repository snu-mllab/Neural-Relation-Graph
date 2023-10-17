# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
import os
import sys
import time
import torch
import torch.nn as nn

basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader
import models
import numpy as np
from traintest import validate
from math import ceil

print("I am process %s, running on %s: starting (%s)" %
      (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default='', help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default='ast', help="the model used")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used")

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w',
                    '--num-workers',
                    default=32,
                    type=int,
                    metavar='NW',
                    help='# of workers for dataloading (default: 32)')

parser.add_argument("--n-print-steps",
                    type=int,
                    default=100,
                    help="number of steps to print statistics")

# the stride used in patch spliting, e.g., for patch size 16*16, a stride of 16 means no overlapping, a stride of 10 means overlap of 6.
parser.add_argument("--fstride",
                    type=int,
                    default=10,
                    help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride",
                    type=int,
                    default=10,
                    help="soft split time stride, overlap=patch_size-stride")

parser.add_argument("--dataset_mean",
                    type=float,
                    default=-4.2677393,
                    help="the dataset spectrogram mean")
parser.add_argument("--dataset_std",
                    type=float,
                    default=4.5689974,
                    help="the dataset spectrogram std")
parser.add_argument("--audio_length", type=int, default=1024, help="the dataset spectrogram std")

parser.add_argument("--metrics",
                    type=str,
                    default=None,
                    help="evaluation metrics",
                    choices=["acc", "mAP"])
parser.add_argument("--loss", type=str, default=None, help="loss function", choices=["BCE", "CE"])
args = parser.parse_args()

# transformer based model
if args.model == 'ast':
    print('now test a audio spectrogram transformer model')
    val_audio_conf = {
        'num_mel_bins': 128,
        'target_length': args.audio_length,
        'freqm': 0,
        'timem': 0,
        'mixup': 0,
        'dataset': args.dataset,
        'mode': 'evaluation',
        'mean': args.dataset_mean,
        'std': args.dataset_std,
        'noise': False
    }

    train_loader = torch.utils.data.DataLoader(dataloader.AudiosetDataset(
        args.data_train, label_csv=args.label_csv, audio_conf=val_audio_conf),
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.num_workers,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(dataloader.AudiosetDataset(args.data_val,
                                                                        label_csv=args.label_csv,
                                                                        audio_conf=val_audio_conf),
                                             batch_size=args.batch_size * 2,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=True)

    audio_model = models.ASTModel(label_dim=args.n_class,
                                  fstride=args.fstride,
                                  tstride=args.tstride,
                                  input_fdim=128,
                                  input_tdim=args.audio_length,
                                  model_size='base384')

    print(f"\nTrain: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

if args.loss == 'BCE':
    args.loss_fn = nn.BCEWithLogitsLoss()
elif args.loss == 'CE':
    args.loss_fn = nn.CrossEntropyLoss()

# for speechcommands dataset, evaluate the best model on validation set on the test set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sd = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location=device)
audio_model = torch.nn.DataParallel(audio_model)
audio_model.load_state_dict(sd)

# best model on train set
train_acc, _, pred, target = validate(audio_model,
                                      train_loader,
                                      args,
                                      'train_set',
                                      return_target=True)
print('---------------evaluate on the train set---------------')
print("Accuracy: {:.3f}".format(train_acc))

n_total = len(target)
noise_ratio = 0.1
n = int(n_total * noise_ratio)
top1 = pred[1][:, 0]
top2 = pred[1][:, 1]

cor = torch.arange(n_total)[top1 == target]
step = int(ceil(len(cor) / n))

target_chg = torch.clone(target)
if args.dataset == 'esc50':
    path = '/home/janghyun/Codes/speech/ast/egs/esc50/target_noisy0.1.pt'
elif args.dataset == 'speechcommands':
    path = '/home/janghyun/Codes/speech/ast/egs/speechcommands/target_noisy0.1.pt'

idx_chg = cor[::step]

target_chg[idx_chg] = top2[idx_chg]
torch.save({'idx_clean': None, 'idx_chg': idx_chg, 'targets': target_chg, 'orig': target}, path)

print((pred[1][:, 0] == target).float().mean())
print((pred[1][:, 0] == target_chg).float().mean())
print(target[idx_chg], target_chg[idx_chg])

# best model on the validation set
val_acc, _ = validate(audio_model, val_loader, args, 'valid_set')
print('---------------evaluate on the validation set---------------')
print("Accuracy: {:.3f}".format(val_acc))

if args.dataset == 'speechcommands':
    # test the model on the evaluation set
    eval_loader = torch.utils.data.DataLoader(dataloader.AudiosetDataset(args.data_eval,
                                                                         label_csv=args.label_csv,
                                                                         audio_conf=val_audio_conf),
                                              batch_size=args.batch_size * 2,
                                              shuffle=False,
                                              num_workers=args.num_workers,
                                              pin_memory=True)
    eval_acc, _ = validate(audio_model, eval_loader, args, 'eval_set')
    print('---------------evaluate on the test set---------------')
    print("Accuracy: {:.3f}".format(eval_acc))
    np.savetxt(args.exp_dir + '/eval_result.csv', [val_acc, eval_acc])
