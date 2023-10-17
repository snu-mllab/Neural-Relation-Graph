# -*- coding: utf-8 -*-
# @Time    : 6/10/21 11:00 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

import sys
import os
import datetime

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast, GradScaler
from sklearn import metrics


def train(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    # best_cum_mAP is checkpoint ensemble from the first epoch to the best epoch
    best_epoch, best_acc = 0, -np.inf
    global_step, epoch = 0, 0
    exp_dir = args.exp_dir

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)
    # Set up the optimizer
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(
        sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(
        sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    # dataset specific settings
    main_metrics = args.metrics
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    warmup = args.warmup
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     list(
                                                         range(args.lrscheduler_start, 1000,
                                                               args.lrscheduler_step)),
                                                     gamma=args.lrscheduler_decay)
    args.loss_fn = loss_fn
    print(
        'now training with {:s}, main metrics: {:s}, loss function: {:s}, learning rate scheduler: {:s}'
        .format(str(args.dataset), str(main_metrics), str(loss_fn), str(scheduler)))
    print(
        'The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epochs'
        .format(args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step))

    epoch += 1
    # for amp
    scaler = GradScaler()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    audio_model.train()
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        for i, (audio_input, labels) in enumerate(train_loader):

            B = audio_input.size(0)
            audio_input = audio_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / audio_input.shape[0])
            dnn_start_time = time.time()

            # first several steps for warm-up
            if global_step <= 1000 and global_step % 50 == 0 and warmup == True:
                warm_lr = (global_step / 1000) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

            with autocast():
                audio_output = audio_model(audio_input)
                if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                    loss = loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
                else:
                    loss = loss_fn(audio_output, labels)

            # optimization if amp is not used
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time) / audio_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time) / audio_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps / 10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                      'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                      'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                      'Train Loss {loss_meter.avg:.4f}\t'.format(
                          epoch,
                          i,
                          len(train_loader),
                          per_sample_time=per_sample_time,
                          per_sample_data_time=per_sample_data_time,
                          per_sample_dnn_time=per_sample_dnn_time,
                          loss_meter=loss_meter),
                      flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        print('start validation')
        acc, valid_loss = validate(audio_model, test_loader, args, epoch)
        print("acc: {:.6f}".format(acc))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))

        if acc > best_acc:
            best_acc = acc
            if main_metrics == 'acc':
                best_epoch = epoch

        if best_epoch == epoch:
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            # torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))
        if epoch % 5 == 0:
            torch.save(audio_model.state_dict(),
                       "%s/models/audio_model_epoch%d.pth" % (exp_dir, epoch))

        scheduler.step()
        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time - begin_time))
        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        loss_meter.reset()
        per_sample_dnn_time.reset()


def validate(audio_model, val_loader, args, epoch, return_target=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    # switch to evaluate mode
    audio_model.eval()

    end = time.time()
    A_predictions = []
    A_targets = []
    A_loss = []
    with torch.no_grad():
        for i, (audio_input, labels) in enumerate(val_loader):
            audio_input = audio_input.to(device)

            # compute output
            audio_output = audio_model(audio_input)
            predictions = audio_output.to('cpu').detach()

            A_predictions.append(predictions)
            A_targets.append(labels)

            # compute the loss
            labels = labels.to(device)
            if isinstance(args.loss_fn, torch.nn.CrossEntropyLoss):
                loss = args.loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
            else:
                loss = args.loss_fn(audio_output, labels)
            A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(audio_output, 1))

        # save the prediction here
        if epoch == 'valid_set':
            os.makedirs(args.exp_dir + '/predictions', exist_ok=True)
            np.savetxt(args.exp_dir + '/predictions/target.csv', target, delimiter=',')
            np.savetxt(args.exp_dir + '/predictions/predictions_' + str(epoch) + '.csv',
                       audio_output,
                       delimiter=',')

    if return_target:
        pred = torch.topk(torch.tensor(audio_output).float(), k=2, dim=-1)
        target = torch.argmax(torch.tensor(target).float(), dim=-1)
        return acc, loss, pred, target
    else:
        return acc, loss


def validate_feat(audio_model, val_loader, args, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    end = time.time()
    A_predictions = []
    A_targets = []
    A_feat = []
    with torch.no_grad():
        for i, (audio_input, labels) in enumerate(val_loader):
            audio_input = audio_input.to(device)

            # compute output
            audio_output, feat = audio_model(audio_input, return_feat=True)
            predictions = audio_output.to('cpu').detach()

            A_predictions.append(predictions)
            A_targets.append(labels)
            A_feat.append(feat.to('cpu').detach())

            # compute the loss
            labels = labels.to(device)
            batch_time.update(time.time() - end)
            end = time.time()
            print(f"{i}/{len(val_loader)}", end='\r')

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        feat = torch.cat(A_feat)

        acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(audio_output, 1))

        # save the prediction here

    return acc, feat, audio_output


def validate_ensemble(args, epoch):
    exp_dir = args.exp_dir
    target = np.loadtxt(exp_dir + '/predictions/target.csv', delimiter=',')
    if epoch == 1:
        cum_predictions = np.loadtxt(exp_dir + '/predictions/predictions_1.csv', delimiter=',')
    else:
        cum_predictions = np.loadtxt(exp_dir + '/predictions/cum_predictions.csv',
                                     delimiter=',') * (epoch - 1)
        predictions = np.loadtxt(exp_dir + '/predictions/predictions_' + str(epoch) + '.csv',
                                 delimiter=',')
        cum_predictions = cum_predictions + predictions
        # remove the prediction file to save storage space
        os.remove(exp_dir + '/predictions/predictions_' + str(epoch - 1) + '.csv')

    cum_predictions = cum_predictions / epoch
    np.savetxt(exp_dir + '/predictions/cum_predictions.csv', cum_predictions, delimiter=',')

    stats = calculate_stats(cum_predictions, target)
    return stats
