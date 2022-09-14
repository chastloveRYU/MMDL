#!/usr/bin/env python3
import argparse
import datetime
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim

from dataset import load_dataset, Dataset
import models
import utils
import h5py
from metrics import Evaluator
from torchsummary import summary
from unlabeled_dataset import load_unlabeled_dataset
from torch.utils.data import DataLoader
from ClassAwareSampler import ClassAwareSampler


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--name', required=True, type=str)
    parser.add_argument('--data_dir', default=['/home/ubuntu/LiZhao/201910to202009_hdf_ASI_cut_full_info/',
                                               '/home/ubuntu/LiZhao/201910to202009_sar_asi_charts'])
    parser.add_argument('--additional_data_dir', default='/home/ubuntu/LiZhao/202010to202011_labeled_hdf')
    parser.add_argument('--train_filelist', default=['./datasets/train_filelist_western.hdf',
                                                     './datasets/train_filelist_eastern.hdf'])
    parser.add_argument('--test_filelist', default=['./datasets/test_filelist_western.hdf',
                                                    './datasets/test_filelist_eastern.hdf'])
    parser.add_argument('--save_dir', default='./logs', type=str)
    parser.add_argument('--model_file', default='resnet_dynamic_mlp', type=str, help='model file name')
    parser.add_argument('--model_name', default='resnet50', type=str, help='model type in detail')
    parser.add_argument('--fold', default=1, type=int, help='training fold')
    parser.add_argument('--random_seed', default=37, type=int)
    parser.add_argument('--gpu', default="0", type=str)

    # train
    parser.add_argument('--img_size', default=30, type=int)
    parser.add_argument('--stride', default=30, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--data_type', default='icetype', type=str)
    parser.add_argument('--num_channels', default=2, type=int)
    parser.add_argument('--sampler', default=None)
    parser.add_argument('--warmup', default=2, type=int)
    parser.add_argument('--start_lr', default=0.04, type=float)
    parser.add_argument('--stop_epoch', default=90, type=int)
    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--save_name', default='best', type=str)
    parser.add_argument('--semi', action='store_true', default=False)
    parser.add_argument('--mixup', action='store_true', default=False)
    parser.add_argument('--parallel', action='store_true', default=False)
    parser.add_argument('--label_smoothing', action='store_true', default=False)
    # parser.add_argument('--droprate', default=0, type=float)

    # data
    parser.add_argument('--augmentation', action='store_true', default=False)
    parser.add_argument('--tencrop', action='store_true', default=False)
    parser.add_argument('--image_only', action='store_true', default=False)
    parser.add_argument('--metadata', default=None)
    parser.add_argument('--metadata_aug', default='', type=str)

    # model
    parser.add_argument('--retrain', action='store_true', default=False)
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--evaluate', action='store_true', help='evaluate model on validation set')

    # dynamic MLP
    parser.add_argument('--mlp_type', default='c', type=str, help='dynamic mlp versions: a|b|c')
    parser.add_argument('--mlp_d', default=256, type=int)
    parser.add_argument('--mlp_h', default=64, type=int)
    parser.add_argument('--mlp_n', default=2, type=int)

    args = parser.parse_args()
    args.mlp_cin = 0
    if args.metadata is not None:
        if 'geo' in args.metadata:
            args.mlp_cin += 4
        if 'temporal' in args.metadata:
            args.mlp_cin += 2
        if 'inci' in args.metadata:
            args.mlp_cin += 2

    args.num_classes_icetype = 11
    args.num_classes_icecon = 12

    if not args.parallel:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # # get logger
    # creat_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    # args.path_log = os.path.join(args.save_dir, f'{args.model_file}')
    # os.makedirs(args.path_log, exist_ok=True)
    # logger = utils.create_logging(os.path.join(args.path_log, '%s_train.log' % creat_time))
    #
    # # get net
    # net = models.__dict__[args.model_file].__dict__[args.model_name](logger, args)
    # print(net)
    # return None

    # set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    args.nprocs = torch.cuda.device_count()

    # get logger
    creat_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    args.path_log = os.path.join(args.save_dir, f'{args.model_file}')
    os.makedirs(args.path_log, exist_ok=True)
    logger = utils.create_logging(os.path.join(args.path_log, '%s_train.log' % creat_time))

    # print args
    for param in sorted(vars(args).keys()):
        logger.info('--{0} {1}'.format(param, vars(args)[param]))

    # get net
    net = models.__dict__[args.model_file].__dict__[args.model_name](logger, args)
    net.cuda()
    if args.parallel:
        net = torch.nn.DataParallel(net)

    if not args.evaluate:
        # get datasets
        filelist = []
        for dir, train_filelist in zip(args.data_dir, args.train_filelist):
            with h5py.File(train_filelist, 'r') as f:
                files = f['filelist'][()]
                filelist.extend([os.path.join(dir, file) for file in files])

        train_dataset, val_dataset = load_dataset(args, filelist, phase='train',
                                                  img_size=args.img_size,
                                                  stride=args.stride,
                                                  data_type=args.data_type,
                                                  metadata=args.metadata,
                                                  augmentation=args.augmentation)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=False, shuffle=True,
                                  pin_memory=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                pin_memory=True, num_workers=args.num_workers)

    # get criterion
    if args.label_smoothing:
        criterion = utils.LabelSmoothingLoss(classes=args.num_classes, smoothing=0.1).cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda()

    # get optimizer
    optimizer = optim.SGD(net.parameters(), lr=args.start_lr, momentum=0.9, weight_decay=1e-4)

    start_epoch = 1
    if args.evaluate:
        filelist = []
        for dir, test_filelist in zip(args.data_dir, args.test_filelist):
            with h5py.File(test_filelist, 'r') as f:
                files = f['filelist'][()]
                filelist.extend([os.path.join(dir, file) for file in files])
        test_dataset = load_dataset(args, filelist, phase='test',
                                    img_size=args.img_size,
                                    stride=args.stride,
                                    data_type=args.data_type,
                                    metadata=args.metadata,
                                    augmentation=args.augmentation)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                 pin_memory=True, num_workers=args.num_workers)

        checkpoint = './logs/%s/fold%s_%s.pth' % (args.model_file, args.fold, args.save_name)
        print("==> load checkpoint '{}".format(checkpoint))
        checkpoint = torch.load(checkpoint)
        state_dict = checkpoint['model']
        net.load_state_dict(state_dict)
        epoch = start_epoch - 1
        score = []
        micro_score = []
        macro_score = []
        for dataloader in [test_loader]:
            F1_score, F1_score_micro_average, F1_score_macro_average, outputs = validate(dataloader, net, criterion,
                                                                                         epoch, logger, args)
            score.append(F1_score)
            micro_score.append(F1_score_micro_average)
            macro_score.append(F1_score_macro_average)
        print("F1 score:", score)
        print("micro score:", micro_score)
        print("macro score:", macro_score)
        save_filename = './performance/%s.hdf' % args.save_name
        with h5py.File(save_filename, 'w') as f:
            f.create_dataset("F1_score", data=score)
            f.create_dataset("micro_score", data=micro_score)
            f.create_dataset("macro_score", data=macro_score)

        # logger.info('\t'.join(outputs))
        # logger.info('Exp path: %s' % args.path_log)
        return

    if args.retrain:
        print("==> retrain classifier")
        checkpoint = './logs/%s/fold%s_%s.pth' % (args.model_file, args.fold, args.save_name.replace('_CRT', ''))
        print("==> load checkpoint '{}".format(checkpoint))
        checkpoint = torch.load(checkpoint)
        state_dict = checkpoint['model']

        # 不加载fc层的权重
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'fc' not in k and 'loc_att' not in k:
                new_state_dict[k] = v
        net.load_state_dict(new_state_dict, strict=False)
        for name, param in net.named_parameters():
            if 'fc' not in name and 'loc_att' not in name:
                param.requires_grad = False

        # optimize only the fc layer
        parameters = list(filter(lambda p: p.requires_grad, net.parameters()))
        optimizer = optim.SGD(parameters, lr=args.start_lr, momentum=0.9, weight_decay=1e-4)

    best_macro = 0.0
    best_micro = 0.0
    args.time_sec_tot = 0.0
    args.start_epoch = start_epoch
    for epoch in range(start_epoch, args.stop_epoch + 1):
        train(train_loader, net, criterion, optimizer, epoch, logger, args)
        utils.save_checkpoint(epoch, net, optimizer, args, save_name=args.save_name + '_latest')
        F1_score_icetype, F1_score_micro_average_icetype, F1_score_macro_average_icetype, \
        F1_score_icecon, F1_score_micro_average_icecon, F1_score_macro_average_icecon, \
        outputs = validate(val_loader, net, criterion, epoch, logger, args)
        F1_score_macro_average = (F1_score_macro_average_icetype + F1_score_macro_average_icecon) * 0.5
        F1_score_micro_average = (F1_score_micro_average_icetype + F1_score_micro_average_icecon) * 0.5
        if F1_score_macro_average > best_macro:
            best_macro = F1_score_macro_average
            best_micro = F1_score_micro_average
            # best_acc5 = acc5
            utils.save_checkpoint(epoch, net, optimizer, args, save_name=args.save_name)

        outputs += [
            'best_macro: {:.4f}'.format(best_macro),
            'best_micro: {:.4f}'.format(best_micro)
        ]
        logger.info('\t'.join(outputs))
        logger.info('Exp path: %s' % args.path_log)

        # scheduler.step()
    # utils.save_checkpoint(epoch, net, optimizer, args, save_name=args.save_name)

    # # evaluate
    # checkpoint = './logs/%s/fold%s_%s.pth' % (args.model_file, args.fold, args.save_name)
    # print("==> load checkpoint '{}".format(checkpoint))
    # checkpoint = torch.load(checkpoint)
    # state_dict = checkpoint['model']
    # net.load_state_dict(state_dict)
    # epoch = start_epoch - 1
    # acc_train, _, acc_class_train, kappa_train = validate(train_loader, net, criterion, epoch, logger, args)
    # acc_val, _, acc_class_val, kappa_val = validate(val_loader, net, criterion, epoch, logger, args)
    # print("train acc: {:.4f}, val acc: {:.4f}, val kappa: {:.4f}".format(acc_train, acc_val, kappa_val))
    # print("train kappa: {:.4f}, val kappa: {:.4f}".format(kappa_train, kappa_val))
    # print("train class acc: ", acc_class_train)
    # print("val class acc: ", acc_class_val)

    # logger.info('\t'.join(outputs))
    # logger.info('Exp path: %s' % args.path_log)


def train(train_loader, net, criterion, optimizer, epoch, logger, args):
    # switch to train mode
    net.train()
    minibatch_count = len(train_loader)
    scaler = torch.cuda.amp.GradScaler()
    tstart = time.time()
    for i, (images, target, target_con, location) in enumerate(train_loader):
        # change learning rate
        learning_rate = utils.adjust_learning_rate(optimizer, i, epoch, minibatch_count, args)
        # learning_rate = optimizer.state_dict()['param_groups'][0]['lr']

        # measure data loading time
        tdata = time.time() - tstart

        images = images.cuda(non_blocking=True)
        target = target.long().cuda(non_blocking=True)
        target_con = target_con.long().cuda(non_blocking=True)
        location = location.cuda(non_blocking=True).float()

        # compute output
        with torch.cuda.amp.autocast():
            if args.image_only:
                output_icetype, output_icecon = net(images)
            else:
                output_icetype, output_icecon = net(images, location)

            loss_icetype = criterion(output_icetype, target)
            loss_icecon = criterion(output_icecon, target_con)

        acc_icetype, _ = utils.accuracy(output_icetype, target, topk=(1, 5))
        acc_icecon, _ = utils.accuracy(output_icecon, target_con, topk=(1, 5))

        # compute gradient and do sgd step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        tend = time.time()
        ttrain = tend - tstart
        tstart = tend

        args.time_sec_tot += ttrain
        time_sec_avg = args.time_sec_tot / ((epoch - args.start_epoch) * minibatch_count + i + 1)
        eta_sec = time_sec_avg * ((args.stop_epoch + 1 - epoch) * minibatch_count - i - 1)
        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))

        outputs = [
            "e: {}/{},{}/{}".format(epoch, args.stop_epoch, i, minibatch_count),
            "{:.2f} mb/s".format(1. / ttrain),
            'eta: {}'.format(eta_str),
            'time: {:.3f}'.format(ttrain),
            'data_time: {:.3f}'.format(tdata),
            'lr: {:.4f}'.format(learning_rate),
            'acc_icetype: {:.4f}'.format(acc_icetype.item()),
            'acc_icecon: {:.4f}'.format(acc_icecon.item()),
            'loss: {:.4f}'.format(loss.item()),
        ]

        if tdata / ttrain > .05:
            outputs += [
                "dp/tot: {:.4f}".format(tdata / ttrain),
            ]

        if i % 100 == 0:
            logger.info('\t'.join(outputs))


def validate(val_loader, net, criterion, epoch, logger, args):
    # switch to evaluate mode
    logger.info('eval epoch {}'.format(epoch))
    net.eval()

    loss = 0
    evaluator_icetype = Evaluator(args.num_classes_icetype)
    evaluator_icecon = Evaluator(args.num_classes_icecon)
    for i, (images, target, target_con, location) in enumerate(val_loader):
        # compute output
        images = images.cuda(non_blocking=True)
        target = target.long().cuda(non_blocking=True)
        target_con = target_con.long().cuda(non_blocking=True)
        location = location.cuda(non_blocking=True).float()

        # print("target shape:", target.shape)

        with torch.no_grad():
            if args.image_only:
                output_icetype, output_icecon = net(images)
            else:
                output_icetype, output_icecon = net(images, location)
        # print("output shape:", output.shape)

        # measure accuracy and record loss
        pred_icetype = torch.argmax(output_icetype, 1)
        pred_icecon = torch.argmax(output_icecon, 1)
        # print("pred shape:", pred.shape)
        # pred = torch.squeeze(pred).cpu().numpy()
        # print("pred shape 1:", pred.shape)
        evaluator_icetype.add_batch(target.cpu().numpy(), pred_icetype.cpu().numpy())
        evaluator_icecon.add_batch(target_con.cpu().numpy(), pred_icecon.cpu().numpy())
        # acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        # num = target.size(0)
        # valdation_num += num
        # acc1_sum += acc1.item() * num
        # acc5_sum += acc5.item() * num
        loss += (criterion(output_icetype, target).item() + criterion(output_icecon, target_con).item()) * 0.5
        if i % 20 == 0:
            logger.info('iter {}/{}'.format(i, len(val_loader)))

    loss = loss / len(val_loader)
    # acc1 = acc1_sum / valdation_num
    # acc5 = acc5_sum / valdation_num
    F1_score_icetype = evaluator_icetype.F1_Score()
    F1_score_micro_average_icetype = evaluator_icetype.F1_Score(method='micro_average')
    F1_score_macro_average_icetype = evaluator_icetype.F1_Score(method='macro_average')

    F1_score_icecon = evaluator_icecon.F1_Score()
    F1_score_micro_average_icecon = evaluator_icecon.F1_Score(method='micro_average')
    F1_score_macro_average_icecon = evaluator_icecon.F1_Score(method='macro_average')

    # if args.evaluate:
    #
    #     outputs = [
    #         "val e: {}".format(epoch),
    #         'acc: {:.4f}'.format(acc),
    #         'loss: {:.4f}'.format(loss),
    #     ]
    #     # print(acc_class)
    # else:
    outputs = [
        "val e: {}".format(epoch),
        'micro_average_icetype: {:.4f}'.format(F1_score_micro_average_icetype),
        'macro_average_icetype: {:.4f}'.format(F1_score_macro_average_icetype),
        'micro_average_icecon: {:.4f}'.format(F1_score_micro_average_icecon),
        'macro_average_icecon: {:.4f}'.format(F1_score_macro_average_icecon),
        'loss: {:.4f}'.format(loss),
    ]

    return F1_score_icetype, F1_score_micro_average_icetype, F1_score_macro_average_icetype, \
           F1_score_icecon, F1_score_micro_average_icecon, F1_score_macro_average_icecon, \
           outputs

if __name__ == '__main__':
    main()
