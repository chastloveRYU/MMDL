#!/usr/bin/env python3
import datetime
import json
import math
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader
import h5py
from statistics import mode
from ClassAwareSampler import ClassAwareSampler
import matplotlib
import matplotlib.pyplot as plt
from utils import *
# from read_sea_ice_charts import read_icetype
import torchvision.transforms.functional as F
import random
from typing import Sequence
from metrics import Evaluator


class unlabeled_Dataset(Dataset):
    def __init__(self, filelist, img_size=30, stride=30, metadata=None,
                 transform=transforms.ToTensor()):

        self.metadata = metadata
        self.transform = transform

        self.img = []
        self.lon = []
        self.lat = []
        self.incidence = []
        self.date = []

        count = 0
        for file in filelist:
            with h5py.File(file, 'r') as f:
                sigma0 = f['sigma0'][()]
                lon = f['longitude'][()]
                lat = f['latitude'][()]
                inci = f['incidence'][()]
                mask = f['mask'][()]
                date = file.split('_')[-4][0:8]

            row, col = mask.shape
            for j in range(0, (row - (img_size - stride)) // stride):
                for k in range(0, (col - (img_size - stride)) // stride):
                    img = sigma0[j * stride: j * stride + img_size, k * stride: k * stride + img_size, :]
                    img_mask = mask[j * stride: j * stride + img_size, k * stride: k * stride + img_size]
                    longitude = lon[j * stride: j * stride + img_size, k * stride: k * stride + img_size]
                    latitude = lat[j * stride: j * stride + img_size, k * stride: k * stride + img_size]
                    incidence = inci[j * stride: j * stride + img_size, k * stride: k * stride + img_size]
                    if not np.isnan(img_mask).any():
                        self.img.append(img)
                        self.lon.append(np.mean(longitude))
                        self.lat.append(np.mean(latitude))
                        self.incidence.append(np.mean(incidence))
                        self.date.append(date)

        print("Total unlabeled samplers: {}".format(len(self.img)))

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):

        date_time = datetime.datetime.strptime(self.date[idx], '%Y%m%d')
        date = get_scaled_date_ratio(date_time)
        lat = float(self.lat[idx]) / 90
        lon = float(self.lon[idx]) / 180
        # 18.9 - 47 degree
        inci = float(self.incidence[idx])
        inci = 2 * ((inci - 18.9) / (47 - 18.9) - 0.5)  # normalized to [-1, 1]

        extra = []
        if self.metadata is not None:
            if 'geo' in self.metadata:
                extra += [lat, lon]
            if 'temporal' in self.metadata:
                extra += [date]
            if 'inci' in self.metadata:
                extra += [inci]

        extra = np.array(extra)
        extra = encode_loc_time(extra)

        if self.transform is not None:
            img = self.transform(self.img[idx])
        return img, extra


def encode_loc_time(loc_time):
    # assumes inputs location and date features are in range -1 to 1
    # location is lon, lat
    feats = np.concatenate((np.sin(math.pi * loc_time), np.cos(math.pi * loc_time)))
    return feats


def _is_leap_year(year):
    if year % 4 != 0 or (year % 100 == 0 and year % 400 != 0):
        return False
    return True


def get_scaled_date_ratio(date_time):
    r'''
    scale date to [-1,1]
    '''
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    total_days = 365
    year = date_time.year
    month = date_time.month
    day = date_time.day
    if _is_leap_year(year):
        days[1] += 1
        total_days += 1

    assert day <= days[month - 1]
    sum_days = sum(days[:month - 1]) + day
    assert sum_days > 0 and sum_days <= total_days

    return (sum_days / total_days) * 2 - 1


class pseudo_Dataset(Dataset):
    def __init__(self, unlabeled_dataset, pseudo_labels, prob, conf, F1_core, type_percent_train, transform=None):

        self.transform = transform
        # self.img = []
        # self.extra = []
        # self.label = []

        samples = {x: [] for x in range(11)}
        for (img, extra), label, p in zip(unlabeled_dataset, pseudo_labels, prob):
            # if p >= conf[int(label)]:
            if p >= 0.95 * 0.9 * F1_core[int(label)]:
                # self.img.append(img)
                # self.extra.append(extra)
                # self.label.append(label)
                samples[int(label)].append((img, label, extra))

        num = [len(x) for _, x in samples.items()]

        print("==> selected unlabeled samples:", sum(num))
        print("==> the sampler num of each type in unlabeled datasets:", num)
        print("==> imbalance ratio of unlabeled set", max(num) / min(num))

        # samples7 = samples[7]
        # samples10 = samples[10]
        # random.shuffle(samples7)
        # random.shuffle(samples10)
        # samples[7] = random.sample(samples7, num[6])
        # samples[10] = random.sample(samples10, num[6])
        #
        # selected_data = []
        # # selected_data = [[].extend(samples[i]) for i in range(11)]
        # for i in range(11):
        #     selected_data.extend(samples[i])
        #
        # self.img = []
        # self.extra = []
        # self.label = []
        # for img, label, extra in selected_data:
        #     self.img.append(img)
        #     self.extra.append(extra)
        #     self.label.append(label)
        #
        # num = [len(x) for _, x in samples.items()]
        #
        # print("==> resampling")
        # print("==> selected unlabeled samples:", sum(num))
        # print("==> the sampler num of each type in unlabeled datasets:", num)
        # print("==> imbalance ratio of unlabeled set", max(num) / min(num))

        print("==> resample unlabeled data")
        # min_num_idx = type_percent_train.index(min(type_percent_train))
        # ratio = np.array(type_percent_train) / min(type_percent_train)
        # selected_num = [int(num[min_num_idx] * r) for r in ratio.tolist()]
        selected_num = []
        for i in range(11):
            if num[i] / min(num) > 10:
                selected_num.append(int(min(num) * 10))
            else:
                selected_num.append(num[i])

        print("==> number of selected unlabeled samples:", selected_num)
        print("total number:", sum(selected_num))
        print("==> imbalance ratio of selected unlabeled samples:", max(selected_num) / min(selected_num))

        selected_data = []
        for i in range(11):
            data = samples[i]
            random.shuffle(data)
            selected_data.extend(random.sample(data, selected_num[i]))

        self.img = []
        self.extra = []
        self.label = []
        for img, label, extra in selected_data:
            self.img.append(img)
            self.extra.append(extra)
            self.label.append(label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        img = self.img[item]
        extra = self.extra[item]
        label = int(self.label[item])

        if self.transform is not None:
            img = self.transform(img)

        return img, label, extra


class Rotation:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)


def load_unlabeled_dataset(data_dir, net, val_loader, type_percent_train, img_size=30, stride=30, batch_size=512,
                           metadata=None, num_workers=32, augmentation=False):
    # 测试模式
    net.eval()

    # make predictions on val dataset
    print("==> make predictions on val dataset")
    confidence = []
    truth = []
    false_confidence = []
    false_truth = []
    evaluator = Evaluator(11)
    for i, (images, target, location) in enumerate(val_loader):
        # compute output
        images = images.cuda(non_blocking=True)
        location = location.cuda(non_blocking=True).float()

        with torch.no_grad():
            output = net(images, location)

        output = torch.softmax(output, 1)
        prob, idx = torch.max(output, 1)
        prob = prob.cpu().numpy()
        idx = idx.cpu().numpy()

        target = target.cpu().numpy()

        confidence.extend(prob[idx == target].tolist())
        truth.extend(target[idx == target].tolist())

        false_confidence.extend(prob[idx != target].tolist())
        false_truth.extend(target[idx != target].tolist())

        evaluator.add_batch(target, idx)

    acc = evaluator.Mean_Pixel_Accuracy_Class()
    acc_class = evaluator.Pixel_Accuracy_Class()
    F1_score = evaluator.F1_Score()
    print("F1_score of val set:", F1_score)

    print("confidence > 95:", len(np.array(confidence)[np.array(confidence) > 0.95]))
    print("==> correct pred num:", len(confidence))
    print("==> analyze the confidence")
    conf = {x: [] for x in range(11)}
    for c, t in zip(confidence, truth):
        conf[t].append(c)
    for key, value in conf.items():
        conf[key] = np.mean(np.array(value))
    print(conf)

    # false_conf = {x: [] for x in range(11)}
    # for c, t in zip(false_confidence, false_truth):
    #     false_conf[t].append(c)
    # for key, value in false_conf.items():
    #     false_conf[key] = np.mean(np.array(value))
    # print(false_conf)

    # acc = []
    # for i in range(11):
    #     acc.append(len(conf[i]) / (len(conf[i]) + len(false_conf[i])))
    # print("val acc:", acc)

    filelist = []
    for dir in data_dir:
        filelist.extend(os.path.join(dir, file) for file in os.listdir(dir))
    unlabeled_dataset = unlabeled_Dataset(filelist, img_size=img_size, stride=stride, metadata=metadata)
    unlabeled_data_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False,
                                       drop_last=False, pin_memory=True, num_workers=num_workers)

    # make predictions on unlabeled dataset
    # switch to evaluate mode
    print("==> make predictions on unlabeled dataset")

    # generate pseudo labels
    label = []
    probability = []
    for i, (images, location) in enumerate(unlabeled_data_loader):
        # compute output
        images = images.cuda(non_blocking=True)
        location = location.cuda(non_blocking=True).float()

        with torch.no_grad():
            output = net(images, location)

        output = torch.softmax(output, 1)
        prob, idx = torch.max(output, 1)
        prob = prob.cpu().numpy()
        idx = idx.cpu().numpy()

        # # 选取概率最大的前两个比较
        # prob, idx = torch.sort(output, dim=1, descending=True)
        # prob = prob.cpu().numpy()[:, 0:2]
        # idx = idx.cpu().numpy()[:, 0:2]
        # idx = idx[(prob[:, 0] - prob[:, 1]) >= 0.4, :][:, 0]
        # prob = prob[(prob[:, 0] - prob[:, 1]) >= 0.4, :][:, 0]

        # # exclude class 7 and 10
        # prob = prob[(idx != 7) & (idx != 10)]
        # idx = idx[(idx != 7) & (idx != 10)]

        probability.extend(prob.tolist())

        # # measure accuracy and record loss
        # pred = torch.argmax(output, 1)
        # pred = torch.squeeze(pred).cpu().numpy().tolist()

        label.extend(idx.tolist())

    if augmentation:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            Rotation([90, 180, 270])
        ])
    else:
        transform = None

    dataset = pseudo_Dataset(unlabeled_dataset, label, probability, conf, F1_score, type_percent_train,
                             transform=transform)

    return dataset
