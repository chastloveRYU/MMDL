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

fontdict = {'family': 'times new roman',
            'color': 'k',
            'weight': 'normal',
            'size': 10.5,
            }
ice_type_code = {'OW': 0, 'NI': 1, 'GI': 2, 'GWI': 3, 'ThinFYI': 4, 'MFYI': 5,
                 'ThickFYI': 6, 'OI': 7, 'SYI': 8, 'MYI': 9, 'Land': 10}


class Dataset(Dataset):
    def __init__(self, filelist, img_size=30, stride=30, data_type='icetype', metadata=None,
                 transform=transforms.ToTensor()):

        self.data_type = data_type
        self.metadata = metadata
        self.transform = transform

        self.img = []
        self.label = []
        self.label_con = []
        self.lon = []
        self.lat = []
        self.incidence = []
        self.date = []

        count = 0
        for file in filelist:
            with h5py.File(file, 'r') as f:
                sigma0 = f['sigma0'][()]
                icetype = f['icetype'][()]
                icecon = f['sic_charts'][()]
                lon = f['lon'][()]
                lat = f['lat'][()]
                inci = f['incidence'][()]
                date = file.split('_')[-4][0:8]

            row, col = icetype.shape
            for j in range(0, (row - (img_size - stride)) // stride):
                for k in range(0, (col - (img_size - stride)) // stride):
                    img = sigma0[j * stride: j * stride + img_size, k * stride: k * stride + img_size, :]
                    label = icetype[j * stride: j * stride + img_size, k * stride: k * stride + img_size]
                    label_con = icecon[j * stride: j * stride + img_size, k * stride: k * stride + img_size]
                    longitude = lon[j * stride: j * stride + img_size, k * stride: k * stride + img_size]
                    latitude = lat[j * stride: j * stride + img_size, k * stride: k * stride + img_size]
                    incidence = inci[j * stride: j * stride + img_size, k * stride: k * stride + img_size]
                    if not np.isnan(label).any():
                        try:
                            maintype = mode(label.reshape(-1))
                            maincon = mode(label_con.reshape(-1))
                            # percent = label.reshape(-1).tolist().count(maintype) / len(label.reshape(-1))
                            # if percent > 0.5:
                            # self.stc.append(percent)
                            self.label.append(maintype)
                            self.label_con.append(maincon)
                            self.img.append(img)
                            self.lon.append(np.mean(longitude))
                            self.lat.append(np.mean(latitude))
                            self.incidence.append(np.mean(incidence))
                            self.date.append(date)
                        except:
                            count += 1

        print("Total {} samplers were ignored!".format(count))

        num = []
        for i in range(11):
            num.append(self.label.count(i))
        print("num_classes_icetype:", num)

        num = []
        for i in range(12):
            num.append(self.label_con.count(i))
        print("num_classes_icecon:", num)

    def __len__(self):
        return len(self.label)

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

        return img, self.label[idx], self.label_con[idx], extra

    def get_temporal_spatial_distribution(self):

        lon = np.array(self.lon)
        lat = np.array(self.lat)
        label = np.array(self.label)
        date = np.array(self.date)

        # remove land
        lon = lon[label < 10].tolist()
        lat = lat[label < 10].tolist()
        date = date[label < 10].tolist()
        label = label[label < 10].tolist()

        mon = []
        mon += ['2019' + str(x) for x in range(10, 13)]
        mon += ['20200' + str(x) for x in range(1, 10)]
        print(mon)

        ice_type = ['OW', 'NI', 'GI', 'GWI', 'ThinFYI', 'MFYI', 'ThickFYI', 'OI', 'SYI', 'MYI']

        lon_mon = {m: [] for m in mon}
        lat_mon = {m: [] for m in mon}
        label_mon = {m: [] for m in mon}
        for lon_, lat_, date_, label_ in zip(lon, lat, date, label):
            lon_mon[date_[0:6]].append(lon_)
            lat_mon[date_[0:6]].append(lat_)
            label_mon[date_[0:6]].append(label_)

        fig = plt.figure(figsize=(8, 4))
        ax = [[] for i in range(12)]
        bx = [[] for i in range(12)]
        for idx, mon in enumerate(mon, 1):
            ax[idx - 1] = fig.add_subplot(2, 6, idx)
            bx[idx - 1] = plt.scatter(lon_mon[mon], lat_mon[mon], s=0.1, c=label_mon[mon], marker='o',
                                      cmap=WMO_ice_chart_color_code(), vmin=0, vmax=9)
            plt.title(mon, fontdict=fontdict)
            plt.xticks([])
            plt.yticks([])
            # cbar = plt.colorbar()
            # ticks = list(np.linspace(0.5, 8.5, 10))
            # ticks = [x for x in ticks]
            # cbar.set_ticks(ticks=ticks)
            # cbar.set_ticklabels(ice_type)
        cbar = fig.colorbar(bx[0], ax=ax)
        ticks = list(np.linspace(0.5, 8.5, 10))
        ticks = [x for x in ticks]
        cbar.set_ticks(ticks=ticks)
        cbar.set_ticklabels(ice_type, fontsize=8.5, fontproperties='Times New Roman')
        plt.show()

    def get_incidence_dependence(self):

        sigma0 = []

        for img in self.img:
            sigma0.append(np.mean(img[:, :, 0]))

        inci = np.array(self.incidence)
        sigma0 = np.array(sigma0)
        label = np.array(self.label)

        print(np.min(inci), np.max(inci))
        print(np.min(sigma0), np.max(sigma0))

        ice_type = [x for x, _ in ice_type_code.items()]
        fig = plt.figure(figsize=(8, 4))
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.scatter(inci[label == i], sigma0[label == i], s=1)
            plt.xlim(18, 47)
            plt.ylim(-7.5, 4)
            plt.xticks([])
            plt.yticks([])
            plt.title(ice_type[i], fontdict=fontdict)
        fig.text(0.45, 0.05, 'Incidence angle')
        fig.text(0.09, 0.38, 'Normalized NRCS', rotation=90)
        plt.show()


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

class Rotation:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)


class data_augmentation(Dataset):
    def __init__(self, data, args, transform=None):
        self.data = data
        self.image_only = args.image_only
        self.metadata = args.metadata
        self.metadata_aug = args.metadata_aug
        self.transform = transform
        self.transform_meta = 0

    def __getitem__(self, item):
        img, label, label_con, extra = self.data[item]

        if self.transform is not None:
            img = self.transform(img)

        if not self.image_only:
            if self.metadata == 'geo_temporal_inci':
                if 'temporal' in self.metadata_aug:
                    dt = 2 * random.randint(-3, 3) / 365
                    sin = extra[2]
                    cos = extra[6]
                    extra[2] = sin * np.cos(math.pi * dt) + cos * np.sin(math.pi * dt)
                    extra[6] = cos * np.cos(math.pi * dt) - sin * np.sin(math.pi * dt)

                if 'geo' in self.metadata_aug:
                    dlon = random.uniform(-5/180, 5/180)
                    dlat = random.uniform(-2/90, 2/90)
                    dll = np.array([dlat, dlon])
                    sin = extra[0: 2]
                    cos = extra[4: 6]
                    extra[0: 2] = sin * np.cos(math.pi * dll) + cos * np.sin(math.pi * dll)
                    extra[4: 6] = cos * np.cos(math.pi * dll) - sin * np.sin(math.pi * dll)

                if 'inci' in self.metadata_aug:
                    # inci = 2 * ((inci - 18.9) / (47 - 18.9) - 0.5)
                    dinci = 2 * random.uniform(-5, 5) / (47 - 18.9)
                    sin = extra[3]
                    cos = extra[7]
                    extra[3] = sin * np.cos(math.pi * dinci) + cos * np.sin(math.pi * dinci)
                    extra[7] = cos * np.cos(math.pi * dinci) - sin * np.sin(math.pi * dinci)

        return img, label, label_con, extra

    def __len__(self):
        return len(self.data)


def load_dataset(args, filelist, phase='train', img_size=30, stride=30, data_type='icetype', metadata=None,
                 augmentation=False):
    dataset = Dataset(filelist, img_size=img_size, stride=stride, data_type=data_type, metadata=metadata)

    if phase == 'train':

        # add_filelist = [os.path.join(args.additional_data_dir, file)
        #                        for file in os.listdir(args.additional_data_dir)]
        # add_dataset = Dataset(add_filelist, img_size=img_size, stride=stride, num_classes=4,
        #                       metadata=metadata)
        # dataset = dataset + add_dataset
        len_train = int(len(dataset) * 0.9)
        len_test = len(dataset) - len_train
        train_dataset, test_dataset = random_split(dataset, [len_train, len_test],
                                                   generator=torch.Generator().manual_seed(args.random_seed))

        if augmentation:
            print("==> use augmentation")
            train_dataset = data_augmentation(train_dataset, args,
                                              transform=transforms.Compose([
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.RandomVerticalFlip(),
                                                  Rotation([90, 180, 270])
                                              ]))

        return train_dataset, test_dataset

    elif phase == 'test':
        return dataset


if __name__ == "__main__":
    path = r'E:\sea_ice_classification\data\S1\Western Arctic\201910to202009_hdf_ASI_cut_full_info'.replace('\\', '/')
    filename = './datasets/filelist_western_1day.hdf'

    with h5py.File('./datasets/train_filelist_western.hdf', 'r') as f:
        filelist = f['filelist'][()]

    filelist = [os.path.join(path, file.decode()) for file in filelist][0:10]

    dataset = Dataset(filelist, metadata='geo_temporal_inci')

    # dataset.get_incidence_dependence()
    dataloader = DataLoader(dataset, batch_size=128)

    for img, label, extral in dataloader:
        print(extral.shape)

    # data = Dataset(filelist, img_size=30, stride=30, num_classes=11, metadata='inci')
    # data.get_incidence_dependence()
    # for img, label, loc in train_loader:
    #     print(img[0].shape)
    #     print(img[1].shape)
    #     # print(img[0])
    #     # print(img[1])
    #     break

    # path = r'E:\sea_ice_classification\data\charts_gridded\Western Arctic'.replace('\\', '/')
    # filelist = []
    # for year in ['2019', '2020']:
    #     filelist.extend([os.path.join(path, year, file) for file in os.listdir(os.path.join(path, year))])
    #
    # get_temporal_spatial_distribution(filelist)
