import h5py
import numpy as np

from dataset import Dataset
import os


with h5py.File('./datasets/train_filelist_western.hdf', 'r') as f:
    train_filelist = f['filelist'][()]

with h5py.File('./datasets/test_filelist_western.hdf', 'r') as f:
    test_filelist = f['filelist'][()]

print(len(train_filelist))
print(len(test_filelist))

data_dir = '/home/ubuntu/LiZhao/201910to202009_hdf_ASI_cut_full_info/'
train_filelist = [os.path.join(data_dir, file) for file in train_filelist]
test_filelist = [os.path.join(data_dir, file) for file in test_filelist]

train_data = Dataset(train_filelist, img_size=30, stride=30, num_classes=11)
test_data = Dataset(test_filelist, img_size=30, stride=30, num_classes=11)

train_class_num = {x: 0 for x in range(11)}
for _, label, _ in train_data:
    train_class_num[int(label)] += 1

test_class_num = {x: 0 for x in range(11)}
for _, label, _ in test_data:
    test_class_num[int(label)] += 1

train_class_num = [x for _, x in train_class_num.items()]
test_class_num = [x for _, x in test_class_num.items()]

train_class_ratio = np.array(train_class_num) / sum(train_class_num)
test_class_ratio = np.array(test_class_num) / sum(test_class_num)

with h5py.File('./datasets/data_dist.hdf', 'w') as f:
    f.create_dataset('train_class_num', data=train_class_num)
    f.create_dataset('test_class_num', data=test_class_num)
    f.create_dataset('train_class_ratio', data=train_class_ratio)
    f.create_dataset('test_class_ratio', data=test_class_ratio)
