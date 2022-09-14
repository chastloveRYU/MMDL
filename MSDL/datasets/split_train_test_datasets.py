import os
import h5py
import random

# set random seed
random.seed(37)

path = r'E:\sea_ice_classification\data\S1\Eastern Arctic\201910to202009_sar_asi_charts'.replace('\\', '/')
filelist = os.listdir(path)

filelist_mon = {x: [] for x in range(1, 13)}
print(filelist_mon)

for file in filelist:
    mon = int(file.split('_')[4][4:6])
    filelist_mon[mon].append(file)

train_filelist = []
test_filelist = []
for i in range(1, 13):
    file = filelist_mon[i]
    random.shuffle(file)
    file_len = len(file)
    train_len = int(0.9 * len(file))
    test_len = file_len - train_len

    train_file = random.sample(file, train_len)
    test_file = list(set(file) ^ set(train_file))

    train_filelist.extend(train_file)
    test_filelist.extend(test_file)

print(len(train_filelist))
print((len(test_filelist)))

print(list(set(train_filelist) ^ set(test_filelist)))
print(len(list(set(train_filelist) ^ set(test_filelist))))
# filelist_len = len(filelist)
# train_len = int(0.9 * len(filelist))
# test_len = filelist_len - train_len
#
# train_filelist = random.Random(66).sample(filelist, train_len)
# test_filelist = list(set(filelist) ^ set(train_filelist))

with h5py.File('./train_filelist_eastern.hdf', 'w') as f:
    f.create_dataset('filelist', data=train_filelist)

with h5py.File('./test_filelist_eastern.hdf', 'w') as f:
    f.create_dataset('filelist', data=test_filelist)

