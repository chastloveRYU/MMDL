import h5py
import matplotlib.pyplot as plt
import numpy as np

with h5py.File('./data_dist.hdf', 'r') as f:
    train_class_ratio = f['train_class_ratio'][()] * 100
    test_class_ratio = f['test_class_ratio'][()] * 100

# 柱形宽度
width = 0.2

# 柱形间隔
x1 = []
x2 = []
for i in range(11):
    x1.append(i)
    x2.append(i + width)

fontdict = {'family': 'times new roman',
            'size': 10.5,
            }
color = ['#6F6F6F', '#E88482', '#8E8BFE']

fig, ax = plt.subplots(figsize=(6, 4))
# 绘制柱形图1
bar1 = ax.bar(x1, train_class_ratio, width=width, label='Training dataset', color=color[0])

# 绘制柱形图2
bar2 = ax.bar(x2, test_class_ratio, width=width, label='Test dataset', color=color[1])

# 绘制柱形图3
# bar3 = ax.bar(x3, test_set, width=width, label='Test dataset', color=color[2])

plt.legend(handles=[bar1, bar2], frameon=False, ncol=1)
ax.set_xlabel('Surface type', fontdict=fontdict)
ax.set_ylabel('Percentage (%)', fontdict=fontdict)
label = ['OW', 'NI', 'GI', 'GWI', 'ThinFYI', 'MFYI',
         'ThickFYI', 'OI', 'SYI', 'MYI', 'Land']
ax.set_xticks(x2, label, rotation=15, fontproperties='Times New Roman')
plt.legend(loc='upper left', prop=fontdict)
plt.tight_layout()
plt.show()