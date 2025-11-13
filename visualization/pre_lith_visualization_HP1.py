"""
读取测井数据和预测数据，并绘制地层剖面图，用于对比真实地层和预测结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import set_option
import torch

set_option("display.max_rows", 10)#设置要显示的默认行数，显示的最大行数是10
pd.options.mode.chained_assignment = None #为了在增加列表行数的时候防止出现setting with copy warning
# 读取数据
df = pd.read_excel("../dataset/hp.xlsx")
df.dropna(inplace=True)
blind_data = df[df['Well Name'] == 'STUART'].dropna()
print(blind_data.shape)

# 读取多个预测结果文件
file_path_ours = "../datasave/blind_HP/y_pre/DPA_BCL.txt"
file_path_smote = "../datasave/blind_HP/y_pre/SMOTE-1D-CNN.txt"
file_path_sgan = "../datasave/blind_HP/y_pre/SGAN.txt"

# 读取DPA_BCL预测数据
with open(file_path_ours, "r") as file:
    data_list_ours = []
    for line in file:
        data_list_ours.append(int(line.strip()))
data_list_ours = data_list_ours[-474:]  # 直接取后474条
pre_log_ours = torch.tensor(data_list_ours)

# 读取SMOTE-1D-CNN预测数据
with open(file_path_smote, "r") as file:
    data_list_smote = []
    for line in file:
        data_list_smote.append(int(line.strip()))
data_list_smote = data_list_smote[-474:]  # 取后474条
pre_log_smote = torch.tensor(data_list_smote)

# 读取SGAN预测数据
with open(file_path_sgan, "r") as file:
    data_list_sgan = []
    for line in file:
        data_list_sgan.append(int(line.strip()))
data_list_sgan = data_list_sgan[-474:]  # 取后474条
pre_log_sgan = torch.tensor(data_list_sgan)

blind_data = blind_data

print("pre_Log DPA_BCL:", pre_log_ours.shape)
print("pre_Log SMOTE-1D-CNN:", pre_log_smote.shape)
print("pre_Log SGAN:", pre_log_sgan.shape)

# 定义9种地层的颜色和标签
facies_colors = ['#f4d03e', '#f5b041', '#db7733', '#6f2c00', '#194e70', '#2d86c0', '#aed6f0', '#a56abd', '#196f3e']

facies_labels = [
    "SS",
    "CSiS",
    "FSiS",
    "SiSh",
    "MS",
    "WS",
    "D",
    "PS",
    "BS"
]
# 创建标签与颜色的映射
facies_color_map = {}
for ind, label in enumerate(facies_labels):
    facies_color_map[label] = facies_colors[ind]

font = FontProperties(family='Times New Roman', size=18)

def make_facies_log_plot(logs, pre_log_ours, pre_log_smote, pre_log_sgan, facies_colors):
    # 确认测井数据是以深度排序的
    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(facies_colors, 'indexed')

    ztop = logs.Depth.min()
    zbot = logs.Depth.max()

    # 生成地层分类和预测的地层数据
    cluster = np.repeat(np.expand_dims(logs['Facies'].values - 1, 1), 100, 1)
    pre_cluster_ours = np.repeat(np.expand_dims(pre_log_ours.numpy(), 1), 100, 1)
    pre_cluster_smote = np.repeat(np.expand_dims(pre_log_smote.numpy(), 1), 100, 1)
    pre_cluster_sgan = np.repeat(np.expand_dims(pre_log_sgan.numpy(), 1), 100, 1)

    # 创建包含所有预测结果的图表（增加到9个子图）
    f, ax = plt.subplots(nrows=1, ncols=9, figsize=(18, 18))

    ax[0].plot(logs.GR, logs.Depth, '-g', linewidth=1)
    ax[1].plot(logs.ILD_log10, logs.Depth, '-', linewidth=1)
    ax[2].plot(logs.DeltaPHI, logs.Depth, '-', color='0.5', linewidth=1)
    ax[3].plot(logs.PHIND, logs.Depth, '-', color='r', linewidth=1)
    ax[4].plot(logs.PE, logs.Depth, '-', color='black', linewidth=1)

    # 按照要求放置数据：真实数据、SGAN（倒数第三）、SMOTE-1D-CNN（倒数第二）、DPA_BCL（最后一个）
    im = ax[5].imshow(cluster, interpolation='none', aspect='auto',
                      cmap=cmap_facies, vmin=0, vmax=8)
    ax[6].imshow(pre_cluster_smote, interpolation='none', aspect='auto',
                 cmap=cmap_facies, vmin=0, vmax=8)  # SMOTE-1D-CNN在倒数第三
    ax[7].imshow(pre_cluster_sgan, interpolation='none', aspect='auto',
                 cmap=cmap_facies, vmin=0, vmax=8)  # SGAN在倒数第二
    ax[8].imshow(pre_cluster_ours, interpolation='none', aspect='auto',
                 cmap=cmap_facies, vmin=0, vmax=8)  # DPA_BCL在最后

    divider = make_axes_locatable(ax[8])

    # 或者使用更精确的中间位置
    tick_positions = np.arange(len(facies_labels))

    cax = divider.append_axes("right", size="30%", pad=0.1)
    # 修改后的colorbar部分代码
    cbar = plt.colorbar(im, cax=cax, ticks=tick_positions)
    cbar.ax.set_yticklabels(facies_labels, fontproperties=font)
    cbar.set_label((19 * ' ').join(['BS', 'PS', 'D', 'WS', 'MS', 'SiSh', 'FSiS', 'CSiS', 'SS']), fontproperties=font)

    # 关键修改部分：删除其他刻度线
    for idx, tick_line in enumerate(cbar.ax.yaxis.get_ticklines()):
        if idx != 0:  # 只保留第一个刻度线（最底部）
            tick_line.set_visible(False)
    cbar.set_ticklabels('')

    for i in range(len(ax) - 4):  # 更新循环范围以适应新的子图数量
        ax[i].set_ylim(ztop, zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
        ax[i].tick_params(axis='y', labelsize=18)

    ax[0].set_xlabel("GR", fontproperties=font)
    ax[0].set_xlim(logs.GR.min(), logs.GR.max())
    ax[1].set_xlabel("ILD_log10", fontproperties=font)
    ax[1].set_xlim(logs.ILD_log10.min(), logs.ILD_log10.max())
    ax[2].set_xlabel("DeltaPHI", fontproperties=font)
    ax[2].set_xlim(logs.DeltaPHI.min(), logs.DeltaPHI.max())
    ax[3].set_xlabel("PHIND", fontproperties=font)
    ax[3].set_xlim(logs.PHIND.min(), logs.PHIND.max())
    ax[4].set_xlabel("PE", fontproperties=font)
    ax[4].set_xlim(logs.PE.min(), logs.PE.max())

    ax[5].set_xlabel('Facies', fontproperties=font)
    ax[6].set_xlabel('SMOTE-CNN', fontproperties=font)
    ax[7].set_xlabel('SGAN', fontproperties=font)
    ax[8].set_xlabel('DPA-MSCL', fontproperties=font)

    # 设置xticklabels为空
    for i in range(5, 9):
        ax[i].set_xticklabels([])

    # 设置yticklabels为空
    for i in range(1, 9):
        ax[i].set_yticklabels([])

    f.suptitle('Well: %s' % logs.iloc[0]['Well Name'], fontsize=16, y=0.94)
    return f, ax

# 调用绘图函数
fig, axes = make_facies_log_plot(
    blind_data, pre_log_ours, pre_log_smote, pre_log_sgan,
    facies_colors)

# 明确指定字体，例如宋体（SimSun）
font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=12)

# 在绘图之前设置字体
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

plt.show()
