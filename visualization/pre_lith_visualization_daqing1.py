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
df = pd.read_excel("../dataset/daqing_blind_data.xlsx")
df.dropna(inplace=True)
blind_data = df[df['井'] == '乐34'].dropna()
print(blind_data.shape)

file_path = "../datasave/blind_daqing/y_pre/DPA-BCL.txt"
data_list = []

# 打开文件进行读取操作
with open(file_path, "r") as file:
    # 逐行读取文件内容
    for line in file:
        # 将读取的行转换为整数，并添加到数据列表中
        data_list.append(int(line.strip()))
data_list = data_list[105:137]
# 将数据列表转换为张量
res= torch.tensor(data_list)

pre_log=res

blind_data = blind_data

# 将数据列表转换为张量
res = torch.tensor(data_list)
pre_log = res
print("pre_Log:", pre_log.shape)

# 定义9种地层的颜色和标签
facies_colors = ['#0070b5', '#a76ab5', '#ffcb21', '#e07229', '#a4cff0', '#ffb232']

facies_labels = [
    "SS",
    "DS",
    "HS",
    "Hgs",
    "BS",
    "T"
]
# 创建标签与颜色的映射
facies_color_map = {}
for ind, label in enumerate(facies_labels):
    facies_color_map[label] = facies_colors[ind]

font = FontProperties(family='Times New Roman', size=18)
def make_facies_log_plot(logs,pre_log, facies_colors):
    # 确认测井数据是以深度排序的
    logs = logs.sort_values(by='顶深')
    cmap_facies = colors.ListedColormap(facies_colors, 'indexed')

    ztop = logs.顶深.min()
    zbot = logs.顶深.max()

    # 生成地层分类和预测的地层数据
    cluster = np.repeat(np.expand_dims(logs['Face'].values, 1), 100, 1)
    pre_cluster = np.repeat(np.expand_dims(pre_log.numpy(), 1), 100, 1)


    f, ax = plt.subplots(nrows=1, ncols=8, figsize=(14, 16))

    ax[0].plot(logs.SP, logs.顶深, '-g', linewidth=1)
    ax[1].plot(logs.PE, logs.顶深, '-', linewidth=1)
    ax[2].plot(logs.GR, logs.顶深, '-', color='0.5', linewidth=1)
    ax[3].plot(logs.AC, logs.顶深, '-', color='r', linewidth=1)
    ax[4].plot(logs.CNL, logs.顶深, '-', color='black', linewidth=1)
    ax[5].plot(logs.AT10, logs.顶深, '-', color='purple', linewidth=1)


    im = ax[6].imshow(cluster, interpolation='none', aspect='auto',
                      cmap=cmap_facies, vmin=0, vmax=5)
    ax[7].imshow(pre_cluster, interpolation='none', aspect='auto',
                 cmap=cmap_facies, vmin=0, vmax=5)

    divider = make_axes_locatable(ax[7])

    # 或者使用更精确的中间位置
    tick_positions = np.arange(len(facies_labels))

    cax = divider.append_axes("right", size="30%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ticks=tick_positions)
    cbar.ax.set_yticklabels(facies_labels, fontproperties=font)
    cbar.set_label((19 * ' ').join(["SS","DS","HS","Hgs","BS","T"]), fontproperties=font)

    cbar.set_ticklabels('')


    for i in range(len(ax) - 2):
        ax[i].set_ylim(ztop, zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
        ax[i].tick_params(axis='y', labelsize=18)

    ax[0].set_xlabel("SP", fontproperties=font)
    ax[0].set_xlim(logs.SP.min(), logs.SP.max())
    ax[1].set_xlabel("PE", fontproperties=font)
    ax[1].set_xlim(logs.PE.min(), logs.PE.max())
    ax[2].set_xlabel("GR", fontproperties=font)
    ax[2].set_xlim(logs.GR.min(), logs.GR.max())
    ax[3].set_xlabel("AC", fontproperties=font)
    ax[3].set_xlim(logs.AC.min(), logs.AC.max())
    ax[4].set_xlabel("CNL", fontproperties=font)
    ax[4].set_xlim(logs.CNL.min(), logs.CNL.max())
    ax[5].set_xlabel("AT_log10", fontproperties=font)
    ax[5].set_xlim(logs.AT10.min(), logs.AT10.max())

    ax[6].set_xlabel('Face', fontproperties=font)
    ax[7].set_xlabel('Predict Face', fontproperties=font)

    ax[6].set_xticklabels([])
    ax[7].set_xticklabels([])

    ax[1].set_yticklabels([])
    ax[2].set_yticklabels([])
    ax[3].set_yticklabels([])
    ax[4].set_yticklabels([])
    ax[5].set_yticklabels([])


    ax[6].set_yticklabels([])
    ax[7].set_yticklabels([])

    f.suptitle('Well: %s' % logs.iloc[0]['井'], fontsize=14, y=0.94)

make_facies_log_plot(
    blind_data,pre_log,
    facies_colors)

# 明确指定字体，例如宋体（SimSun）
font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=12)

# 在绘图之前设置字体
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

plt.show()


# 调用绘图函数
make_facies_log_plot(blind_data, pre_log, facies_colors)
