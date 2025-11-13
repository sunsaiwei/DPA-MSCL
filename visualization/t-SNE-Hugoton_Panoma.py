import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from utils.utils import *
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap


# 1. 读取本地 Excel 数据
file_path_Hugoton_Panoma = "../dataset/hp.xlsx"
df_Hugoton_Panoma = pd.read_excel(file_path_Hugoton_Panoma)
df_Hugoton_Panoma.dropna(inplace=True)

# 2. 提取特征和标签
features_Hugoton_Panoma = df_Hugoton_Panoma.loc[:,
    ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS", "Facies"]].values
labels_Hugoton_Panoma = df_Hugoton_Panoma.loc[:, "Facies"].values - 1

# 调整后的岩性配色方案（色盲友好+连续渐变色）
lithology_info = {
    0: ('SS', '#4E79A7', 'Gravelly Sandstone'),      # 深蓝
    1: ('CSiS', '#A0CBE8', 'Coarse Siltstone'),      # 浅蓝
    2: ('FSiS', '#F28E2B', 'Fine-grained Sandstone'),# 橙黄
    3: ('SiSH', '#E15759', 'Siliceous Shale'),       # 红
    4: ('MS', '#76B7B2', 'Marl stone'),              # 青绿
    5: ('WS', '#59A14F', 'Wacke stone'),             # 绿
    6: ('D', '#EDC948', 'Dolo stone'),               # 金
    7: ('PS', '#B07AA1', 'Pack stone'),              # 紫
    8: ('BS', '#FF9DA7', 'Bound stone')              # 浅粉
}

# 标准化处理
scaler = StandardScaler()
features_Hugoton_Panoma = scaler.fit_transform(features_Hugoton_Panoma)

# ===== 新增类间距计算模块 ===== #
# 计算类中心
unique_labels = np.unique(labels_Hugoton_Panoma)
class_centers = {}
for label in unique_labels:
    mask = labels_Hugoton_Panoma == label
    class_centers[label] = features_Hugoton_Panoma[mask].mean(axis=0)

# 构建距离矩阵
distance_matrix = np.zeros((len(unique_labels), len(unique_labels)))
for i in unique_labels:
    for j in unique_labels:
        distance_matrix[i, j] = np.linalg.norm(class_centers[i] - class_centers[j])

# 统计分析
triu_indices = np.triu_indices_from(distance_matrix, k=1)
print(f"[类间距离分析]\n"
      f"平均距离: {distance_matrix[triu_indices].mean():.4f}\n"
      f"最小距离: {distance_matrix[triu_indices].min():.4f}\n"
      f"最大距离: {distance_matrix[triu_indices].max():.4f}\n")

# 可选：可视化距离热力图
plt.figure(figsize=(10, 8))
sns.heatmap(distance_matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=lithology_info.keys(),
            yticklabels=lithology_info.keys())
plt.show()
# ===== 结束新增模块 ===== #

# 3. t-SNE 降维
tsne_Hugoton_Panoma = TSNE(
    n_components=3,
    perplexity=10,
    learning_rate=500,
    random_state=250
)
features_Hugoton_Panoma_3d = tsne_Hugoton_Panoma.fit_transform(features_Hugoton_Panoma)

# 4. 增强可视化
fig = plt.figure(figsize=(9.6, 6.4))
ax = fig.add_subplot(111, projection='3d')

# # 岩性颜色配置（高亮度方案）
# lithology_info = {
#     0: ('SS', '#FF00FF', 'Gravelly Sandstone'),       # 品红
#     1: ('CSiS', '#00BFFF', 'Coarse Siltstone'),       # 深天蓝
#     2: ('FSiS', '#00FF00', 'Fine-grained Sandstone'), # 亮绿
#     3: ('SiSH', '#FF4500', 'Siliceous Shale'),        # 橙红
#     4: ('MS', '#FFD700', 'Marl stone'),               # 保持原金色
#     5: ('WS', '#FF6347', 'Wacke stone'),              # 番茄红
#     6: ('D', '#7CFC00', 'Dolo stone'),                # 草坪绿
#     7: ('PS', '#40E0D0', 'Pack stone'),               # 绿松石
#     8: ('BS', '#DA70D6', 'Bound stone')               # 兰紫色
# }





# 保持后续代码完全不变...
# 创建自定义颜色映射
cmap = ListedColormap([v[1] for v in lithology_info.values()])

# 绘制散点图（参数保持不变）
sc = ax.scatter(
    features_Hugoton_Panoma_3d[:, 0],
    features_Hugoton_Panoma_3d[:, 1],
    features_Hugoton_Panoma_3d[:, 2],
    c=labels_Hugoton_Panoma,
    vmin=0,  # ← 新增这个
    vmax=8,  # ← 新增这个
    cmap=cmap,
    alpha=0.65,  # 降低透明度增强层次感
    s=28,  # 减小点尺寸
    edgecolors='none',
    linewidths=0.3,  # 边框粗细
    depthshade=True  # 保持深度阴影
)

# 后续所有图例、标签、布局设置保持原样...


# # 双图例系统
# # 颜色条（右侧）
# cbar = plt.colorbar(sc, pad=0.12,
#                   ticks=np.arange(0,9,1),  # 显式指定0-8的整数
#                   boundaries=np.arange(-0.5,9,1))
# cbar.set_label('Lithology Codes', rotation=270, labelpad=20)
# cbar.ax.set_yticklabels([v[0] for v in lithology_info.values()])
#
# # 独立图例框（正下方）
# legend_elements = [plt.Line2D([0], [0],
#                    marker='o',
#                    color='w',
#                    label=f'{v[0]} - {v[2]}',
#                    markerfacecolor=v[1],
#                    markersize=8) for v in lithology_info.values()]
#
# # 将图例定位在画布正下方
# ax.legend(handles=legend_elements,
#          title='Lithological Facies',
#          bbox_to_anchor=(0.5, -0.15),  # 水平居中，垂直位置在画布下方15%处
#          loc='upper center',           # 锚点定位在上部中心
#          borderaxespad=0.,
#          ncol=3,                       # 分3列显示
#          fontsize=8,
#          frameon=False)
#
# # 调整底部留白空间
# plt.subplots_adjust(bottom=0.15)  # 增大底部空间
#
#
#
#
# # 坐标轴标签
# ax.set_xlabel("t-SNE Component 1", labelpad=10)
# ax.set_ylabel("t-SNE Component 2", labelpad=10)
# ax.set_zlabel("t-SNE Component 3", labelpad=10)
# ax.set_title("hp Dataset 3D t-SNE Projection", pad=15)

# 单图例系统
# 独立图例框（右上方垂直排列）
legend_elements = [plt.Line2D([0], [0],
                   marker='o',
                   color='w',
                   label=v[0],
                   markerfacecolor=v[1],
                   markersize=8) for v in lithology_info.values()]

# 添加图例并设置位置样式
ax.legend(handles=legend_elements,
         loc='upper right',
         bbox_to_anchor=(0.94, 0.9),
         fontsize=10,            # 调小字体
         frameon=True,
         fancybox=True,
         framealpha=0.7,        # 调低框透明度
         borderpad=0.6,
         edgecolor='#404040')   # 添加边框颜色

plt.tight_layout()
plt.show()

