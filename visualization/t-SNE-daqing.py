import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from utils.utils import *
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap



# 1. 读取本地 Excel 数据
file_path_daqing = "../dataset/daqing.xlsx"
df_daqing = pd.read_excel(file_path_daqing)
df_daqing.dropna(inplace=True)

#去除岩性为凝灰岩的数据，也就是Face=5的数据
df_daqing = df_daqing[df_daqing['Face']!= 5]

# 2. 提取特征和标签
features_daqing = df_daqing.loc[:,
    ["SP", "PE", "GR", "AT10", "AT20", "AT30", "AT60", "AT90", "AC", "CNL", "DEN", "POR_index", "Ish", "Face"]].values
labels_daqing = df_daqing.loc[:, "Face"].values

# 标准化处理
scaler = StandardScaler()
features_daqing = scaler.fit_transform(features_daqing)
# 岩性颜色配置（中等饱和度方案）
lithology_info = {
    0: ('SS', '#6a4c93'),       # 薰衣草紫
    1: ('DS', '#1982c4'),      # 钴蓝色
    2: ('HS', '#8ac926'),# 嫩绿色
    3: ('Hgs', '#ff595e'),     # 珊瑚红
    4: ('BS', '#ffca3a'),     # 芥末黄
    5: ('T', '#a5a5a5')         # 灰色
}

# ===== 新增类间距计算模块 ===== #
# 计算类中心
unique_labels = np.unique(labels_daqing)
class_centers = {}
for label in unique_labels:
    mask = labels_daqing == label
    class_centers[label] = labels_daqing[mask].mean(axis=0)

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
plt.title("Inter-class Distance Matrix")
plt.show()
# ===== 结束新增模块 ===== #

# 3. t-SNE 降维
tsne_daqing = TSNE(
    n_components=3,
    perplexity=40,  # 较原始值15适当增大（建议范围30-50）
    learning_rate=200,  # 较原始500降低（建议范围100-300）
    random_state=3024,
)
features_daqing_3d = tsne_daqing.fit_transform(features_daqing)

# 4. 增强可视化
fig = plt.figure(figsize=(9.6, 6.4))
ax = fig.add_subplot(111, projection='3d')





# 保持后续代码完全不变...
# 创建自定义颜色映射
cmap = ListedColormap([v[1] for v in lithology_info.values()])

# 绘制散点图（参数保持不变）
sc = ax.scatter(
    features_daqing_3d[:, 0],
    features_daqing_3d[:, 1],
    features_daqing_3d[:, 2],
    c=labels_daqing,
    vmin=0,  # ← 新增这个
    vmax=5 ,  # ← 新增这个
    cmap=cmap,
    alpha=0.65,        # 降低透明度增强层次感
    s=28,              # 减小点尺寸
    edgecolors='none',
    linewidths=0.3,    # 边框粗细
    depthshade=True    # 保持深度阴影
)

# 后续所有图例、标签、布局设置保持原样...


# # 双图例系统
# # 颜色条（右侧）
# cbar = plt.colorbar(sc, pad=0.12,
#                   ticks=np.arange(0,6,1),  # 显式指定0-5的整数
#                   boundaries=np.arange(-0.5,6,1))
# cbar.set_label('Lithology Codes', rotation=270, labelpad=20)
# cbar.ax.set_yticklabels([v[0] for v in lithology_info.values()])

# # 独立图例框（正下方）
# legend_elements = [plt.Line2D([0], [0],
#                    marker='o',
#                    color='w',
#                    label=f'{v[0]} - {v[1]}',
#                    markerfacecolor=v[1],
#                    markersize=8) for v in lithology_info.values()]

# # 将图例定位在画布正下方
# ax.legend(handles=legend_elements,
#          # title='Lithological Facies',
#          bbox_to_anchor=(0.5, -0.15),  # 水平居中，垂直位置在画布下方15%处
#          loc='upper center',           # 锚点定位在上部中心
#          borderaxespad=0.,
#          ncol=3,                       # 分3列显示
#          fontsize=8,
#          frameon=False)
#
# # 调整底部留白空间
# plt.subplots_adjust(bottom=0.15)  # 增大底部空间

# 坐标轴标签
# ax.set_xlabel("t-SNE Component 1", labelpad=10)
# ax.set_ylabel("t-SNE Component 2", labelpad=10)
# ax.set_zlabel("t-SNE Component 3", labelpad=10)
# ax.set_title("DaQing Dataset 3D t-SNE Projection", pad=15)

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

