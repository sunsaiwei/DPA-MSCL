"""
该模块用于绘制箱线图，以便对比不同类别的测井曲线，包含岩性颜色图例
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

# 读取测井数据集
df = pd.read_excel("../dataset/hp.xlsx")

# 去除缺失值
df.dropna(inplace=True)

# 选择数值型测井曲线进行可视化
selected_features = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS"]
category_column = 'Facies'

# 岩性类别映射字典
face_mapping = {
    1: 'SS',
    2: 'CSiS',
    3: 'FSiS',
    4: 'SiSh',
    5: 'Ms',
    6: 'WS',
    7: 'D',
    8: 'PS',
    9: 'BS',
}

# 数据预处理
df_selected = df[selected_features + [category_column]].copy()
df_selected[selected_features] = df_selected[selected_features].apply(lambda x: x + 1e-10).apply(np.log1p)  # 添加一个很小的正值以避免log1p计算出负无穷或零
df_selected[selected_features] = df_selected[selected_features].replace([-np.inf, np.inf], np.nan).dropna()  # 将负无穷和正无穷替换为NaN，然后去除这些行

# 转换数字为岩性名称
df_selected[category_column] = df_selected[category_column].map(face_mapping)

# 标准化处理
scaler = StandardScaler()
df_selected[selected_features] = scaler.fit_transform(df_selected[selected_features])

# 获取岩性类别及其对应颜色（按原始数字顺序）
categories = [face_mapping[i] for i in sorted(face_mapping.keys())]
palette = sns.color_palette("Set2", n_colors=len(categories))

# 设置图形
sns.set(style="whitegrid")
plt.figure(figsize=(20, 15))

# 绘制箱线图
for i, feature in enumerate(selected_features, 1):
    plt.subplot(2, 4, i)
    sns.boxplot(data=df_selected, x=category_column, y=feature, hue=category_column, palette=palette, order=categories, legend=False)

    plt.title(f"{feature}")
    plt.xlabel("Lithology")
    plt.ylabel("Standardized Value")
    plt.xticks(rotation=45)

# 在最后一个子图位置添加图例
plt.subplot(2, 4, 8)
plt.axis('off')
legend_handles = [plt.Rectangle((0,0),1,1, color=color, ec="k") for color in palette]
plt.legend(legend_handles, categories,
           title="Lithology Colors",
           loc="center",
           frameon=True,
           borderpad=1,
           handlelength=1.5,
           handleheight=1.5)

plt.tight_layout()
plt.show()
