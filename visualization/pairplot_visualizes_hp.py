"""
此模块用于可视化大庆数据集的交汇图矩阵
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# 读取测井数据集
df = pd.read_excel("../dataset/hp.xlsx")
df.dropna(inplace=True)



# 选择数值型测井曲线数据进行可视化
selected_features = ["GR", "DeltaPHI", "ILD_log10", "PHIND", "PE", "Facies"]  # 选择你感兴趣的测井曲线
data = df[selected_features].dropna()  # 去除 NaN 数据

#分离出特征
Features = data.drop(selected_features, axis=1)
#分离出标签
Face = data['Facies']

# 映射类别名称
data['Facies'] = data['Facies'].map({
    1: 'SS',
    2: 'CSiS',
    3: 'FSiS',
    4: 'SiSh',
    5: 'Ms',
    6: 'WS',
    7: 'D',
    8: 'PS',
    9: 'BS',
})

# 改良版配色方案（基于默认色系调整）
facies_palette = {
    'SS': '#4A8EC6',   # 蓝色：降原默认饱和度40%，保持冷调
    'CSiS': '#EB9A4F', # 橙色：降饱和度30%，适当增加明度
    'FSiS': '#5DA05D', # 绿色：保持植被绿基础，降饱和度25%
    'SiSh': '#9A77BD', # 紫色：降饱和度35%，增加灰度感知
    'Ms': '#C54F4F',  # 红色：保核心色相，降饱和度30%
    'WS': '#8C8C8C',   # 中性灰：保持中间调性
    'D': '#D4B15A',   # 金色：降饱和度50%且灰化处理
    'PS': '#D991C2',  # 粉色：继承默认粉基调，降饱和度40%
    'BS': '#A07166'   # 棕色：土壤棕降饱和度20%
}


sns.pairplot(data, hue='Facies', palette=facies_palette, diag_kind='kde', plot_kws={'edgecolor': 'white', 'linewidth': 0.5, 'alpha': 0.8})
plt.show()
