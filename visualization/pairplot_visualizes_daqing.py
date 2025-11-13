"""
此模块用于可视化大庆数据集的交汇图矩阵
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# 读取测井数据集
df = pd.read_excel("../dataset/daqing.xlsx")
df.dropna(inplace=True)



# 选择数值型测井曲线数据进行可视化
selected_features = ["SP", "CNL", "PE", "GR", "AC", "AT10", "Face"]  # 选择你感兴趣的测井曲线
data = df[selected_features].dropna()  # 去除 NaN 数据

#分离出特征
Features = data.drop(selected_features, axis=1)
#分离出标签
Face = data['Face']

# 映射类别名称
data['Face'] = data['Face'].map({
    0: 'SS',
    1: 'DS',
    2: 'HS',
    3: 'Hgs',
    4: 'BS',
    5: 'T',

})

facies_palette = {
    'SS': '#4A8EC6',   # 类别0 - 砂岩（继承默认首色蓝色基调）
    'DS': '#EB9A4F',   # 类别1 - 泥岩（延续第二色橙色系）
    'HS': '#5DA05D',   # 类别2 - 页岩（沿用第三色绿色变体）
    'Hgs': '#C54F4F', # 类别3 - 硅质页岩（保留第四位红色调性）
    'BS': '#9A77BD',  # 类别4 - 基岩（匹配第五色紫色系）
    'T': '#A07166'    # 类别5 - 过渡岩（承接末位棕色系）
}




sns.pairplot(data, hue='Face', palette=facies_palette, diag_kind='kde', plot_kws={'edgecolor': 'white', 'linewidth': 0.5, 'alpha': 0.8})
plt.show()

