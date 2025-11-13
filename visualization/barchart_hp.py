import matplotlib.pyplot as plt
import pandas as pd


facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh',  'MS', 'WS', 'D', 'PS', 'BS']
# 确保颜色数量与标签数量匹配（9种颜色）
facies_colors = ['#fd9700', '#fd9700', '#fd9700', '#fd9700', '#fd9700', '#fd9700', '#fd9700', '#fd9700', '#fd9700']


#读取数据集
data = pd.read_excel("../dataset/hp.xlsx")


data.dropna(inplace=True)

# 计算每个岩性的样本数和百分比
facies_counts = data['Facies'].value_counts().sort_index()
total = facies_counts.sum()
percentages = (facies_counts / total * 100).round(1)

# 应用岩性标签和颜色
facies_counts.index = facies_labels

# 创建图形
plt.figure(figsize=(12, 10))
ax = facies_counts.plot(kind='bar',
                        color=facies_colors,
                        edgecolor='black')

# 设置坐标轴标签
plt.xlabel('Facies', fontsize=14)
plt.ylabel('Number', fontsize=14)
plt.xticks(rotation=45, ha='right')

# 增加刻度标签设置
ax.tick_params(axis='both', which='major', labelsize=11)  # <-- 新增这行


# 在柱顶添加单行双标签
for i, (count, pct) in enumerate(zip(facies_counts, percentages)):
    ax.text(i,
            count + max(facies_counts)*0.005,  # 垂直偏移量
            f"{count} ({pct}%)",  # 用空格代替换行符
            ha='center',
            va='bottom',
            fontsize=11,
            linespacing=0.8)


# 调整布局
plt.tight_layout()
plt.show()

# 返回统计结果
pd.DataFrame({
    '岩相类型': facies_labels,
    '样本数量': facies_counts.values,
    '百分比(%)': percentages.values
})
