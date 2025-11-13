import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- 自定义部分 --------------------
# 1. 定义混淆矩阵的数值（手动填充每个格子）
custom_cm = np.array([
    [95, 3, 0, 2],   # 真实类别0的预测分布（行索引0）
    [1, 85, 4, 10],  # 真实类别1的预测分布（行索引1）
    [0, 2, 78, 5],   # 真实类别2的预测分布（行索引2）
    [3, 7, 2, 88]    # 真实类别3的预测分布（行索引3）
])

# 2. 定义类别名称（与矩阵行列对应）
class_names = ["Cat", "Dog", "Bird", "Fish"]

# 3. 自定义颜色和格式
cmap = "Blues"       # 颜色主题：可选 "Reds", "Greens", "viridis" 等
fmt = ".2f"            # 数值格式："d"（整数）或 ".2f"（小数）
# -----------------------------------------------------

# 生成热力图
plt.figure(figsize=(10, 8))
sns.heatmap(custom_cm,
            annot=True,          # 显示数值
            fmt=fmt,             # 数值格式
            cmap=cmap,          # 颜色主题
            xticklabels=class_names,
            yticklabels=class_names)

# 添加标题和标签
plt.title("Custom Confusion Matrix", fontsize=14, pad=20)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(rotation=45, ha='right')  # 调整标签旋转角度
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()