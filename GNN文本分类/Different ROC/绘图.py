import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np  # For mathematical calculations

# 设置matplotlib的全局字体
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 14

# 读取Excel文件
df = pd.read_excel('111.xlsx')

# 数据标签在第一列
labels = df.iloc[:, 0]

# 从第二列到第五列是数据
data = df.iloc[:, 1:5]

# 创建画布
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
axes = axes.flatten()

# 对每一列进行处理
for i, (column_name, column_data) in enumerate(data.items()):
    # 根据标签分组
    positive_data = column_data[labels == 1]
    negative_data = column_data[labels == 0]

    # 计算正样本和负样本的均值和标准差
    pos_mean = positive_data.mean()
    pos_std = positive_data.std()
    neg_mean = negative_data.mean()
    neg_std = negative_data.std()

    # 计算样本大小
    pos_n = positive_data.count()
    neg_n = negative_data.count()

    # 计算置信区间
    pos_ci = 1.96 * pos_std / np.sqrt(pos_n)
    neg_ci = 1.96 * neg_std / np.sqrt(neg_n)

    # 绘制直方图和核密度估计
    sns.histplot(positive_data, bins=15, kde=True, color='blue', alpha=0.6, ax=axes[i])
    sns.histplot(negative_data, bins=15, kde=True, color='red', alpha=0.6, ax=axes[i])
    # 设置标题
    axes[i].set_title(f'{column_name}')

    # 添加均值、标准差和置信区间
    axes[i].text(0.80, 0.95,
                 f'Pos Mean: {pos_mean:.2f}\nPos Std: {pos_std:.2f}\nPos CI: [{pos_mean - pos_ci:.2f}, {pos_mean + pos_ci:.2f}]\nNeg Mean: {neg_mean:.2f}\nNeg Std: {neg_std:.2f}\nNeg CI: [{neg_mean - neg_ci:.2f}, {neg_mean + neg_ci:.2f}]',
                 transform=axes[i].transAxes, verticalalignment='top', horizontalalignment='right',
                 fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

# 调整子图间距
plt.tight_layout()

# 保存图像，分辨率为600 dpi
plt.savefig('distribution_by_labels_with_CI.jpg', dpi=600)

# 显示图像
plt.show()