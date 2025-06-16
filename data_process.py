import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
df = pd.read_csv('cancer_reg.csv')
#
# # 初步查看数据概况
print(df.info())
# print(df.describe())
# print(df.isnull().sum())
#
#
# plt.figure(figsize=(15, 12))  # 调整图表整体大小
# corr_matrix = df.corr(numeric_only=True)
#
# # 创建更清晰的热力图，调整格子大小和间距
# sns.heatmap(
#     corr_matrix,
#     annot=True,
#     fmt=".2f",
#     cmap='coolwarm',
#     linewidths=1,  # 增加格子之间的间距，使它们更清晰
#     square=True,     # 使每个格子为正方形
#     annot_kws={"size": 4}  # 调整注释文字大小
# )
#
# plt.title("Correlation Heatmap")
# plt.tight_layout()  # 确保布局紧凑
# plt.show()

# 可视化：收入与死亡率的关系
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x='medIncome', y='TARGET_deathRate', data=df)
# plt.title("The relationship between income and mortality.")
# plt.xlabel("Median Income")
# plt.ylabel("Death Rate")
# plt.grid(True)
# plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 加载数据
df = pd.read_csv("cancer_reg.csv")

# 创建目录保存图片
output_dir = "eda_plots"
os.makedirs(output_dir, exist_ok=True)
df['Incidence_Death_Ratio'] = df['incidenceRate'] / (df['TARGET_deathRate'] + 1e-6)
# 特征列表（横轴）与目标列（纵轴）
target = 'TARGET_deathRate'
features = [
    # 'MedianAge',
    # 'PctBachDeg25_Over',
    # 'medIncome',
    # 'povertyPercent',
    # 'PctEmployed16_Over',
    # 'PctPrivateCoverage',
    # 'PctPublicCoverage',
    # 'PercentMarried',
    # 'BirthRate'
    'incidenceRate',
    'Incidence_Death_Ratio'
]

# 开始绘图并保存
# for feature in features:
#     if feature not in df.columns:
#         print(f"跳过：列 '{feature}' 不存在")
#         continue
#
#     plt.figure(figsize=(10, 6))
#     sns.regplot(data=df, x=feature, y=target, scatter_kws={"s": 15}, line_kws={"color": "red"})
#     plt.xlabel(feature, fontsize=12)
#     plt.ylabel("Death Rate (per 100,000)", fontsize=12)
#     plt.title(f"{feature} vs. Death Rate", fontsize=14, fontweight='bold')
#     plt.grid(True)
#     plt.tight_layout()
#
#     # 保存图片
#     filename = os.path.join(output_dir, f"{feature}_vs_DeathRate.png")
#     plt.savefig(filename)
#     plt.close()
#
# print(f"✅ 所有图已保存至文件夹：{output_dir}")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os




# 计算基于90%数据分布的坐标轴范围
def calculate_axis_limits(data):
    """计算基于90%数据分布的坐标轴范围，排除极端异常值"""
    # 计算5%和95%分位数
    q_low = np.percentile(data, 5)
    q_high = np.percentile(data, 95)

    # 计算范围并添加小边距
    range_val = q_high - q_low
    margin = range_val * 0.05  # 额外添加5%的边距

    return [q_low - margin, q_high + margin]


for feature in features:
    if feature not in df.columns:
        print(f"跳过：列 '{feature}' 不存在")
        continue

    plt.figure(figsize=(10, 6))
    sns.regplot(data=df, x=feature, y=target, scatter_kws={"s": 15}, line_kws={"color": "red"})

    # 设置基于90%数据分布的横坐标范围
    x_limits = calculate_axis_limits(df[feature].dropna())  # 确保处理缺失值
    plt.xlim(x_limits)

    plt.xlabel(feature, fontsize=12)
    plt.ylabel("Death Rate (per 100,000)", fontsize=12)
    plt.title(f"{feature} vs. Death Rate", fontsize=14, fontweight='bold')
    plt.grid(True)
    plt.tight_layout()

    # 保存图片
    filename = os.path.join(output_dir, f"{feature}_vs_DeathRate.png")
    plt.savefig(filename)
    plt.close()

print(f"✅ 所有图已保存至文件夹：{output_dir}")