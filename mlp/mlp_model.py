import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import numpy as np

# Load the dataset
df = pd.read_csv('../new_cancer_reg.csv')

# Extract features and target variable
X = df.drop('TARGET_deathRate', axis=1)
y = df['TARGET_deathRate']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the MLP regressor
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1)

# Create a TensorBoard writer
writer = SummaryWriter('mlp_logs')

# Fit the model and log training loss and validation loss to TensorBoard
mlp.fit(X_train, y_train)
for epoch, loss in enumerate(mlp.loss_curve_):
    writer.add_scalar('train_loss', loss, epoch)
    writer.add_scalar('validation_loss', mlp.validation_scores_[epoch], epoch)

# Make predictions on the test set
y_pred = mlp.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(mlp.loss_curve_, label='Training Loss')

# Convert validation_scores_ to a NumPy array before negating
validation_losses = -np.array(mlp.validation_scores_)
plt.plot(validation_losses, label='Validation Loss')

plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('MLP Learning Curve')
plt.legend()
plt.show()

# Get all layer weights and plot a histogram
import numpy as np
import matplotlib.pyplot as plt

# 收集所有神经网络权重
all_weights = np.array([])
for coef in mlp.coefs_:
    all_weights = np.append(all_weights, coef.flatten())

# 创建画布和子图
plt.figure(figsize=(20, 10))  # 增加清晰度

# 绘制美化后的直方图
plt.hist(
    all_weights,
    bins=50,              # 保留原始分箱数量
    density=False,        # 显示频率而非密度
    alpha=0.7,            # 设置透明度
    color='#3498db',      # 使用更美观的蓝色
    edgecolor='#2980b9',  # 设置边缘颜色
    linewidth=0.8         # 设置边缘线宽
)

# 添加均值和中位数参考线
plt.axvline(x=np.mean(all_weights), color='#e74c3c', linestyle='--', linewidth=1.5, label=f'Mean: {np.mean(all_weights):.4f}')
plt.axvline(x=np.median(all_weights), color='#f39c12', linestyle='-.', linewidth=1.5, label=f'Median: {np.median(all_weights):.4f}')

# 美化图表
plt.xlabel('Weight Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('MLP Weight Distribution Histogram', fontsize=14, pad=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)  # 添加水平网格线
plt.legend(frameon=True, framealpha=0.9, loc='upper right')  # 添加图例
plt.tight_layout()  # 优化布局

# 显示图表
plt.show()

# Close the TensorBoard writer
writer.close()