import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 加载数据集
df = pd.read_csv('new_cancer_reg.csv')

# 提取特征和目标变量
X = df.drop('TARGET_deathRate', axis=1)
y = df['TARGET_deathRate']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化决策树回归器
dt = DecisionTreeRegressor(random_state=42)

# 拟合模型
dt.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = dt.predict(X_test)

# 计算评估指标
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# 输出结果
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)