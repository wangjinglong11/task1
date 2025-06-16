import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import joblib
from tensorboardX import SummaryWriter

# 1. Load and preprocess data
data = pd.read_csv('../cancer_reg.csv')

# Data preview
print("First 5 rows:")
print(data.head())
print("\nData description:")
print(data.describe())
print("\nMissing value check:")
print(data.isnull().sum())

# Fill missing values - median for numeric, mode for categorical
for col in data.columns:
    if data[col].dtype == 'object':
        data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        data[col].fillna(data[col].median(), inplace=True)

# 2. Feature engineering
def feature_engineering(df):
    df['Incidence_Death_Ratio'] = df['incidenceRate'] / (df['TARGET_deathRate'] + 1e-6)
    df['Income_Poverty_Ratio'] = df['medIncome'] / (df['povertyPercent'] + 1e-6)
    df['Employment_Stability'] = df['PctEmployed16_Over'] / (df['PctUnemployed16_Over'] + 1e-6)
    df['binnedInc_numeric'] = df['binnedInc'].str.extract(r'(\d+\.?\d*)').astype(float)
    df['State'] = df['Geography'].str.extract(r', (\w+)$')
    df = df.drop(['binnedInc', 'Geography'], axis=1)
    return df


data = feature_engineering(data)

# 3. Separate features and target
X = data.drop('TARGET_deathRate', axis=1)
y = data['TARGET_deathRate']

# 4. Identify categorical features
cat_features = [i for i, col in enumerate(X.columns) if X[col].dtype == 'object']

# Replace NaN in categorical features with string 'missing'
for i in cat_features:
    col = X.columns[i]
    X[col] = X[col].astype(str).fillna("missing")

print("\nCategorical feature indices:", cat_features)
print("Categorical feature names:", X.columns[cat_features].tolist())

# Convert categorical features to one-hot encoding
X = pd.get_dummies(X)

# 5. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Base MLP model
print("\nTraining base MLPRegressor model...")
base_model = MLPRegressor(
    hidden_layer_sizes=(100,),
    activation='relu',
    solver='adam',
    max_iter=150,
    random_state=42,
    verbose=True  # 开启详细输出以记录损失值
)

# 创建 TensorBoard 写入器
writer_base = SummaryWriter('logs/base_model')

base_model.fit(X_train_scaled, y_train)

# 记录训练损失到 TensorBoard
for step, loss in enumerate(base_model.loss_curve_):
    writer_base.add_scalar('training_loss', loss, step)

# 关闭 TensorBoard 写入器
writer_base.close()

# 绘制并保存基础模型的损失曲线
plt.figure(figsize=(10, 6))
plt.plot(base_model.loss_curve_)
plt.title('Base Model Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('base_model_loss_curve.png')  # 保存图像
plt.show()  # 显示图像

# Evaluate base model
y_pred_base = base_model.predict(X_test_scaled)
mse_base = mean_squared_error(y_test, y_pred_base)
rmse_base = np.sqrt(mse_base)
r2_base = r2_score(y_test, y_pred_base)

print("\nBase model performance:")
print(f"MSE: {mse_base:.4f}")
print(f"RMSE: {rmse_base:.4f}")
print(f"R²: {r2_base:.4f}")

# 7. Feature importance analysis is not available for MLPRegressor
print("\nFeature importance is not available for MLPRegressor.")

# Use all features for retraining as feature selection based on importance is not applicable
X_train_sel = X_train_scaled
X_test_sel = X_test_scaled

# 8. Optimized MLP model
optimized_model = MLPRegressor(
    hidden_layer_sizes=(200, 100),
    activation='relu',
    solver='adam',
    max_iter=150,
    alpha=0.0001,
    random_state=42,
    verbose=True  # 开启详细输出以记录损失值
)

# 创建 TensorBoard 写入器
writer_opt = SummaryWriter('logs/optimized_model')

print("\nTraining optimized MLPRegressor model...")
optimized_model.fit(X_train_sel, y_train)

# 记录训练损失到 TensorBoard
for step, loss in enumerate(optimized_model.loss_curve_):
    writer_opt.add_scalar('training_loss', loss, step)

# 关闭 TensorBoard 写入器
writer_opt.close()

# 绘制并保存优化模型的损失曲线
plt.figure(figsize=(10, 6))
plt.plot(optimized_model.loss_curve_)
plt.title('Optimized Model Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('optimized_model_loss_curve.png')  # 保存图像
plt.show()  # 显示图像

# Evaluate optimized model
y_pred_opt = optimized_model.predict(X_test_sel)
mse_opt = mean_squared_error(y_test, y_pred_opt)
rmse_opt = np.sqrt(mse_opt)
r2_opt = r2_score(y_test, y_pred_opt)

print("\nOptimized model performance:")
print(f"MSE: {mse_opt:.4f}")
print(f"RMSE: {rmse_opt:.4f}")
print(f"R²: {r2_opt:.4f}")

# 9. Visualize predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_opt, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
plt.grid(True)
plt.savefig('actual_vs_predicted.png')  # 保存预测对比图
plt.show()

# 10. Compare and save models
results = pd.DataFrame({
    'Model': ['Base MLPRegressor', 'Optimized MLPRegressor'],
    'MSE': [mse_base, mse_opt],
    'RMSE': [rmse_base, rmse_opt],
    'R²': [r2_base, r2_opt]
})

print("\nModel performance comparison:")
print(results)

# Save best model
joblib.dump({
    'model': optimized_model,
    'scaler': scaler,
    'features': X.columns
}, 'optimized_mlp_model.pkl')

print("\nBest model saved!")