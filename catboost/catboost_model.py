import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

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

# 5. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create CatBoost Pool objects
train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

# 6. Base CatBoost model
print("\nTraining base CatBoost model...")
base_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=4,
    l2_leaf_reg=3,
    loss_function='RMSE',
    random_seed=42,
    verbose=100
)

base_model.fit(train_pool)

# Evaluate base model
y_pred_base = base_model.predict(test_pool)
mse_base = mean_squared_error(y_test, y_pred_base)
rmse_base = np.sqrt(mse_base)
r2_base = r2_score(y_test, y_pred_base)

print("\nBase model performance:")
print(f"MSE: {mse_base:.4f}")
print(f"RMSE: {rmse_base:.4f}")
print(f"R²: {r2_base:.4f}")

# 7. Feature importance analysis
feature_importances = base_model.get_feature_importance()
feature_names = X.columns

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nTop 15 Important Features:")
print(importance_df.head(15))

# Visualize feature importance
plt.figure(figsize=(12, 8))
plt.barh(importance_df['Feature'][:15], importance_df['Importance'][:15])
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Top 15 Important Features')
plt.tight_layout()
plt.savefig('Top 15 Important Features.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. Optimize model - based on feature importance
selected_features = importance_df[importance_df['Importance'] > importance_df['Importance'].mean()]['Feature'].values
print(f"\nSelected {len(selected_features)} important features")

# Retrain with selected features
X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]

# Update categorical feature indices
cat_features_sel = [i for i, col in enumerate(X_train_sel.columns) if X_train_sel[col].dtype == 'object']

train_pool_sel = Pool(X_train_sel, y_train, cat_features=cat_features_sel)
test_pool_sel = Pool(X_test_sel, y_test, cat_features=cat_features_sel)

# Optimized CatBoost model
optimized_model = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.03,
    depth=4,
    l2_leaf_reg=1,
    grow_policy='Lossguide',
    loss_function='RMSE',
    random_seed=42,
    verbose=100
)

print("\nTraining optimized CatBoost model...")
optimized_model.fit(train_pool_sel)

# Evaluate optimized model
y_pred_opt = optimized_model.predict(test_pool_sel)
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
plt.savefig('Actual vs Predicted Values.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. Compare and save models
results = pd.DataFrame({
    'Model': ['Base CatBoost', 'Optimized CatBoost'],
    'MSE': [mse_base, mse_opt],
    'RMSE': [rmse_base, rmse_opt],
    'R²': [r2_base, r2_opt]
})

print("\nModel performance comparison:")
print(results)

# Save best model
joblib.dump({
    'model': optimized_model,
    'features': selected_features,
    'cat_features': cat_features_sel
}, 'optimized_catboost_model.pkl')

print("\nBest model saved!")
