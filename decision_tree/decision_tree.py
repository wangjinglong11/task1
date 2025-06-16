import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree
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

# Convert categorical features to one-hot encoding
X = pd.get_dummies(X)

# 5. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Base DecisionTreeRegressor model
print("\nTraining base DecisionTreeRegressor model...")
base_model = DecisionTreeRegressor(
    max_depth=4,
    random_state=42
)

base_model.fit(X_train_scaled, y_train)

# Evaluate base model
y_pred_base = base_model.predict(X_test_scaled)
mse_base = mean_squared_error(y_test, y_pred_base)
rmse_base = np.sqrt(mse_base)
r2_base = r2_score(y_test, y_pred_base)

print("\nBase model performance:")
print(f"MSE: {mse_base:.4f}")
print(f"RMSE: {rmse_base:.4f}")
print(f"R²: {r2_base:.4f}")

# 7. Visualize decision tree
plt.figure(figsize=(20, 10))
plot_tree(base_model, feature_names=X.columns, filled=True, max_depth=2, fontsize=10)
plt.title('Decision Tree Visualization (First 2 Levels)')
plt.savefig('Decision Tree Visualization (First 2 Levels).png', dpi=300, bbox_inches='tight')
plt.show()

# Use all features for retraining
X_train_sel = X_train_scaled
X_test_sel = X_test_scaled

# 8. Optimized DecisionTreeRegressor model
optimized_model = DecisionTreeRegressor(
    max_depth=8,
    min_samples_split=5,
    random_state=42
)

print("\nTraining optimized DecisionTreeRegressor model...")
optimized_model.fit(X_train_sel, y_train)

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
plt.savefig('Actual vs Predicted Values.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. Compare and save models
results = pd.DataFrame({
    'Model': ['Base DecisionTreeRegressor', 'Optimized DecisionTreeRegressor'],
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
}, 'optimized_decision_tree_model.pkl')

print("\nBest model saved!")