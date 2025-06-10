import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('../new_cancer_reg.csv')

# Extract features and target variable
X = df.drop('TARGET_deathRate', axis=1)
y = df['TARGET_deathRate']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the CatBoost regressor
cb = CatBoostRegressor(random_state=42)

# Fit the model
cb.fit(X_train, y_train, verbose=False)

# Make predictions on the test set
y_pred = cb.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Get feature importance
feature_importance = cb.get_feature_importance()
feature_names = X.columns

# Print evaluation metrics
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)


# Plot feature importance bar chart
plt.figure(figsize=(20, 8))  # Increase figure width to fit all labels
bars = plt.bar(feature_names, feature_importance)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('CATBoost Feature Importance')
plt.xticks(rotation=90, ha='right', fontsize=6)  # Rotate labels and adjust horizontal alignment
plt.tight_layout()  # Automatically adjust subplot parameters to give specified padding

# 保存图片到当前路径
plt.savefig('catboost_feature.png', dpi=300, bbox_inches='tight')

plt.show()