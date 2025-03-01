import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('new_cancer_reg.csv')

# Extract features and target variable
X = df.drop('TARGET_deathRate', axis=1)
y = df['TARGET_deathRate']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost regressor
xgb = XGBRegressor(random_state=42)

# Fit the model
xgb.fit(X_train, y_train)

# Make predictions on the test set
y_pred = xgb.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Get feature importance
feature_importance = xgb.get_booster().get_score(importance_type='weight')

# Print evaluation metrics
print('Mean Squared Error:', mse)
print('Root Mean Squared Error:', rmse)
print('R-squared:', r2)

# Set figure resolution
plt.rcParams['figure.dpi'] = 300

# Set font to display non-English characters (optional)
# plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']

# Plot feature importance bar chart
plt.figure(figsize=(30, 6))
plt.bar(feature_importance.keys(), feature_importance.values())
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('XGBoost Feature Importance')
plt.xticks(rotation=90)
plt.show()