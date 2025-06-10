# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
# import numpy as np
#
# # Load the dataset
# df = pd.read_csv('../new_cancer_reg.csv')
#
# # Extract features and target variable
# X = df.drop('TARGET_deathRate', axis=1)
# y = df['TARGET_deathRate']
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Initialize the linear regression model
# model = LinearRegression()
#
# # Fit the model
# model.fit(X_train, y_train)
#
# # Calculate the weights and intercept
# weights = model.coef_
# intercept = model.intercept_
#
# # Make predictions on the test set
# y_pred = model.predict(X_test)
#
# # Calculate evaluation metrics
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)
#
# # Print the results
# print('Weights:', weights)
# print('Intercept:', intercept)
# print('Mean Squared Error:', mse)
# print('Root Mean Squared Error:', rmse)
# print('R-squared:', r2)

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set English font support
plt.rcParams["font.family"] = ["Arial", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # Solve negative sign display issue

# Load the dataset
df = pd.read_csv('../new_cancer_reg.csv')

# Extract features and target variable
X = df.drop('TARGET_deathRate', axis=1)
y = df['TARGET_deathRate']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Calculate the weights and intercept
weights = model.coef_
intercept = model.intercept_

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Create a DataFrame of features and coefficients for sorting and visualization
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient Value': weights
}).sort_values('Coefficient Value', ascending=False)  # Sort by coefficient absolute value

# Print the results
print('Model Weights (Coefficients):')
print(coefficients)
print(f'Intercept: {intercept:.4f}')
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'Coefficient of Determination (R²): {r2:.4f}')

# Generate linear regression coefficients bar chart
plt.figure(figsize=(12, 8))  # Set chart size
sns.set_style("whitegrid")  # Set chart style

# Draw bar chart, positive coefficients in blue, negative coefficients in green
colors = ['#00A1FF' if w >= 0 else '#5ed935' for w in coefficients['Coefficient Value']]
bars = sns.barplot(
    x='Coefficient Value',
    y='Feature',
    data=coefficients,
    palette=colors,
    zorder=3  # Adjust layer order to avoid grid lines covering bars
)

# Add title and labels
plt.title('Linear Regression Model Coefficients Visualization', fontsize=16)
plt.xlabel('Coefficient Value', fontsize=12)
plt.ylabel('Feature', fontsize=12)

# Add grid lines (horizontal only)
plt.grid(axis='x', linestyle='-', zorder=0)

# Optimize chart display
plt.tight_layout()  # Automatically adjust layout
plt.savefig('linear_regression_coefficients.png', dpi=300, bbox_inches='tight')  # Save the chart
plt.show()