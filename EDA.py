
import pandas as pd

# 提出缺失值
df = pd.read_csv('cancer_reg.csv')
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_data = pd.concat([missing_values, missing_percentage], axis=1, keys=['Number of Missing Values', 'Missing Value Percentage (%)'])
print('Missing Data Information:')
print(missing_data[missing_data['Number of Missing Values'] > 0])

#确定特征的影响
import seaborn as sns
import matplotlib.pyplot as plt

# Encode object type data using one-hot encoding and drop rows with missing values
df_encoded = pd.get_dummies(df.dropna(), columns=['binnedInc', 'Geography'])

# Calculate the correlation with TARGET_deathRate (round to two decimal places)
correlation = df_encoded.corr()['TARGET_deathRate'].round(2).sort_values(ascending=False)

print('Correlation of each feature with TARGET_deathRate:')
print(correlation)

# Set figure resolution
plt.rcParams['figure.dpi'] = 600

# Plot bar chart
plt.figure(figsize=(10, 8))
correlation.plot(kind='bar')
plt.title('Correlation of each feature with TARGET_deathRate')
plt.xlabel('Feature')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=90)
#plt.show()

#选择最合适的特征
selected_features = correlation[abs(correlation) > 0.3].index.tolist()
print('select features：', selected_features)
#生成新的数据集
new_df = df_encoded[selected_features]
new_df.to_csv('new_cancer_reg.csv', index=False)