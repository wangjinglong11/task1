import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
df = pd.read_csv('cancer_reg.csv')

print('数据基本信息：')
df.info()

# 查看数据集行数和列数
rows, columns = df.shape

if rows < 100 and columns < 20:
    # 短表数据（行数少于100且列数少于20）查看全量数据信息
    print('数据全部内容信息：')
    print(df.to_csv(sep='\t', na_rep='nan'))
else:
    # 长表数据查看数据前几行信息
    print('数据前几行内容信息：')
    print(df.head().to_csv(sep='\t', na_rep='nan'))

# 对 binnedInc 和 Geography 列进行独热编码
encoded_data = pd.get_dummies(df, columns=['binnedInc', 'Geography'])




# 绘制柱状图展示编码结果的分布
encoded_data.iloc[:, -len(encoded_data.columns)+len(df.columns):].sum().plot(kind='bar')
plt.title('独热编码结果分布')
plt.xlabel('编码特征')
plt.ylabel('数量')
plt.xticks(rotation=90)
plt.show()