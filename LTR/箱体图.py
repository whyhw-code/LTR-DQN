import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 读取Excel文件
df = pd.read_csv('end/institution/all.csv')  # 读取前两行作为多级列名

# 将数据转换为长格式
df_long = df.melt(id_vars=['group'], var_name='Indicator', value_name='Value')

# 自定义颜色和样式
custom_palette = {'Main Board': 'white', 'ChiNext': 'black'}

# 绘制箱形图
plt.figure(figsize=(12, 6))
ax = sns.boxplot(x='Indicator', y='Value', hue='group', data=df_long, palette='Set3')



# 去掉图例中的`group`词
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, ['Main board market', 'ChiNext market'], title='')

# 添加标题和标签
# plt.title('Boxplot of Indicators by Group', fontsize=16)
plt.ylabel('Value', fontsize=14)
plt.xlabel('Evaluation metrics', fontsize=14)

# 显示图形
plt.show()