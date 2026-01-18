import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np



# 将列表转换成pandas DataFrame
df = pd.read_csv('end/oc/batch123/chouyang_all_main.csv')
##'Main Board LambdaRank','Main Board LambdaMART','Main Board LTR-DQN','ChiNext LambdaRank','ChiNext LambdaMART','ChiNext LTR-DQN'
models = ['LambdaRank','LambdaMART','LTR-DQN']

model_colors = {
    'LambdaRank':'#00ccff',
    'LambdaMART':'#ff6600',
    'LTR-DQN':'#c0c0c0',
    # 'ChiNext LambdaRank':'#ffcc00',
    # 'ChiNext LambdaMART':'#993300',
    # 'ChiNext LTR-DQN':'#99cc00'
}

flierprops_dict = {model: dict(markerfacecolor=color, markeredgecolor=color, marker='o', alpha=0.7)
                   for model, color in model_colors.items()}

# 绘制箱型图
plt.figure(figsize=(12, 6))
plt.rcParams['font.size'] = 14  # 设置全局字体大小
plt.rcParams['font.weight'] = 'bold'  # 设置全局字体加粗
sns.boxplot(x='Sampling Rate', y='Value', hue='Model', data=df, palette=model_colors, hue_order=models)

ax = plt.gca()

# 遍历每个模型，设置对应的离群点颜色
for i, artist in enumerate(ax.artists):
    # 设置离群点的颜色与箱体颜色一致
    box = artist
    model_name = models[i % len(models)]
    color = model_colors[model_name]

    # 找到属于当前模型的所有离群点并设置其颜色
    fliers = ax.lines[2 * i + 2]  # 离群点在lines列表中的位置是固定的
    fliers.set_markerfacecolor(color)
    fliers.set_markeredgecolor(color)

# 添加标题和标签
plt.ylabel('Annualized Return')
plt.xlabel('Sampling Rate')
plt.legend(loc='upper right')
plt.ylabel('Annualized Return', fontsize=16, fontweight='bold')  # 设置y轴标签字体大小和加粗
plt.xlabel('Sampling Rate', fontsize=16, fontweight='bold')  # 设置x轴标签字体大小和加粗
plt.legend(loc='upper right', fontsize=12, title_fontsize='14')  # 设置图例字体大小和加粗

# 显示图形
plt.show()