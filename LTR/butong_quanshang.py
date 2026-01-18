import pandas as pd
import numpy as np

# 读取Excel文件
df = pd.read_excel('data/3068report_broker_merged.xlsx')

# 过滤日期范围
df = df[(df['qid_date'] >= 20171206) & (df['qid_date'] <= 20230303)]

# 按券商和日期分组，计算每日平均收益
df_grouped = df.groupby(['institution', 'qid_date']).agg({
    'real_return': 'mean'
}).reset_index()

# 初始化每个券商的累计收益
df_grouped['cumulative_return'] = 1.0

# 计算每个券商的累计收益
for institution, group in df_grouped.groupby('institution'):
    sr = (((1+group['real_return'].mean())**242)-1-0.025)/(group['real_return'].std()*242**0.5)
    cumulative_return = 1.0
    for i in range(len(group)):
        cumulative_return *= (1 + group.iloc[i]['real_return'])
        df_grouped.loc[group.index[i], 'cumulative_return'] = cumulative_return

# 提取每个券商的最终累计收益
final_cumulative_returns = df_grouped.groupby('institution')['cumulative_return'].last().reset_index()
final_cumulative_returns.columns = ['institution', 'final_cumulative_return']

# 计算每日收益的方差
daily_return_sr = (((1+df_grouped.groupby('institution')['real_return'].mean())**242)-1-0.025)/(df_grouped.groupby('institution')['real_return'].std()*242**0.5)
sr = daily_return_sr.to_frame()
sr = sr.reset_index(names='institution')
sr.columns = ['institution', 'SR']

# 计算累计收益的最大回撤
def max_drawdown(cumulative_returns):
    peak = cumulative_returns.cummax()  # 计算到当前日期的历史最大值
    drawdown = (peak - cumulative_returns) / peak  # 计算回撤率
    return drawdown.max()  # 返回最大回撤率

max_drawdowns = df_grouped.groupby('institution')['cumulative_return'].apply(max_drawdown).reset_index()
max_drawdowns.columns = ['institution', 'max_drawdown']

# 计算胜率
def win_rate(returns):
    return (returns > 0).mean()

win_rates = df.groupby('institution')['real_return'].apply(win_rate).reset_index()
win_rates.columns = ['institution', 'win_rate']

# 合并所有结果
result = pd.merge(final_cumulative_returns, sr, on='institution')
result = pd.merge(result, max_drawdowns, on='institution')
result = pd.merge(result, win_rates, on='institution')

# 保存结果到新的Excel文件
result.to_excel('end/institution/3068不同券商分析.xlsx', index=False)