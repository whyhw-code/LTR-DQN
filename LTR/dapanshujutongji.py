import torch
import pandas as pd
import numpy as np
import copy
# 绘图相关
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import ndcg_score
from xgboost import plot_importance
import xgboost as xgb
import lightgbm as lgb

from sklearn import metrics
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit

import logging
logging.basicConfig(level=logging.INFO)

# 检查是否有可用的 GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("device:cuda")
else:
    device = torch.device('cpu')
    print("device:cpu")


col_name = ['trade_date', 'open', 'high', 'low', 'close', 'vol',
            'amount', 'pct_chg','group_len']


dapan_code = '0060'
train_start = 20171206
test_end = 20230303

df = pd.read_csv(f'data/{dapan_code}merge.csv', usecols=col_name)

# 创建一个空的DataFrame来存储统计信息
stats_df = pd.DataFrame()

# 遍历每一列并计算所需统计量
for column in df.columns:
    if pd.api.types.is_numeric_dtype(df[column]):  # 确保只处理数值型列
        stats = {
            '最大值': df[column].max(),
            '最小值': df[column].min(),
            '中位数': df[column].median(),
            '平均数': df[column].mean(),
            '标准差': df[column].std(),
            '偏度': df[column].skew(),
            '样本数量': df[column].count()
        }
        # 将统计信息添加到DataFrame中
        stats_df[column] = pd.Series(stats)

# 转置DataFrame以使列为统计量，行为原始列名
stats_df = stats_df.T

# 将结果保存到Excel文件
output_path = '0060_dapan_statistics_results.xlsx'
stats_df.to_excel(output_path, sheet_name='Statistics')

print(f"统计结果已保存到 {output_path}")
