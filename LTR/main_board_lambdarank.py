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


def topk_max(g):
    top_k_results = g.apply(top_k_cumulative_max)
    topk_df = pd.DataFrame(top_k_results.values.tolist(), columns=['qid_date', 'top_k_count', 'ret1', 'group_len'])
    return topk_df

# 定义函数计算累积和最大的前几个值
def top_k_cumulative_max(group):
    k = len(group)
    max_sum = 0  # 初始化最大累积和
    current_sum = 0  # 初始化当前累积和
    top_k = 0  # 初始化累积和最大的前几个值的数量
    # 按照 prediction 的大小从大到小排序
    sorted_group = group.sort_values(by='prediction', ascending=False)
    sorted_group.reset_index(drop=True, inplace=True)
    # 计算累积和并更新最大累积和的前几个值的数量
    for idx, row in sorted_group.iterrows():
        current_sum += row['real_return'] - 0.002
        if current_sum > max_sum:
            max_sum = current_sum
            top_k = idx+1
    return row['qid_date'], top_k, len(group[group['real_return']-0.02 > 0]), k

col_name = ['stock_code', 'page', 'advance_reaction', 'star_analyst', 'title_len', 'num_sentence',
            'avg_sentence_len', 'sd_sentence_len',
            'num_authors', 'analyst_coverage', 'rm_rf', 'smb', 'hml', 'rmw', 'cma', 'broker_size', 'listed',
            'prior_performance_avg', 'prior_performance_sd', 'broker_status', 'qid_date', 'real_return', 'ind_1','ind_2','ind_3','ind_4','ind_5','ind_6']
Xcol_name = ['page', 'advance_reaction', 'star_analyst', 'title_len', 'num_sentence',
             'avg_sentence_len', 'sd_sentence_len',
             'num_authors', 'analyst_coverage', 'rm_rf', 'smb', 'hml', 'rmw', 'cma', 'broker_size', 'listed',
             'prior_performance_avg', 'prior_performance_sd', 'broker_status', 'qid_date', 'ind_1', 'ind_2', 'ind_3', 'ind_4', 'ind_5', 'ind_6']
Ycol_name = ['real_return']

dapan_code = '0060'
metric = 'ndcg'
obj = 'rank:pairwise'
test_batch = 3
train_or_test = 'test'
m = 'pairwise11'
train_year = 2
train_start = 20200929
train_end = 20220928
test_start = 20220929
test_end = 20230303

all_df = pd.read_csv(f'data/{dapan_code}report.csv', usecols=col_name)
all_df['qid_date'] = pd.to_numeric(all_df['qid_date'].str.replace('-', ''))
all_df = all_df[(train_start <= all_df['qid_date']) & (all_df['qid_date'] <= test_end)]
yuan_all_df = copy.deepcopy(all_df)
def transform_column(column):
    return column.applymap(lambda x: x)
all_df[Ycol_name] = transform_column(all_df[Ycol_name])
#标准化
features_to_normalize = all_df.drop(columns=['qid_date', 'stock_code'])
min_vals = features_to_normalize.min()
max_vals = features_to_normalize.max()
normalized_features = 2 * (features_to_normalize - min_vals) / (max_vals - min_vals) - 1
df_normalized = all_df[['qid_date']].join(normalized_features)

train_df = df_normalized[(train_start <= df_normalized['qid_date']) & (df_normalized['qid_date'] <= train_end)]
yuan_train_df = yuan_all_df[(train_start <= yuan_all_df['qid_date']) & (yuan_all_df['qid_date'] <= train_end)]
if train_or_test == 'test':
    test_df = df_normalized[(test_start <= df_normalized['qid_date']) & (df_normalized['qid_date'] <= test_end)]
    yuan_test_df = yuan_all_df[(test_start <= yuan_all_df['qid_date']) & (yuan_all_df['qid_date'] <= test_end)]
else:
    test_df = copy.deepcopy(train_df)
    yuan_test_df = copy.deepcopy(yuan_train_df)

train_groups = train_df.groupby('qid_date').size().to_frame('size').sort_values(by='qid_date')['size'].to_numpy().tolist()
# test_df['qid'] = pd.to_datetime(test_df['qid_date'])
test_groups = test_df.groupby('qid_date').size().to_frame('size').sort_values(by='qid_date')['size'].to_numpy().tolist()

X_train = train_df[Xcol_name]
Y_train = train_df[Ycol_name]
X_test = test_df[Xcol_name]
Y_test = test_df[Ycol_name]

model = xgb.XGBRanker(
    lambdarank_num_pair_per_sample=8,
    eval_metric=metric,
    objective=obj,
    learning_rate=0.1,
    lambdarank_pair_method="topk"
)

x_train = X_train.drop(['qid_date'], axis=1)
y_train = Y_train.to_numpy()
# ranker = model.fit(x_train, y_train)
ranker = model.fit(x_train, y_train, group=train_groups)
# model_file = "model/0060lambdamart_model.dat"
# ranker.save_model(model_file)

feature_importance = ranker.feature_importances_
varimp = pd.DataFrame()
varimp["Features"] = X_train.drop(['qid_date'], axis=1).columns
varimp["VarImp"] = ranker.feature_importances_
# varimp.to_csv("lambdarank/feature_importance.csv")
print(varimp)

x_test = X_test.drop(['qid_date'], axis=1)
predictions = []
ndcg_scores = []
start_idx = 0
for group_size in test_groups:
    # 提取当前分组的数据
    group_data = x_test.iloc[start_idx:start_idx+group_size]
    group_y = Y_test.iloc[start_idx:start_idx+group_size]
    # 预测当前分组的结果
    group_predictions = ranker.predict(group_data)
    # # 对预测结果进行排序并计算 NDCG
    # sorted_indices = np.argsort(group_predictions)[::-1]  # 对预测结果降序排列的索引
    # sorted_labels = Y_test['real_return'].iloc[sorted_indices].to_numpy()  # 对应的标签按照预测结果排序
    # y = group_y.to_numpy().ravel()
    # ndcg = ndcg_score([y], [group_predictions], k=5)  # 计算 NDCG 值
    # 将 NDCG 值保存到总的 NDCG 列表中
    predictions.extend(group_predictions)
    # ndcg_scores.append(ndcg)
    # 更新索引以处理下一个分组
    start_idx += group_size


# results = pd.DataFrame(columns=['qid_date', 'real_return', 'prediction'])
# temp = pd.concat([X_test, yuan_test_df[Ycol_name]], axis=1)[['qid_date', 'real_return']]
temp = yuan_test_df[['qid_date', 'stock_code', 'real_return']]

temp.loc[:, "prediction"] = copy.deepcopy(predictions)
temp.to_csv(f'temp/batch{test_batch}/{dapan_code}temp_{train_or_test}_{m}_train{train_year}.csv')

#
(* if train_or_test == 'test':
    df = temp
    # df['qid_date'] = pd.to_datetime(df['qid_date'])
    #定义prediction大于的阈值
    threshold = 0
    df.loc[df['prediction'] > threshold, 'real_return'] -= 0.0016

    # 根据日期分组
    grouped = df.groupby('qid_date')
    # 统计每个分组的大小
    group_sizes = grouped.size().to_frame('size')
    # 统计每个分组中 real_return 列大于 0 的数量
    group_positive_count = grouped.apply(lambda x: (x['real_return'] > 0).sum()).to_frame('group_positive_count')
    # 计算每组中 prediction 大于阈值的 real_return 的平均值
    grouped_mean = df[df['prediction'] > threshold].groupby('qid_date')['real_return'].mean().to_frame('mean_real_return')
    # 统计每个分组中 prediction 大于阈值的数据数量
    group_selected_size = df[df['prediction'] > threshold].groupby('qid_date').size().to_frame('selected_size')
    # 统计每个分组中 prediction 大于阈值的数据中 real_return 大于 0 的数量
    group_selected_positive_count = df[df['prediction'] > threshold].groupby('qid_date').apply(lambda x: (x['real_return'] > 0).sum()).to_frame('selected_positive_count')
    # 合并统计结果
    result = group_sizes.join(group_positive_count).join(grouped_mean).join(group_selected_size).join(group_selected_positive_count)
    result = result.reset_index('qid_date')
    result.to_csv(f'end//batch{test_batch}/c{dapan_code}return_{train_or_test}_{m}_train{train_year}.csv', index=False) *)
