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


def sample_or_keep(group):
    if len(group) > 1:
        return group.sample(frac=chouyang_rate, random_state=random_zhongzi)  # 设置随机种子以确保结果可复现
    else:
        return group

col_name = ['stock_code', 'page', 'advance_reaction', 'star_analyst', 'title_len', 'num_sentence',
            'avg_sentence_len', 'sd_sentence_len',
            'num_authors', 'analyst_coverage', 'rm_rf', 'smb', 'hml', 'rmw', 'cma', 'broker_size', 'listed',
            'prior_performance_avg', 'prior_performance_sd', 'broker_status', 'qid_date', 'real_return', 'ind_1','ind_2','ind_3','ind_4','ind_5','ind_6',
            'close', 'pclose', 'volume']
Xcol_name = ['page', 'advance_reaction', 'star_analyst', 'title_len', 'num_sentence',
             'avg_sentence_len', 'sd_sentence_len',
             'num_authors', 'analyst_coverage', 'rm_rf', 'smb', 'hml', 'rmw', 'cma', 'broker_size', 'listed',
             'prior_performance_avg', 'prior_performance_sd', 'broker_status', 'qid_date', 'ind_1', 'ind_2', 'ind_3', 'ind_4', 'ind_5', 'ind_6']
Ycol_name = ['real_return']

chouyang_rate = 1.0
dapan_code = '3068'
metric = 'ndcg'
obj = 'rank:ndcg'
test_batch = 123
train_or_test = 'test'
m = 'ndcg'
train_year = 3
train_start = 20181206
train_end = 20211206
test_start = 20211207
test_end = 20230303

shouxufei = 0.0003
yinhaushui = 0.001



all_df = pd.read_csv(f'data/{dapan_code}merge_open_close_final.csv', usecols=col_name)
# all_df['qid_date'] = pd.to_numeric(all_df['qid_date'].str.replace('-', ''))
all_df = all_df[(train_start <= all_df['qid_date']) & (all_df['qid_date'] <= test_end)]
def mm():
    if chouyang_rate < 1.0:
        yuan_all_df = all_df.groupby('qid_date').apply(sample_or_keep).reset_index(drop=True)
    else:
        yuan_all_df = copy.deepcopy(all_df)

    def transform_column(column):
        return column.applymap(lambda x: x)
    yuan_all_df[Ycol_name] = transform_column(yuan_all_df[Ycol_name])
    #标准化
    features_to_normalize = yuan_all_df.drop(columns=['qid_date', 'stock_code'])
    min_vals = features_to_normalize.min()
    max_vals = features_to_normalize.max()
    normalized_features = 2 * (features_to_normalize - min_vals) / (max_vals - min_vals) - 1
    df_normalized = yuan_all_df[['qid_date']].join(normalized_features)

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
        tree_method='gpu_hist',
        lambdarank_num_pair_per_sample=8,
        booster='gbtree',
        eval_metric=metric,
        objective=obj,
        learning_rate=0.1,
        max_depth=6,
        n_estimators=1000,
        lambdarank_pair_method="topk"
    )

    x_train = X_train.drop(['qid_date'], axis=1)
    y_train = Y_train.to_numpy()
    # ranker = model.fit(x_train, y_train)
    ranker = model.fit(x_train, y_train, group=train_groups)
    # model_file = "model/0060lambdamart_model.dat"
    # ranker.save_model(model_file)

    # varimp = pd.DataFrame()
    # varimp["Features"] = X_train.drop(['qid_date'], axis=1).columns
    # varimp["VarImp"] = ranker.feature_importances_
    # # varimp.to_csv("lambdarank/feature_importance.csv")
    # print(varimp)

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
    temp = yuan_test_df[['qid_date', 'stock_code', 'real_return', 'close', 'pclose', 'volume']]

    temp.loc[:, "prediction"] = copy.deepcopy(predictions)
    # temp.to_csv(f'temp/oc/batch{test_batch}/{dapan_code}temp_{train_or_test}_{m}_train{train_year}.csv')
    # temp.to_csv(f'temp/oc/{dapan_code}temp_{train_or_test}_{m}_train{train_year}_{shouxufei}_{yinhaushui}_{learning_rate}_{max_depth}_{n_estimators}_{chouyang_rate}.csv')

    #
    if train_or_test == 'test':
        df = temp[(20211207 <= temp['qid_date']) & (temp['qid_date'] <= test_end)]
        df.set_index('qid_date', inplace=True)

        initial_capital = 1000000
        daily_results = []

        for qid_date, group in df.groupby('qid_date'):
            top_stocks = group.nlargest(min(4, len(group)), 'prediction')
            if len(top_stocks) == 0:
                print(f"Warning: No stocks available on {qid_date}. Skipping this date.")
                continue

            capital_per_stock = initial_capital / len(top_stocks)

            day_total_profit = 0

            for _, row in top_stocks.iterrows():
                # 计算能买的股数（向下取整）
                shou_num = int(capital_per_stock / (100 * row['pclose']))
                pfei = shou_num * 100 * row['pclose'] * shouxufei
                shares_bought = int((capital_per_stock - pfei) / (100 * row['pclose'])) * 100
                yu = capital_per_stock - pfei - shares_bought * row['pclose']

                if shares_bought == 0:
                    print(f"Warning: Not enough capital to buy any shares of stock {row['stock_code']} on {qid_date}.")
                    yu = capital_per_stock

                # 计算卖出后的资金
                sell_value = shares_bought * row['close'] - shares_bought * row['close'] * (shouxufei + yinhaushui) + yu

                # 累加当日总收益,在此其实是资金剩余总量
                day_total_profit += sell_value

            day_return = (day_total_profit - initial_capital) / initial_capital

            # 更新初始资金量为当日总收益
            initial_capital = day_total_profit

            # 记录每天的结果
            daily_results.append({
                'qid_date': qid_date,
                'total_profit': day_total_profit,
                'day_return': day_return
            })

        # 将结果转换为DataFrame
        results_df = pd.DataFrame(daily_results)

        # print(results_df)
        # results_df.to_csv(f'end/oc/batch{test_batch}/{dapan_code}return_{train_or_test}_{m}_train{train_year}_{shouxufei}_{yinhaushui}_{chouyang_rate}1111.csv', index=False)

        # 初始金额
        initial_amount = 5_000_000

        # 年化收益率 (ARR)
        trading_days = results_df.shape[0]
        ARR = (results_df.iloc[-1]['total_profit'] / initial_amount) ** (242 / trading_days) - 1

        # 最大回撤率
        results_df['cummax'] = results_df['total_profit'].cummax()
        results_df['drawdown'] = (results_df['total_profit'] - results_df['cummax']) / results_df['cummax']
        max_drawdown = results_df['drawdown'].min()

        # 卡尔玛比率
        calmar_ratio = ARR / abs(max_drawdown) if max_drawdown != 0 else np.nan

        # 夏普比率 (假设无风险利率为0.025)
        std_daily_return = results_df['day_return'].std()
        sharpe_ratio = (ARR - 0.025) / std_daily_return if std_daily_return != 0 else np.nan

        # 输出结果
        print(f"年化收益率 (ARR): {ARR:.3f}")
        print(f"最大回撤率: {max_drawdown:.3f}")
        print(f"卡尔玛比率: {calmar_ratio:.3f}")
        print(f"夏普比率: {sharpe_ratio:.3f}")
        dqn_ARR,dqn_MDR,dqn_CR,dqn_SR = dqn_chouyang(df, xuanze_df)
        return ARR,max_drawdown,calmar_ratio,sharpe_ratio, dqn_ARR,dqn_MDR,dqn_CR,dqn_SR

def dqn_chouyang(df, xuanze_df):
    initial_capital1 = 5000000
    daily_results1 = []
    for qid_date1, group1 in df.groupby('qid_date'):
        xxx = xuanze_df.loc[xuanze_df['qid_date'] == qid_date1, '3068']
        try:
            top_n = xxx.values[0]
        except:
            print(f'{qid_date1} error')
            continue
            # top_n = 0
        top_stocks1 = group1.nlargest(min(top_n, len(group1)), 'prediction')
        if len(top_stocks1) == 0:
            day_return1 = 0
            day_total_profit1 = initial_capital1

        else:

            capital_per_stock1 = initial_capital1 / len(top_stocks1)

            day_total_profit1 = 0

            for _, row1 in top_stocks1.iterrows():
                # 计算能买的股数（向下取整）
                shou_num1 = int(capital_per_stock1 / (100 * row1['pclose']))
                pfei1 = shou_num1 * 100 * row1['pclose'] * shouxufei
                shares_bought1 = int((capital_per_stock1 - pfei1) / (100 * row1['pclose'])) * 100
                yu1 = capital_per_stock1 - pfei1 - shares_bought1 * row1['pclose']

                if shares_bought1 == 0:
                    print(f"Warning: Not enough capital to buy any shares of stock {row1['stock_code']} on {qid_date1}.")
                    yu1 = capital_per_stock1

                # 计算卖出后的资金
                sell_value1 = shares_bought1 * row1['close'] - shares_bought1 * row1['close'] * (shouxufei + yinhaushui) + yu1

                # 累加当日总收益,在此其实是资金剩余总量
                day_total_profit1 += sell_value1

            day_return1 = (day_total_profit1 - initial_capital1) / initial_capital1

        # 更新初始资金量为当日总收益
        initial_capital1 = day_total_profit1

        # 记录每天的结果
        daily_results1.append({
            'qid_date': qid_date1,
            'total_profit': day_total_profit1,
            'day_return': day_return1
        })

    # 将结果转换为DataFrame
    results_df1 = pd.DataFrame(daily_results1)

    # print(results_df)
    # results_df1.to_csv(f'end/oc/batch{test_batch}/{dapan_code}return_{train_or_test}_{m}_train{train_year}_{shouxufei}_{yinhaushui}_{chouyang_rate}dqn.csv', index=False)

    # 初始金额
    initial_amount1 = 5_000_000

    # 年化收益率 (ARR)
    trading_days1 = results_df1.shape[0]
    ARR = (results_df1.iloc[-1]['total_profit'] / initial_amount1) ** (242 / trading_days1) - 1

    # 最大回撤率
    results_df1['cummax'] = results_df1['total_profit'].cummax()
    results_df1['drawdown'] = (results_df1['total_profit'] - results_df1['cummax']) / results_df1['cummax']
    max_drawdown = results_df1['drawdown'].min()

    # 卡尔玛比率
    calmar_ratio = ARR / abs(max_drawdown) if max_drawdown != 0 else np.nan

    # 夏普比率 (假设无风险利率为0.025)
    std_daily_return = results_df1['day_return'].std()
    sharpe_ratio = (ARR - 0.025) / std_daily_return if std_daily_return != 0 else np.nan

    # 输出结果
    print(f"年化收益率 (ARR): {ARR:.3f}")
    print(f"最大回撤率: {max_drawdown:.3f}")
    print(f"卡尔玛比率: {calmar_ratio:.3f}")
    print(f"夏普比率: {sharpe_ratio:.3f}")
    return ARR, max_drawdown, calmar_ratio, sharpe_ratio

xuanze_df = pd.read_csv('temp/meiri_xuanze.csv', usecols=['qid_date', '3068', '60'])
random_result = []
for random_zhongzi in range(1, 501):
    ARR,MDR,CR,SR,dqn_ARR,dqn_MDR,dqn_CR,dqn_SR = mm()
    random_result.append({
        'zhongzi': random_zhongzi,
        'ARR': ARR,
        'MDR': MDR,
        'CR': CR,
        'SR': SR,
        'dqn_ARR': dqn_ARR,
        'dqn_MDR': dqn_MDR,
        'dqn_CR': dqn_CR,
        'dqn_SR': dqn_SR
    })
random_df = pd.DataFrame(random_result)

print(random_df)
# random_df.to_csv(
#     f'end/oc/batch{test_batch}/{dapan_code}return_{train_or_test}_{m}_train{train_year}_{chouyang_rate}.csv',
#     index=False)
