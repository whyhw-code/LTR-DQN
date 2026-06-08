import argparse
import copy
import logging

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from dl_dqn2 import T4ExcelWriter

logging.basicConfig(level=logging.INFO)


col_name = [
    'stock_code', 'page', 'advance_reaction', 'star_analyst', 'title_len', 'num_sentence',
    'avg_sentence_len', 'sd_sentence_len',
    'num_authors', 'analyst_coverage', 'rm_rf', 'smb', 'hml', 'rmw', 'cma', 'broker_size', 'listed',
    'prior_performance_avg', 'prior_performance_sd', 'broker_status', 'qid_date', 'real_return',
    'ind_1', 'ind_2', 'ind_3', 'ind_4', 'ind_5', 'ind_6',
    'close', 'pclose'
]

Xcol_name = [
    'page', 'advance_reaction', 'star_analyst', 'title_len', 'num_sentence',
    'avg_sentence_len', 'sd_sentence_len',
    'num_authors', 'analyst_coverage', 'rm_rf', 'smb', 'hml', 'rmw', 'cma', 'broker_size', 'listed',
    'prior_performance_avg', 'prior_performance_sd', 'broker_status', 'qid_date',
    'ind_1', 'ind_2', 'ind_3', 'ind_4', 'ind_5', 'ind_6'
]

Ycol_name = ['real_return']


def get_train_range(train_year):
    if train_year == 2:
        return 20191206, 20211206
    elif train_year == 3:
        return 20181206, 20211206
    elif train_year == 4:
        return 20171206, 20211206
    else:
        raise ValueError("train_year must be 2, 3, or 4.")


def transform_column(column):
    return column.applymap(lambda x: x)


def run_backtest(temp, test_end, shouxufei, yinhaushui):
    df = temp[(20211207 <= temp['qid_date']) & (temp['qid_date'] <= test_end)].copy()
    df.set_index('qid_date', inplace=True)

    initial_capital = 5_000_000
    daily_results = []
    shenglv_fenmu = 0
    shenglv_fenzi = 0

    for qid_date, group in df.groupby('qid_date'):
        top_stocks = group.nlargest(min(4, len(group)), 'prediction')
        if len(top_stocks) == 0:
            continue

        capital_per_stock = initial_capital / len(top_stocks)
        shenglv_fenmu += len(top_stocks)

        day_total_profit = 0

        for _, row in top_stocks.iterrows():
            # 计算能买的股数（向下取整）
            shou_num = int(capital_per_stock / (100 * row['pclose']))
            pfei = shou_num * 100 * row['pclose'] * shouxufei
            shares_bought = int((capital_per_stock - pfei) / (100 * row['pclose'])) * 100
            yu = capital_per_stock - pfei - shares_bought * row['pclose']

            if shares_bought == 0:
                yu = capital_per_stock

            # 计算卖出后的资金
            sell_value = (
                shares_bought * row['close']
                - shares_bought * row['close'] * (shouxufei + yinhaushui)
                + yu
            )

            if sell_value > capital_per_stock:
                shenglv_fenzi += 1

            # 累加当日总收益，在此其实是资金剩余总量
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

    results_df = pd.DataFrame(daily_results)

    if len(results_df) == 0:
        raise ValueError("results_df is empty. Please check test_start/test_end or input data.")

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

    # 夏普比率（假设无风险利率为 0.025）
    std_daily_return = results_df['day_return'].std()
    sharpe_ratio = (
        (((1 + results_df['day_return'].mean()) ** 242) - 1 - 0.025)
        / (std_daily_return * 242 ** 0.5)
        if std_daily_return != 0 else np.nan
    )

    shenglv = shenglv_fenzi / shenglv_fenmu if shenglv_fenmu != 0 else np.nan

    return results_df, ARR, max_drawdown, calmar_ratio, sharpe_ratio, shenglv


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_or_test', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--train_year', type=int, default=3)
    parser.add_argument('--test_batch', type=int, default=123)

    parser.add_argument('--shouxufei', type=float, default=0.0003)
    parser.add_argument('--yinhaushui', type=float, default=0.001)

    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--max_depth', type=int, default=4)
    parser.add_argument('--n_estimators', type=int, default=1100)

    # 默认不写入 Excel，避免 F4 敏感性分析时反复覆盖 Fig.5/F4 汇总表
    parser.add_argument('--write_excel', action='store_true')

    # 默认不保存 temp 预测文件；需要复查预测结果时再手动开启
    parser.add_argument('--save_temp', action='store_true')

    # 默认不打印每日资金表
    parser.add_argument('--print_df', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dapan_code = '3068'
    metric = 'ndcg'
    obj = 'rank:ndcg'
    test_batch = args.test_batch
    train_or_test = args.train_or_test
    m = 'ndcg'
    train_year = args.train_year
    train_start, train_end = get_train_range(train_year)
    test_start = 20211207
    test_end = 20230303

    shouxufei = args.shouxufei
    yinhaushui = args.yinhaushui
    learning_rate = args.learning_rate
    max_depth = args.max_depth
    n_estimators = args.n_estimators

    all_df = pd.read_csv(
        f'data/{dapan_code}merge_open_close_final.csv',
        usecols=col_name
    )
    # all_df['qid_date'] = pd.to_numeric(all_df['qid_date'].str.replace('-', ''))
    all_df = all_df[(train_start <= all_df['qid_date']) & (all_df['qid_date'] <= test_end)]

    yuan_all_df = copy.deepcopy(all_df)
    yuan_all_df[Ycol_name] = transform_column(yuan_all_df[Ycol_name])

    # 标准化
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

    train_groups = (
        train_df.groupby('qid_date')
        .size()
        .to_frame('size')
        .sort_values(by='qid_date')['size']
        .to_numpy()
        .tolist()
    )

    test_groups = (
        test_df.groupby('qid_date')
        .size()
        .to_frame('size')
        .sort_values(by='qid_date')['size']
        .to_numpy()
        .tolist()
    )

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
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        lambdarank_pair_method='topk'
    )

    x_train = X_train.drop(['qid_date'], axis=1)
    y_train = Y_train.to_numpy()
    ranker = model.fit(x_train, y_train, group=train_groups)

    # 重要性
    # varimp = pd.DataFrame()
    # varimp["Features"] = X_train.drop(['qid_date'], axis=1).columns
    # varimp["VarImp"] = ranker.feature_importances_
    # varimp["LTR-DQN"] = (varimp["VarImp"] - varimp["VarImp"].min()) / (varimp["VarImp"].max() - varimp["VarImp"].min())
    # varimp.to_csv(f"../result/batch{test_batch}/feature_importance_LTR-DQN_{dapan_code}.csv")

    x_test = X_test.drop(['qid_date'], axis=1)
    predictions = []
    start_idx = 0

    for group_size in test_groups:
        # 提取当前分组的数据
        group_data = x_test.iloc[start_idx:start_idx + group_size]
        # 预测当前分组的结果
        group_predictions = ranker.predict(group_data)
        predictions.extend(group_predictions)
        # 更新索引以处理下一个分组
        start_idx += group_size

    temp = yuan_test_df[['qid_date', 'stock_code', 'real_return', 'close', 'pclose']].copy()
    temp.loc[:, 'prediction'] = copy.deepcopy(predictions)

    if args.save_temp:
        temp_path = (
            f'temp/oc/batch{test_batch}/'
            f'{dapan_code}temp_{train_or_test}_{m}_train{train_year}_'
            f'{shouxufei}_{yinhaushui}_{learning_rate}_{max_depth}_{n_estimators}.csv'
        )
        temp.to_csv(temp_path, index=False)

    if train_or_test == 'test':
        results_df, ARR, max_drawdown, calmar_ratio, sharpe_ratio, shenglv = run_backtest(
            temp=temp,
            test_end=test_end,
            shouxufei=shouxufei,
            yinhaushui=yinhaushui
        )

        if args.write_excel:
            T4ExcelWriter(scale=5).write(
                results_df,
                dapan_code,
                'LambdaMART'
            )

        if args.print_df:
            print(results_df)

        # 输出结果
        print(f"年化收益率 (ARR): {ARR:.8f}")
        print(f"最大回撤率: {-max_drawdown:.3f}")
        print(f"卡尔玛比率: {calmar_ratio:.3f}")
        print(f"夏普比率: {sharpe_ratio:.3f}")
        print(f"WR: {shenglv:.3f}")
