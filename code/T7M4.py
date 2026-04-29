import pandas as pd
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)


shouxufei = 0.0003
yinhaushui = 0.001


dapan_code = '0060'
m = 'ndcg'
all_df = pd.read_csv(f'data/ESG/{dapan_code}temp_test_{m}_train3_esg.csv')
esg = 5.52
def mm():
    df = all_df
    df.set_index('qid_date', inplace=True)

    dqn_ARR,dqn_MDR,dqn_CR,dqn_SR,dqn_WR = dqn_chouyang(df, xuanze_df)
    return dqn_ARR,dqn_MDR,dqn_CR,dqn_SR,dqn_WR

def dqn_chouyang(df, xuanze_df):
    initial_capital1 = 1000000
    daily_results1 = []
    shenglv_fenmu1 = 0
    shenglv_fenzi1 = 0
    for qid_date1, group1 in df.groupby('qid_date'):
        xxx = xuanze_df.loc[xuanze_df['qid_date'] == qid_date1, '60']
        try:
            top_n = xxx.values[0]
        except:
            print(f'{qid_date1} error')
            continue
            # top_n = 0
        top_stocks1 = group1.nlargest(min(top_n, len(group1)), 'prediction')
        top_stocks1 = top_stocks1[top_stocks1['ESG'] >= esg]  #NS
        if len(top_stocks1) == 0:
            day_return1 = 0
            day_total_profit1 = initial_capital1

        else:

            capital_per_stock1 = initial_capital1 / len(top_stocks1)
            shenglv_fenmu1 = shenglv_fenmu1 + len(top_stocks1)

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
                if sell_value1 > capital_per_stock1:
                    shenglv_fenzi1 = shenglv_fenzi1 + 1

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
    # results_df1.to_csv(f'end/oc/batch123/{dapan_code}return_dqn{esg}PI.csv', index=False)

    # 初始金额
    initial_amount1 = 1_000_000

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
    sharpe_ratio = (((1 + results_df1['day_return'].mean()) ** 242) - 1 - 0.025) / (
                std_daily_return * 242 ** 0.5) if std_daily_return != 0 else np.nan


    shenglv1 = shenglv_fenzi1 / shenglv_fenmu1

    # 输出结果
    print(f"年化收益率 (ARR): {ARR:.3f}")
    print(f"最大回撤率: {max_drawdown:.3f}")
    print(f"卡尔玛比率: {calmar_ratio:.3f}")
    print(f"夏普比率: {sharpe_ratio:.3f}")
    print(f"DQN  WR: {shenglv1:.3f}")
    return ARR, max_drawdown, calmar_ratio, sharpe_ratio,shenglv1

xuanze_df = pd.read_csv('temp/oc/batch123/meiri_xuanze.csv', usecols=['qid_date', '3068', '60'])
random_result = []

dqn_ARR,dqn_MDR,dqn_CR,dqn_SR,dqn_WR = mm()
random_result.append({
        'dqn_ARR': dqn_ARR,
        'dqn_MDR': dqn_MDR,
        'dqn_CR': dqn_CR,
        'dqn_SR': dqn_SR,
        'dqn_WR': dqn_WR
    })
random_df = pd.DataFrame(random_result)

print(random_df)
