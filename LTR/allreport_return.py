import numpy as np
import pandas as pd

col_name = ['stock_code',  'qid_date', 'real_return',
            'close', 'pclose']

dapan_code = '0060'

shouxufei = 0.0003
yinhaushui = 0.001

all_df = pd.read_csv(f'data/{dapan_code}merge_open_close_final.csv', usecols=col_name)


df = all_df[(20171206 <= all_df['qid_date']) & (all_df['qid_date'] <= 20230303)]
df.set_index('qid_date', inplace=True)

initial_capital = 1000000
daily_results = []
shenglv_fenmu = 0
shenglv_fenzi = 0

for qid_date, group in df.groupby('qid_date'):
    top_stocks = group
    if len(top_stocks) == 0:
        print(f"Warning: No stocks available on {qid_date}. Skipping this date.")
        continue

    capital_per_stock = initial_capital / len(top_stocks)
    shenglv_fenmu = shenglv_fenmu + len(top_stocks)

    day_total_profit = 0

    for _, row in top_stocks.iterrows():
        # 计算能买的股数（向下取整）
        shou_num = int(capital_per_stock / (100 * row['pclose']))
        pfei = shou_num * 100 * row['pclose'] * shouxufei
        shares_bought = int((capital_per_stock - pfei) / (100 * row['pclose'])) * 100
        yu = capital_per_stock - pfei - shares_bought * row['pclose']

        if shares_bought == 0:
            print(f"Warning: Not enough capital to buy any shares of stock {row['stock_code']} on {qid_date}.")

        # 计算卖出后的资金
        sell_value = shares_bought * row['close'] - shares_bought * row['close'] * (shouxufei + yinhaushui) + yu
        if sell_value > capital_per_stock:
            shenglv_fenzi = shenglv_fenzi + 1

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
results_df.to_csv(f'end/oc/all_report{dapan_code}return17_2311.csv', index=False)

# 初始金额
initial_amount = 1_000_000

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
sharpe_ratio = (((1 + results_df['day_return'].mean()) ** 242) - 1 - 0.025) / (
            std_daily_return * 242 ** 0.5) if std_daily_return != 0 else np.nan

shenglv = shenglv_fenzi / shenglv_fenmu

# 输出结果
print(f"年化收益率 (ARR): {ARR:.3f}")
print(f"最大回撤率: {-max_drawdown:.3f}")
print(f"卡尔玛比率: {calmar_ratio:.3f}")
print(f"夏普比率: {sharpe_ratio:.3f}")




