import numpy as np
import pandas as pd

col_name = ['trade_date',  'pct_chg', 'total_profit']

dapan_code = '0060'

results_df = pd.read_csv(f'data/{dapan_code}merge.csv', usecols=col_name)

initial_amount = 1000000
daily_results = []
shenglv_fenmu = 0
shenglv_fenzi = 0


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
std_daily_return = results_df['pct_chg'].std()
sharpe_ratio = (((1 + results_df['pct_chg'].mean()) ** 242) - 1 - 0.025) / (
            std_daily_return * 242 ** 0.5) if std_daily_return != 0 else np.nan

shenglv = (results_df['pct_chg'] > 0).sum() / trading_days

# 输出结果
print(f"年化收益率 (ARR): {ARR:.3f}")
print(f"最大回撤率: {-max_drawdown:.3f}")
print(f"卡尔玛比率: {calmar_ratio:.3f}")
print(f"夏普比率: {sharpe_ratio:.3f}")
print(f"WR: {shenglv:.3f}")


