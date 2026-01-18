
import torch
import pandas as pd
import numpy as np
# 绘图相关
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVC,SVR

from sklearn.tree import DecisionTreeRegressor
from xgboost import plot_importance
import xgboost as xgb


from sklearn import metrics
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

plt.style.use("fivethirtyeight")

# 警告
import warnings
warnings.filterwarnings('ignore')

# def de_copy(temp):
#     return temp[temp['real_return'].abs() <= 0.05]

# 检查是否有可用的 GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("device:cuda")
else:
    device = torch.device('cpu')
    print("device:cpu")

col_name = ['stock_code', 'page', 'advance_reaction', 'star_analyst', 'title_len', 'num_sentence',
            'avg_sentence_len', 'sd_sentence_len',
            'num_authors', 'analyst_coverage', 'rm_rf', 'smb', 'hml', 'rmw', 'cma', 'broker_size', 'listed',
            'prior_performance_avg', 'prior_performance_sd', 'broker_status', 'qid_date', 'real_return','up_down', 'ind_1','ind_2','ind_3','ind_4','ind_5','ind_6',
            'close', 'pclose'
            ]
Xcol_name = ['page', 'advance_reaction', 'star_analyst', 'title_len', 'num_sentence',
             'avg_sentence_len', 'sd_sentence_len',
             'num_authors', 'analyst_coverage', 'rm_rf', 'smb', 'hml', 'rmw', 'cma', 'broker_size', 'listed',
             'prior_performance_avg', 'prior_performance_sd', 'broker_status', 'ind_1', 'ind_2', 'ind_3', 'ind_4', 'ind_5', 'ind_6'
              ]
# Ycol_name = ['real_return']
# Ycol_name = ['up_down']

Reg_or_Class = 'xgbreg'
dapan_code = '0060'
test_batch = 123
train_or_test = 'test'
train_year = 3

shouxufei = 0.0003
yinhaushui = 0.001
if Reg_or_Class in ['svc','xgbclass','mlpclass']:
    Ycol_name = ['up_down']
else:
    Ycol_name = ['real_return']

all_df = pd.read_csv(f'data/{dapan_code}merge_open_close_final.csv', usecols=col_name)
if train_year == 2:
    train_df = all_df[(20191206 <= all_df['qid_date']) & (all_df['qid_date'] <= 20211206)]
elif train_year == 3:
    train_df = all_df[(20181206 <= all_df['qid_date']) & (all_df['qid_date'] <= 20211206)]
elif train_year == 4:
    train_df = all_df[(20171206 <= all_df['qid_date']) & (all_df['qid_date'] <= 20211206)]
test_df = all_df[(20211207 <= all_df['qid_date']) & (all_df['qid_date'] <= 20230303)]

data = train_df
data_X = data[Xcol_name]
data_Xcopy = data_X[:]
scalerX = MinMaxScaler()
data_Xtransformed = scalerX.fit_transform(data_Xcopy)


data_y3 = data[Ycol_name].values.reshape(-1,1)
data_y3copy = data_y3[:]
scaler3 = MinMaxScaler()
data_y3transformed = scaler3.fit_transform(data_y3copy)


# 建立模型
if Reg_or_Class == 'xgbreg':
    model_best = xgb.XGBRegressor(objective='reg:squarederror', booster='gbtree', tree_method='gpu_hist', n_estimators=100, max_depth=4,learning_rate=0.1, subsample=1.0,colsample_bytree=0.8)
elif Reg_or_Class == 'xgbclass':
    model_best = xgb.XGBClassifier(objective='reg:squarederror', booster='gbtree', tree_method='gpu_hist', n_estimators=200,max_depth=4,learning_rate=0.1, subsample=1.0,colsample_bytree=0.8)
elif Reg_or_Class == 'svr':
    model_best = SVR(kernel='rbf', C=1.0)
elif Reg_or_Class == 'svc':
    model_best = SVC(kernel='rbf', C=1.0)
elif Reg_or_Class == 'mlpreg':
    model_best = MLPRegressor(hidden_layer_sizes=(24), max_iter=100, random_state=42)
elif Reg_or_Class == 'mlpclass':
    model_best = MLPClassifier(hidden_layer_sizes=(24), max_iter=100, random_state=42)
elif Reg_or_Class == 'lr':
    model_best = Lasso(alpha=0.0001)

rmse_list = []
r2_list = []
mae_list = []

model_best.fit(data_Xtransformed, data_y3transformed)

'''测试数据'''
data_test = test_df
test_X = data_test[Xcol_name]
test_Xcopy = test_X[:]
# test_scalerX = MinMaxScaler()
test_Xtransformed = scalerX.fit_transform(test_Xcopy)

y_modelpred = model_best.predict(test_Xtransformed)

# 反归一化
# y_test_orig = scaler3.inverse_transform(y_test.reshape(-1,1))
y_pred_orig = scaler3.inverse_transform(y_modelpred.reshape(-1,1))
temp = data_test[['qid_date', 'stock_code', 'real_return', 'close', 'pclose']]
temp["prediction"] = y_pred_orig

# temp.to_csv(f'temp/oc/batch{test_batch}/{dapan_code}temp_test_{Reg_or_Class}_train{train_year}.csv')


df = temp
df.set_index('qid_date', inplace=True)

initial_capital = 5000000
daily_results = []
shenglv_fenmu = 0
shenglv_fenzi = 0

for qid_date, group in df.groupby('qid_date'):
    shu = (group['prediction'] == 1).sum()
    if Reg_or_Class in ['xgbclass','svc','mlpclass']:
        top_stocks = group.nlargest(min(shu, len(group)), 'prediction')
    else:
        shu1 = (group['prediction'] > 0).sum()
        top_stocks = group.nlargest(min(4, len(group)), 'prediction')
    if len(top_stocks) == 0:
        print(f"Warning: No stocks available on {qid_date}. Skipping this date.")
        continue

    capital_per_stock = initial_capital / len(top_stocks)
    shenglv_fenmu = shenglv_fenmu + len(top_stocks)

    day_total_profit = 0

    for _, row in top_stocks.iterrows():
        # sell_value = capital_per_stock * (1 - shouxufei) * (1 + row['real_return']) * (1 - shouxufei - yinhaushui)

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
    print(qid_date,day_total_profit,day_return)

# 将结果转换为DataFrame
results_df = pd.DataFrame(daily_results)

# print(results_df)
# results_df.to_csv(
#     f'end/oc/batch{test_batch}/{dapan_code}return_{train_or_test}_{Reg_or_Class}_train{train_year}.csv',
#     index=False)

# 初始金额
initial_amount = 5000000

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
sharpe_ratio = (((1+results_df['day_return'].mean())**242)-1-0.025)/(std_daily_return*242**0.5) if std_daily_return != 0 else np.nan

shenglv = shenglv_fenzi / shenglv_fenmu

# 输出结果
print(f"年化收益率 (ARR): {ARR:.3f}")
print(f"最大回撤率: {-max_drawdown:.3f}")
print(f"卡尔玛比率: {calmar_ratio:.3f}")
print(f"夏普比率: {sharpe_ratio:.3f}")
print(f"WR: {shenglv:.3f}")



