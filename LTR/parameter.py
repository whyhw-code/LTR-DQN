import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split,  GridSearchCV
from sklearn.metrics import make_scorer, ndcg_score
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier

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

Reg_or_Class = 'lr'
dapan_code = '3068'
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

X_train, X_test, y_train, y_test = train_test_split(data_Xtransformed, data_y3transformed, test_size=0.2, random_state=42)

# xgb_reg = XGBRegressor(random_state=42)
# xgb_class = XGBClassifier(random_state=42)
if Reg_or_Class == 'xgbreg':
    model_best = xgb.XGBRegressor(objective='reg:squarederror', booster='gbtree', tree_method='gpu_hist')
elif Reg_or_Class == 'xgbclass':
    model_best = xgb.XGBClassifier(objective='reg:squarederror', booster='gbtree', tree_method='gpu_hist')
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

# 定义要搜索的超参数网格
param_grid = {
    # 'n_estimators': [100, 150, 200],          # 树的数量
    # 'max_depth': [3, 4],                  # 树的最大深度
    # 'learning_rate': [0.01, 0.1]     # 学习率
}

# 使用 GridSearchCV 进行网格搜索
grid_search = GridSearchCV(estimator=model_best, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)

# 训练模型
grid_search.fit(X_train, y_train)

# 输出最佳参数和最佳得分
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.4f}".format(np.sqrt(-grid_search.best_score_)))