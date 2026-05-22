
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, LogisticRegression
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


train_df = all_df[(20210106 <= all_df['qid_date']) & (all_df['qid_date'] <= 20211206)]


data = train_df
data_X = data[Xcol_name]
data_Xcopy = data_X[:]
scalerX = MinMaxScaler()
data_Xtransformed = scalerX.fit_transform(data_Xcopy)


data_y3 = data[Ycol_name].values.reshape(-1,1)
data_y3copy = data_y3[:]
scaler3 = MinMaxScaler()
data_y3transformed = scaler3.fit_transform(data_y3copy)


# --- 2. 训练 XGBoost 模型 ---
print("🚀 正在训练 XGBoost 模型...")
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', booster='gbtree', tree_method='gpu_hist', n_estimators=100, max_depth=4,learning_rate=0.1, subsample=1.0,colsample_bytree=0.8)
xgb_model.fit(data_Xtransformed, data_y3transformed.ravel())

# 提取 XGB 特征重要性
xgb_importance = xgb_model.feature_importances_

# --- 3. 训练 Lasso 模型 ---
print("🚀 正在训练 Lasso 模型...")
lasso_model = Lasso(alpha=0.0001)
lasso_model.fit(data_Xtransformed, data_y3transformed.ravel())

# 提取 Lasso 系数 (取绝对值)
lasso_coef = np.abs(lasso_model.coef_)

# --- 4. 读取现有的 LTR-DQN 结果 ---
ltr_file = '../result/batch123/feature_importance_LTR-DQN_3068.csv'

df_ltr = pd.read_csv(ltr_file)
df_ltr = df_ltr[['Features', 'LTR-DQN']].sort_values('Features').reset_index(drop=True)

# --- 5. Min-Max 归一化函数 (核心修改点) ---
def min_max_normalize(arr):
    min_val = arr.min()
    max_val = arr.max()
    # 防止分母为 0 (如果所有值都一样)
    if max_val - min_val == 0:
        return arr - min_val
    return (arr - min_val) / (max_val - min_val)

# 对 XGB 和 Lasso 结果进行 Min-Max 归一化
xgb_normalized = min_max_normalize(xgb_importance)
lasso_normalized = min_max_normalize(lasso_coef)

# --- 6. 创建新模型的结果 DataFrame ---
feature_names = Xcol_name

df_xgb = pd.DataFrame({
    'Features': feature_names,
    'XGBoost_R': xgb_normalized
}).sort_values('Features').reset_index(drop=True)

df_lasso = pd.DataFrame({
    'Features': feature_names,
    'LR': lasso_normalized
}).sort_values('Features').reset_index(drop=True)

# --- 7. 合并所有结果 ---
print("📊 正在合并所有模型结果...")

final_df = df_ltr.copy()
final_df = final_df.merge(df_xgb, on='Features', how='left')
final_df = final_df.merge(df_lasso, on='Features', how='left')

# 重新排列列顺序
final_df = final_df[['Features', 'LTR-DQN', 'LR','XGBoost_R']]

# --- 8. 保存最终结果 ---
output_filename = f'../result/batch123/feature_importance_{dapan_code}.csv'
final_df.to_csv(output_filename, index=False)
print(f"✅ 成功！所有特征重要性已保存至: {output_filename}")

model_cols = ['LTR-DQN', 'LR', 'XGBoost_R']
top_features = {
    model: final_df.nlargest(5, model).sort_values(model, ascending=False)
    for model in model_cols
}

# 设置全局字体（支持中文）
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'DejaVu Sans',  # 解决中文显示问题[5](@ref)
})

# 初始化图表
plt.figure(figsize=(12, 8))
bar_height = 0.25  # 调整柱子高度
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
y_offsets = [0, 5, 10]
all_feature_labels = [''] * 15  # 预初始化标签数组[2](@ref)

# 绘制每个模型的柱子
for i, (model, color) in enumerate(zip(model_cols, colors)):
    # 按分值降序获取特征（高分在上）
    df_model = final_df.nlargest(5, model).sort_values(model, ascending=False)

    # 调整位置生成逻辑：从0到4升序排列（高分值在上）[1](@ref)
    y_pos = np.arange(5) + y_offsets[i]  # 生成0-4的位置序列

    # 绘制水平柱状图（保持数据降序排列）
    bars = plt.barh(y_pos, df_model[model],
                    height=bar_height, color=color, label=model)

    # 添加数值标签（右对齐）
    for bar, val in zip(bars, df_model[model]):
        plt.text(bar.get_width() + 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}",
                 va='center', ha='left', fontsize=12)

    # 按降序存储特征名称（与柱子位置同步）[4](@ref)
    all_feature_labels[y_offsets[i]:y_offsets[i] + 5] = df_model['Features'].tolist()

# 设置y轴刻度和反转坐标轴[2](@ref)
plt.yticks(np.arange(15), all_feature_labels)
plt.gca().invert_yaxis()  # 强制反转y轴显示方向

# 美化图表元素
plt.legend(loc='lower right',
           bbox_to_anchor=(1, 0),
           framealpha=0.9,
           fontsize=12)
plt.xlabel('Feature Importance', labelpad=10)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



