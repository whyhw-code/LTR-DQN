import pandas as pd

test_batch = 3
dapan_code = '0060'
rt = 'topreturn'  # 'return', 'topreturn'
m = 'map100max11'   # 'map100max11','pairmax11' 'ndcg11'
train_year_list = [2, 3, 4, 5]

for train_year in train_year_list:
    print(train_year)
    # 读取两个 CSV 文件
    df_incomplete = pd.read_csv(f'end/batch{test_batch}/{dapan_code}{rt}_test_{m}_train{train_year}.csv')
    df_complete = pd.read_excel(f'D:/py_code/DQN_base-main/stock/result/batch{test_batch}/3068_ndcg11_1year_train1_top4.xlsx')

    # 获取完整的 qid_date 列
    complete_qid_date = df_complete['qid_date']

    # 创建一个新的 DataFrame，只包含完整的 qid_date 列
    df_new = pd.DataFrame({'qid_date': complete_qid_date})

    # 将不完整 DataFrame 中的数据按照 qid_date 列进行合并
    df_merged = pd.merge(df_new, df_incomplete, on='qid_date', how='left')
    if rt == 'return':
        # 删除 'Unnamed: 0' 列
        df_merged.drop(columns=['Unnamed: 0'], inplace=True)
        # 填补 '1' 列的空值部分为上一行的数据，初始为 1
        df_merged['1'] = df_merged['1'].fillna(method='ffill').fillna(1)
        # 将其余列空的位置补为 0
        df_merged.fillna(0, inplace=True)
    elif rt == 'topreturn':
        df_merged['Unnamed: 4'] = df_merged['Unnamed: 4'].fillna(method='ffill').fillna(1)
        df_merged['Unnamed: 8'] = df_merged['Unnamed: 8'].fillna(method='ffill').fillna(1)
        # 将其余列空的位置补为 0
        df_merged.fillna(0, inplace=True)

    # 保存补全后的 DataFrame 到新的 CSV 文件
    df_merged.to_csv(f'end/batch{test_batch}/000_{dapan_code}{rt}_test_{m}_train{train_year}.csv', index=False)
