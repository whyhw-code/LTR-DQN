import pandas as pd

df = pd.read_csv('data/0060report_train.csv')

search_column = 'qid_date'  # 指定要查询的列
search_value = 'nan'  # 指定要查找的值
search_indices = df[df[search_column].isna()].index.tolist()

# 遍历指定值的位置，并将上一行值填充到这些位置
for index in search_indices:
    df.loc[index, search_column] = df.loc[index - 1, search_column]

output_file_path = 'data/0060report_train_qid.csv'
df.to_csv(output_file_path, index=False)