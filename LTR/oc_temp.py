import torch
import pandas as pd
import numpy as np
import copy
import logging
import os
logging.basicConfig(level=logging.INFO)


all_df0060 = pd.read_csv(f'data/0060merge_open_close_final.csv')
all_df3068 = pd.read_csv(f'data/3068merge_open_close_final.csv')

# 设置MultiIndex以加速查找
all_df0060.set_index(['qid_date', 'stock_code'], inplace=True)
all_df3068.set_index(['qid_date', 'stock_code'], inplace=True)

def nnn(filepath):
    df = pd.read_csv(filepath)
    print("Original DataFrame:")
    print(df)

    # 第一步：处理 'pclose' 列中的缺失值，将其设为100
    df['pclose'].fillna(100, inplace=True)

    # 第二步：创建一个新的 'close' 列，用于存储更新后的值
    df['new_close'] = df['close']

    # 第三步：更新 'close' 列中的缺失值
    mask_close_nan = df['close'].isna()

    # # 使用 apply 和 lambda 函数来处理每行数据
    # df.loc[mask_close_nan, 'new_close'] = df.loc[mask_close_nan].apply(
    #     lambda row: 100 + 100 * row['real_return'], axis=1
    # )

    # 如果你想要直接更新原来的 'close' 列而不是创建新的列
    df.loc[mask_close_nan, 'close'] = df.loc[mask_close_nan].apply(
        lambda row: 100 + 100 * row['real_return'], axis=1
    )
    df.to_csv(filepath, index=False)
    print("\nUpdated DataFrame:")

# 创建一个函数来处理单个CSV文件
def process_csv(file_path, df):
    # 读取CSV文件
    temp_df = pd.read_csv(file_path)
    temp_df.set_index(['qid_date', 'stock_code'], inplace=True)

    # 创建一个新的DataFrame用于存储更新后的数据，先复制一份原始数据
    updated_df = temp_df.copy()
    columns_to_fill = ['close', 'pclose']
    for col in columns_to_fill:
        if col not in updated_df.columns:
            updated_df[col] = None

    # 直接按照temp_df中索引的顺序一个个匹配
    for idx in temp_df.index:
        try:
            if isinstance(df.loc[idx, 'close'], pd.Series):
                updated_df.loc[idx, 'close'] = df.loc[idx, 'close']
                updated_df.loc[idx, 'pclose'] = df.loc[idx, 'pclose']
            else:
                updated_df.loc[idx, 'close'] = df.loc[idx, 'close'].iloc[0]
                updated_df.loc[idx, 'pclose'] = df.loc[idx, 'pclose'].iloc[0]
        except KeyError as k:
            # 如果没有找到匹配项，跳过
            print(k)
        except ValueError as v:
            print(v)
            continue

    # 如果需要重置索引回到原始格式
    updated_df.reset_index(inplace=True)

    # 保存更新后的结果
    updated_df.to_csv(file_path, index=False)

    return 1


folder_path = 'temp/oc/batch123'
# 遍历文件夹中的所有CSV文件
for filename in os.listdir(folder_path):
    if filename.endswith('.csv') and ('map' in filename or 'ndcg' in filename or 'pair' in filename):
        file_path = os.path.join(folder_path, filename)

        # 根据文件名开头选择对应的DataFrame
        if filename.startswith('0060'):
            ocdf = all_df0060
        elif filename.startswith('3068'):
            ocdf = all_df3068
        else:
            print(f"Skipping file {filename} as it does not start with 0060 or 3068.")
            continue

        # 处理CSV文件并保存更新后的结果
        # process_csv(file_path, ocdf)
        nnn(file_path)
        print(f"Updated {filename} with close and pclose data.")


print("All CSV files have been processed.")








