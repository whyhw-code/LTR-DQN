import torch
import pandas as pd
import numpy as np
import copy
import logging
import os
logging.basicConfig(level=logging.INFO)

col_name1 = ['stock_code','date', 'qid_date', 'real_return']
col_name2 = ['交易日期', '收盘价', '前收盘价', '成交量']

dapan_code = '0060'
test_start = 20161206    #20211022   20161206
test_end = 20230303      #20230303

all_df = pd.read_csv(f'data/final_{dapan_code}report.csv')
all_df['qid_date'] = pd.to_numeric(all_df['qid_date'].str.replace('-', ''))
# all_df = all_df[(test_start <= all_df['qid_date']) & (all_df['qid_date'] <= test_end)]

# trade_df = pd.read_csv('trade_date.csv', parse_dates=['trade_date'])
#
# trade_df.sort_values('trade_date', inplace=True)
#
# # 创建一个trade_date的日期索引，用于快速查找
# trade_dates = trade_df['trade_date'].values
#
# # 遍历df_report中的每一行，更新qid_date
# for i in range(len(all_df)):
#     report_date = all_df.iloc[i]['date']
#
#     # 找到所有小于等于report_date的trade_date
#     valid_trade_dates = trade_dates[trade_dates <= report_date]
#
#     if len(valid_trade_dates) > 0:
#         # 取最大值（即最近的交易日）
#         closest_trade_date = valid_trade_dates.max()
#         all_df.iloc[i, all_df.columns.get_loc('qid_date')] = closest_trade_date
#     else:
#         # 如果没有找到合适的交易日，可以选择保持原样或设置为NaN
#         all_df.iloc[i, all_df.columns.get_loc('qid_date')] = pd.NaT
#
# all_df['qid_date'] = pd.to_datetime(all_df['qid_date'])
# all_df.to_csv(f'data/final_{dapan_code}report.csv', index=False)




# 定义两个文件夹路径
folder_30_68 = 'data/open_close/oc3068'  # 以30,68开头的股票csv存放的文件夹路径
folder_00_60 = 'data/open_close/oc0060'  # 以00,60开头的股票csv存放的文件夹路径

def remove_first_row_from_csv(stock_code):
    # 确定要打开的文件夹
    if stock_code.startswith(('30', '68')):
        folder_path = folder_30_68
    elif stock_code.startswith(('00', '60')):
        folder_path = folder_00_60
    else:
        print(f"Stock code {stock_code} does not match any specified pattern.")
        return
    # 构造CSV文件名
    csv_filename = f"{stock_code}.csv"
    file_path = os.path.join(folder_path, csv_filename)
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"File not found for stock code {stock_code}: {file_path}")
        return
    # 读取CSV文件
    try:
        df = pd.read_csv(file_path, skiprows=1, encoding='gbk', usecols=col_name2)
        df['real_date'] = pd.to_numeric(df['交易日期'].str.replace('-', ''))
        df.set_index('交易日期', inplace=True)
        # 应用shift方法，向上移动
        shifted_df = df.shift(-1)
        df_reset_index = shifted_df.reset_index()
        return df_reset_index
    except Exception as e:
        print(f"An error occurred while processing {csv_filename}: {e}")


def format_stock_code(stock_code_num):
    return f"{stock_code_num:06d}"

result_df = all_df.copy()

# 确保df_first中已经添加了需要填入的空列
columns_to_fill = ['收盘价', '前收盘价', '成交量', 'real_date']
for col in columns_to_fill:
    if col not in result_df.columns:
        result_df[col] = None

# 遍历DataFrame中的所有股票代码，并调用函数处理每个股票代码
for idx, row in result_df.iterrows():
    stock_code = int(row['stock_code'])
    qid_date = int(row['qid_date'])
    stock_code_str = format_stock_code(stock_code)
    oc_df = remove_first_row_from_csv(stock_code_str)
    oc_df['qid_date'] = pd.to_numeric(oc_df['交易日期'].str.replace('-', ''))
    oc_df.set_index('qid_date', inplace=True)
    column_to_add = ['real_date', '收盘价', '前收盘价', '成交量']
    # 使用iloc逐个填充数据
    for col in column_to_add:
        if pd.isna(result_df.at[idx, col]):  # 只有当目标位置为空时才填充
            try:
                if qid_date not in oc_df.index:
                    larger_indices = oc_df.index[oc_df.index > qid_date]
                    if not larger_indices.empty:
                        # 找到其中最小的一个索引
                        closest_larger_index = larger_indices.min()
                        result_df.at[idx, col] = oc_df.at[closest_larger_index, col]
                else:
                    result_df.at[idx, col] = oc_df.at[qid_date, col]
            except Exception as e:
                result_df.at[idx, col] = -1
                print(f"An KeyError while processing {qid_date}, {stock_code}: {e}")

result_df.to_csv(f'data/open_close/{dapan_code}merge_open_close_final.csv', index=False)



# open_close_df = pd.read_csv(f'data/open_close/{oc_dapan}/{oc_code}.csv', usecols=col_name1)

# 抽样
# grouped1 = all_df.groupby('qid_date')
# sampled_data = grouped1.apply(lambda x: x.sample(frac=0.5))
# yuan_all_df = sampled_data






