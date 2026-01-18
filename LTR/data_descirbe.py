import pandas as pd
import os

# col_name = ['stock_code', 'page', 'advance_reaction', 'star_analyst', 'title_len', 'num_sentence',
#             'avg_sentence_len', 'sd_sentence_len',
#             'num_authors', 'analyst_coverage', 'rm_rf', 'smb', 'hml', 'rmw', 'cma', 'broker_size', 'listed',
#             'prior_performance_avg', 'prior_performance_sd', 'broker_status', 'qid_date', 'real_return', 'ind_1','ind_2','ind_3','ind_4','ind_5','ind_6']
# Xcol_name = ['page', 'advance_reaction', 'star_analyst', 'title_len', 'num_sentence',
#              'avg_sentence_len', 'sd_sentence_len',
#              'num_authors', 'analyst_coverage', 'rm_rf', 'smb', 'hml', 'rmw', 'cma', 'broker_size', 'listed',
#              'prior_performance_avg', 'prior_performance_sd', 'broker_status', 'qid_date', 'ind_1', 'ind_2', 'ind_3', 'ind_4', 'ind_5', 'ind_6']
# Ycol_name = ['real_return']
#
# data = pd.read_csv('data/0060report.csv', usecols=col_name)
# data['qid_date'] = pd.to_numeric(data['qid_date'].str.replace('-', ''))
# data = data[(20171206 <= data['qid_date']) & (data['qid_date'] <= 20230302)]
# # 计算描述统计信息
# description = data.describe()
#
# # 添加偏度和峰度
# description.loc['skewness'] = data.skew()
# description.loc['kurtosis'] = data.kurtosis()
#
# # 绘制表格
# description = description.round(2)  # 将结果保留两位小数
# description = description.transpose()  # 转置，使特征名称作为索引
# description = description[['count', 'mean', 'std', 'min', 'max', 'skewness', 'kurtosis']]  # 重新排列列的顺序
# description.to_csv('data/0060data_description.csv')


# 读取 Excel 文件

# 定义处理每个组的函数
def process_group(group):

    mean_return = (group['real_return']-0.0016).mean()
    count = len(group)
    positive_count = (group['real_return'] > 0).sum()

    # # 计算 top 9 的统计数据
    # top9 = group_sorted.head(9)
    # top9_mean_return = (top9['real_return']-0.0016).mean()
    # top9_count = len(top9)
    # top9_positive_count = (top9['real_return'] > 0).sum()

    return pd.Series({
        'mean_return': mean_return,
        'count': count,
        'positive_count': positive_count,
        # 'top9_mean_return': top9_mean_return,
        # 'top9_count': top9_count,
        # 'top9_positive_count': top9_positive_count
    })

file_path = 'data/0060_report_institution.csv'  # 替换为你的文件路径
df = pd.read_csv(file_path)

# 按照 'institution' 列分组
grouped = df.groupby('institution')

# 获取所有唯一的 'data' 值
all_data = df['qid_date'].unique()

# 创建一个 DataFrame，包含所有 'data'，其余列初始为空
result_df = pd.DataFrame({'qid_date': all_data})

institution_sl_df = pd.DataFrame(columns=['券商名字', '胜率'])

# 遍历每个分组，并合并 'real_return' 列
for institution, group in grouped:
    # 生成一个临时 DataFrame，包含 'data' 和 'real_return'
    temp_df = group[['qid_date', 'real_return']]
    group_results = temp_df.groupby('qid_date').apply(process_group).reset_index()
    group_results.rename(columns={'mean_return': institution, 'count': institution+'_count', 'positive_count':institution+'pos'}, inplace=True)
    sl = (group_results[institution + 'pos'].sum()) / (group_results[institution + '_count'].sum())
    group_results = group_results[['qid_date', institution]]
    new_date = {'券商名字': institution, '胜率': sl}
    institution_sl_df = institution_sl_df.append(new_date, ignore_index=True)
    # 将 'data' 列设置为索引
    group_results.set_index('qid_date', inplace=True)
    # 重命名 'real_return' 列为机构名称
    group_results.rename(columns={'real_return': institution}, inplace=True)
    # 将临时 DataFrame 合并到结果 DataFrame
    result_df = result_df.merge(group_results, on='qid_date', how='left')

# 将 NaN 值替换为 0
result_df.fillna(0, inplace=True)

# 将结果保存到一个新的 Excel 文件中
output_file_path = 'end/institution/0060.csv'  # 替换为你的输出文件路径
result_df.to_csv(output_file_path, index=False,encoding='utf-8-sig')
institution_sl_df.to_csv('end/institution/0060_sl.csv', index=False, encoding='utf-8-sig')

print("拼接后的数据已保存到新的 Excel 文件中。")

# # 读取第一个 Excel 文件，包含需要补充 qid_date 列的表格
# main_file_path = '原始数据/3068report_test_return.xlsx'  # 替换为你的文件路径
# main_df = pd.read_excel(main_file_path)
#
# # 读取第二个 Excel 文件，包含 date 和 qid_date 列的表格
# mapping_file_path = 'data/3068report.csv'  # 替换为你的文件路径
# mapping_df = pd.read_csv(mapping_file_path)
# mapping_df = mapping_df[['date', 'qid_date']]
#
# # 将两个 DataFrame 中的 date 列转换为 datetime 类型
# main_df['date'] = pd.to_datetime(main_df['date'])
# mapping_df['date'] = pd.to_datetime(mapping_df['date'])
# # 去重，确保每个 date 仅保留一个 qid_date
# mapping_df = mapping_df.drop_duplicates(subset=['date','qid_date'])
#
# # 合并两个表格，基于 date 列
# merged_df = pd.merge(main_df, mapping_df, on='date', how='left')
#
# # 将合并后的结果保存到一个新的 Excel 文件中
# output_file_path = 'data/3068_report_institution.csv'  # 替换为你的输出文件路径
# merged_df.to_csv(output_file_path, index=False,encoding='utf_8_sig')
#
# print("补充后的数据已保存到新的 Excel 文件中。")