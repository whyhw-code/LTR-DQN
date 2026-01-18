import pandas as pd

# 读取Excel和CSV文件
xlsx_df = pd.read_excel("data/report_merge_broker_all.xlsx")
csv_df = pd.read_csv("data/0060merge_open_close_final.csv")

# 根据report_id左连接（保留所有Excel中的记录）
merged_df = pd.merge(
    left=xlsx_df,
    right=csv_df[["report_id", "qid_date", "close", "pclose"]],  # 仅选取需要的列
    on="report_id",
    how="left"
)

# 处理缺失值（若csv中无匹配数据，则置空）
merged_df.fillna({
    "qid_date": "",
    "close": "",
    "pclose": ""
}, inplace=True)

# 调整列顺序（按需求排序）
final_df = merged_df[["report_id", "stock_code", "institution", "real_return",
                    "qid_date", "close", "pclose"]]

# 保存结果
final_df.to_excel("data/0060report_broker_merged.xlsx", index=False, engine="openpyxl")