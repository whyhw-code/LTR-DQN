import pandas as pd

df_stocks = pd.read_csv('temp/oc/ESG/0060temp_test_pairwise11_train3.csv')

df_esg = pd.read_csv('temp/oc/ESG/ESG.csv')

merged_df = pd.merge(df_stocks, df_esg[['stock_code', 'ESG']], on='stock_code', how='left')

output_file = 'temp/oc/ESG/0060temp_test_pairwise11_train3_esg.csv'  # 输出文件路径
merged_df.to_csv(output_file, index=False)

print(f"Updated stock data with ESG scores saved to {output_file}")