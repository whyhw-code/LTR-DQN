import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. 文件路径
# =========================
xlsx_path = r"../result/batch123/基准+模型结果对比.xlsx"

# =========================
# 2. 读取 Excel
# =========================
main_df = pd.read_excel(xlsx_path, sheet_name="主板")
chinext_df = pd.read_excel(xlsx_path, sheet_name="创业板")

# 清理列名
main_df.columns = main_df.columns.astype(str).str.strip()
chinext_df.columns = chinext_df.columns.astype(str).str.strip()

print("主板列名：", main_df.columns.tolist())
print("创业板列名：", chinext_df.columns.tolist())

# =========================
# 3. 日期与数值处理
# =========================
main_df["qid"] = pd.to_datetime(main_df["qid"])
chinext_df["qid"] = pd.to_datetime(chinext_df["qid"])

for df in [main_df, chinext_df]:
    for col in [
        "CSI 300 Index",
        "ChiNext Index",
        "LambdaMART",
        "LTR-DQN",
        "number of stocks"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

main_df["number of stocks"] = main_df["number of stocks"].fillna(0)
chinext_df["number of stocks"] = chinext_df["number of stocks"].fillna(0)

print("主板选股数量分布：")
print(main_df["number of stocks"].value_counts().sort_index())

print("创业板选股数量分布：")
print(chinext_df["number of stocks"].value_counts().sort_index())

# =========================
# 4. 画图函数
# =========================
def plot_fig6(df, market_col, title, output_name):
    required_cols = ["qid", "LambdaMART", "LTR-DQN", market_col, "number of stocks"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少列：{col}")

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 5.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # 上方资金曲线
    ax1.plot(df["qid"], df["LambdaMART"], label="LambdaMART", linewidth=1.6)
    ax1.plot(df["qid"], df["LTR-DQN"], label="LTR-DQN", linewidth=1.8)
    ax1.plot(df["qid"], df[market_col], label=market_col, linewidth=1.4)

    ax1.set_ylabel("Total return")
    ax1.set_title(title)
    ax1.legend(loc="upper left", ncol=3, fontsize=8)
    ax1.grid(alpha=0.3)

    # 下方柱状图
    ax2.bar(
        df["qid"],
        df["number of stocks"],
        width=1.0,
        alpha=0.45
    )

    ax2.set_ylabel("Number\nof stocks")
    ax2.set_xlabel("Trading Day")
    ax2.set_ylim(0, 4.5)
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    # plt.savefig(output_name, dpi=300, bbox_inches="tight")
    plt.show()

# =========================
# 5. 复现 Fig. 6
# =========================
plot_fig6(
    main_df,
    market_col="CSI 300 Index",
    title="(a) Main board market",
    # output_name="fig6_main_board.png"
)

plot_fig6(
    chinext_df,
    market_col="ChiNext Index",
    title="(b) ChiNext market",
    # output_name="fig6_chinext.png"
)