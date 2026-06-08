# MESG.py
# 主板 ESG 附录图
# 从“基准+模型结果对比.xlsx”读取 CSI 300 Index / Baseline portfolio / No ESG
# NS / PI 按 data/ESG/0060temp_test_ndcg_train3_esg.csv + meiri_xuanze.csv 计算
# 只出图，不保存文件

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 基础配置
# ============================================================

# 当前代码放在 LTR-DQN-main/code/ 下运行
CODE_DIR = Path(__file__).resolve().parent

dapan_code = "0060"
test_batch = 123
train_year = 3
LTR = "ndcg"

test_start = 20211207
test_end = 20230303

initial_capital = 1_000_000
shouxufei = 0.0003
yinhaushui = 0.001

# 是否把每条曲线都重置为 1 million 起点
NORMALIZE_TO_1M = True

# 主板 ESG 阈值
# NS 25% / PI 25%：ESG >= 5.52
# NS 50% / PI 50%：ESG >= 6.02
ESG_25 = 5.52
ESG_50 = 6.02

# Excel 中 No ESG 对应的列。
# 你的“基准+模型结果对比.xlsx”里通常是 LTR-DQN。
NO_ESG_COL = "LTR-DQN"


# ============================================================
# 路径
# ============================================================

# 优先从这里读取。
# 如果你的文件名或位置不同，直接改 comparison_excel_path。
comparison_excel_path = CODE_DIR / f"../result/batch{test_batch}/基准+模型结果对比.xlsx"
comparison_sheet = "主板"

# 如果上面的路径不存在，代码会尝试这些候选路径
comparison_excel_candidates = [
    comparison_excel_path,
    CODE_DIR / f"../result/batch{test_batch}/基准+模型结果对比(1).xlsx",
    CODE_DIR / "基准+模型结果对比.xlsx",
    CODE_DIR / "基准+模型结果对比(1).xlsx",
]

# ESG 预测结果
esg_temp_path = CODE_DIR / f"data/ESG/{dapan_code}temp_test_{LTR}_train{train_year}_esg.csv"

# 每日固定选择数量
select_path = CODE_DIR / f"temp/oc/batch{test_batch}/meiri_xuanze.csv"


# ============================================================
# 工具函数
# ============================================================

def find_existing_path(candidates):
    for p in candidates:
        if Path(p).exists():
            return Path(p)

    raise FileNotFoundError(
        "找不到“基准+模型结果对比.xlsx”。已尝试路径：\n" +
        "\n".join(str(p) for p in candidates)
    )


def to_int_date_series(s):
    """
    将日期列统一成 YYYYMMDD 整数。
    兼容：
    20211207
    2021-12-07
    2021/12/07
    20211207.0
    Excel 日期
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_numeric(s.dt.strftime("%Y%m%d"), errors="coerce").astype("Int64")

    raw = s.astype(str).str.strip()

    # Excel 日期序列号兜底，例如 44537
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().mean() > 0.8:
        if numeric.dropna().between(30000, 60000).mean() > 0.8:
            d = pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
            return pd.to_numeric(d.dt.strftime("%Y%m%d"), errors="coerce").astype("Int64")

    raw = raw.str.replace(r"\.0$", "", regex=True)
    raw = raw.str.replace("-", "", regex=False)
    raw = raw.str.replace("/", "", regex=False)

    return pd.to_numeric(raw, errors="coerce").astype("Int64")


def to_datetime_from_yyyymmdd(s):
    s_int = to_int_date_series(s)
    raw = s_int.astype(str).str.strip().str.replace("<NA>", "", regex=False)
    return pd.to_datetime(raw, format="%Y%m%d", errors="coerce")


def normalize_curve(df, value_col="total_profit"):
    """
    将资金曲线统一归一化到 1 million 起点。
    """
    df = df.copy()

    if NORMALIZE_TO_1M and len(df) > 0:
        first_value = df[value_col].iloc[0]
        if pd.notna(first_value) and first_value != 0:
            df[value_col] = df[value_col] / first_value * initial_capital

    df["total_profit_million"] = df[value_col] / 1_000_000
    return df


def trade_one_day(top_stocks, current_capital):
    """
    给定当天选股结果，计算当天收盘后的资金。
    """
    if len(top_stocks) == 0:
        return current_capital, 0.0

    capital_per_stock = current_capital / len(top_stocks)
    day_total_profit = 0

    for _, row in top_stocks.iterrows():
        pclose = row["pclose"]
        close = row["close"]

        if pd.isna(pclose) or pd.isna(close) or pclose <= 0:
            day_total_profit += capital_per_stock
            continue

        # 计算能买的股数（向下取整）
        shou_num = int(capital_per_stock / (100 * pclose))
        pfei = shou_num * 100 * pclose * shouxufei

        shares_bought = int((capital_per_stock - pfei) / (100 * pclose)) * 100
        yu = capital_per_stock - pfei - shares_bought * pclose

        if shares_bought == 0:
            yu = capital_per_stock

        # 卖出后资金，卖出扣手续费和印花税
        sell_value = (
            shares_bought * close
            - shares_bought * close * (shouxufei + yinhaushui)
            + yu
        )

        day_total_profit += sell_value

    day_return = (day_total_profit - current_capital) / current_capital
    return day_total_profit, day_return


# ============================================================
# 1. 从“基准+模型结果对比.xlsx”读取基础曲线
# ============================================================

def read_comparison_sheet():
    excel_path = find_existing_path(comparison_excel_candidates)

    xls = pd.ExcelFile(excel_path)
    if comparison_sheet in xls.sheet_names:
        sheet_name = comparison_sheet
    else:
        # 兜底：找包含“主”的 sheet
        main_like = [s for s in xls.sheet_names if "主" in str(s)]
        if len(main_like) > 0:
            sheet_name = main_like[0]
        else:
            raise ValueError(
                f"{excel_path} 中找不到 sheet：{comparison_sheet}。"
                f"当前 sheet：{xls.sheet_names}"
            )

    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df.columns = df.columns.astype(str).str.strip()

    if "qid_date" not in df.columns:
        raise ValueError(
            f"{excel_path} 的 {sheet_name} sheet 中找不到 qid_date 列。"
            f"当前列名：{df.columns.tolist()}"
        )

    df["qid_date"] = to_int_date_series(df["qid_date"])
    df = df.dropna(subset=["qid_date"]).copy()
    df["qid_date"] = df["qid_date"].astype(int)

    df = df[(test_start <= df["qid_date"]) & (df["qid_date"] <= test_end)].copy()
    df = df.sort_values("qid_date")

    df["date"] = pd.to_datetime(
        df["qid_date"].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )

    return df


def find_column(df, candidates):
    """
    在 Excel 中查找列名，支持多个候选名。
    """
    columns = list(df.columns)
    columns_lower = {str(c).strip().lower(): c for c in columns}

    for c in candidates:
        key = str(c).strip().lower()
        if key in columns_lower:
            return columns_lower[key]

    raise ValueError(
        f"找不到列：{candidates}\n"
        f"当前列名：{columns}"
    )


def get_excel_curve(comparison_df, source_col_candidates):
    source_col = find_column(comparison_df, source_col_candidates)

    df = comparison_df[["date", source_col]].copy()
    df[source_col] = pd.to_numeric(df[source_col], errors="coerce")
    df = df.dropna(subset=["date", source_col]).copy()

    df = df.rename(columns={source_col: "total_profit"})

    return normalize_curve(df[["date", "total_profit"]])


# ============================================================
# 2. ESG 相关策略：NS / PI
# ============================================================

def get_select_map():
    """
    读取每日 DQN 固定选择数量。
    主板用 60 列。
    """
    xuanze_df = pd.read_csv(
        select_path,
        usecols=["qid_date", "3068", "60"]
    )

    xuanze_df["qid_date"] = to_int_date_series(xuanze_df["qid_date"])
    xuanze_df = xuanze_df.dropna(subset=["qid_date"]).copy()
    xuanze_df["qid_date"] = xuanze_df["qid_date"].astype(int)

    xuanze_df["60"] = pd.to_numeric(
        xuanze_df["60"],
        errors="coerce"
    ).fillna(0).astype(int)

    return dict(zip(xuanze_df["qid_date"], xuanze_df["60"]))


def get_esg_temp_data():
    df = pd.read_csv(esg_temp_path)

    df["qid_date"] = to_int_date_series(df["qid_date"])
    df = df.dropna(subset=["qid_date"]).copy()
    df["qid_date"] = df["qid_date"].astype(int)

    df = df[(test_start <= df["qid_date"]) & (df["qid_date"] <= test_end)].copy()
    df = df.sort_values("qid_date")

    required_cols = ["qid_date", "stock_code", "prediction", "ESG", "close", "pclose"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"ESG temp 文件缺少列：{missing_cols}")

    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df["ESG"] = pd.to_numeric(df["ESG"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["pclose"] = pd.to_numeric(df["pclose"], errors="coerce")

    return df


def get_strategy_curve(temp_df, select_map, mode="ns", esg_threshold=None):
    """
    mode:
    - ns: 先按 prediction 选 top_n，再筛 ESG >= threshold
    - pi: 先筛 ESG >= threshold，再按 prediction 选 top_n
    """
    current_capital = initial_capital
    daily_results = []

    for qid_date, group in temp_df.groupby("qid_date"):
        top_n = int(select_map.get(int(qid_date), 0))

        if top_n <= 0:
            day_total_profit = current_capital
            day_return = 0.0

        else:
            if mode == "ns":
                # NS: 先按 prediction 选 top_n，再筛 ESG
                top_stocks = group.nlargest(min(top_n, len(group)), "prediction")
                top_stocks = top_stocks[top_stocks["ESG"] >= esg_threshold]

            elif mode == "pi":
                # PI: 先筛 ESG，再按 prediction 选 top_n
                group_filtered = group[group["ESG"] >= esg_threshold]
                top_stocks = group_filtered.nlargest(min(top_n, len(group_filtered)), "prediction")

            else:
                raise ValueError(f"Unknown mode: {mode}")

            day_total_profit, day_return = trade_one_day(
                top_stocks=top_stocks,
                current_capital=current_capital
            )

        current_capital = day_total_profit

        daily_results.append({
            "qid_date": qid_date,
            "total_profit": day_total_profit,
            "day_return": day_return
        })

    result_df = pd.DataFrame(daily_results)
    result_df["date"] = to_datetime_from_yyyymmdd(result_df["qid_date"])

    return normalize_curve(result_df[["date", "total_profit", "day_return"]])


# ============================================================
# 画图
# ============================================================

def format_quarter_axis(ax, curves):
    all_dates = []
    for _, df in curves:
        if "date" in df.columns:
            all_dates.extend(df["date"].dropna().tolist())

    if len(all_dates) == 0:
        return

    min_date = min(all_dates)
    max_date = max(all_dates)

    quarter_starts = pd.date_range(
        start=pd.Timestamp(
            year=min_date.year,
            month=((min_date.month - 1) // 3) * 3 + 1,
            day=1
        ),
        end=max_date,
        freq="QS-JAN"
    )

    ticks = [min_date] + [d for d in quarter_starts if d > min_date]
    ticks = sorted(list(dict.fromkeys(ticks)))

    ax.set_xticks(ticks)

    labels = []
    for d in ticks:
        q = (d.month - 1) // 3 + 1
        labels.append(f"{d.year}/Q{q}")

    ax.set_xticklabels(labels)


def plot_main_appendix(curves):
    fig, ax = plt.subplots(figsize=(7.6, 3.5), dpi=120)

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    color_map = {
        "CSI 300 Index": "#4472C4",
        "Baseline portfolio": "#A5A5A5",
        "No ESG": "#FF0000",
        "NS 25%": "#FFC000",
        "NS 50%": "#ED7D31",
        "PI 25%": "#70AD47",
        "PI 50%": "#264478",
    }

    for label, df in curves:
        ax.plot(
            df["date"],
            df["total_profit_million"],
            label=label,
            color=color_map[label],
            linewidth=1.4
        )

    ax.set_xlabel("Trading Day", fontsize=9, fontweight="bold")
    ax.set_ylabel("Total Fund (million)", fontsize=9, fontweight="bold")

    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)

    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.grid(axis="x", visible=False)

    for spine in ax.spines.values():
        spine.set_color("#A0A0A0")
        spine.set_linewidth(0.8)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=7,
        fontsize=6,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.0,
    )

    format_quarter_axis(ax, curves)

    # 纵轴自动上浮，避免曲线被截断
    y_values = []
    for _, df in curves:
        if "total_profit_million" in df.columns:
            y_values.extend(df["total_profit_million"].dropna().tolist())

    if len(y_values) > 0:
        y_max = max(y_values)
        step = 0.5
        upper = np.ceil((y_max * 1.08) / step) * step

        ax.set_ylim(0, upper)
        ax.set_yticks(np.arange(0, upper + 0.001, step))

    plt.tight_layout()
    plt.show()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    comparison_df = read_comparison_sheet()

    # 这三条直接从“基准+模型结果对比.xlsx”的主板 sheet 读取
    index_curve = get_excel_curve(
        comparison_df=comparison_df,
        source_col_candidates=["CSI 300 Index", "CSI300 Index", "CSI 300"]
    )

    baseline_curve = get_excel_curve(
        comparison_df=comparison_df,
        source_col_candidates=["Baseline portfolio", "Baseline Portfolio", "Baseline"]
    )

    no_esg_curve = get_excel_curve(
        comparison_df=comparison_df,
        source_col_candidates=[NO_ESG_COL, "No ESG", "LTR_DQN", "LTRDQN"]
    )

    # 这四条按 ESG temp + 每日选择数量计算
    temp_df = get_esg_temp_data()
    select_map = get_select_map()

    ns_25_curve = get_strategy_curve(
        temp_df=temp_df,
        select_map=select_map,
        mode="ns",
        esg_threshold=ESG_25
    )

    ns_50_curve = get_strategy_curve(
        temp_df=temp_df,
        select_map=select_map,
        mode="ns",
        esg_threshold=ESG_50
    )

    pi_25_curve = get_strategy_curve(
        temp_df=temp_df,
        select_map=select_map,
        mode="pi",
        esg_threshold=ESG_25
    )

    pi_50_curve = get_strategy_curve(
        temp_df=temp_df,
        select_map=select_map,
        mode="pi",
        esg_threshold=ESG_50
    )

    curves = [
        ("CSI 300 Index", index_curve),
        ("Baseline portfolio", baseline_curve),
        ("No ESG", no_esg_curve),
        ("NS 25%", ns_25_curve),
        ("NS 50%", ns_50_curve),
        ("PI 25%", pi_25_curve),
        ("PI 50%", pi_50_curve),
    ]

    plot_main_appendix(curves)
