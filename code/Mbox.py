# M_sampling_boxplot.py
# 主板 sampling rate 箱线图
# 只出图，不保存文件

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xgboost as xgb


# ============================================================
# 基础配置
# ============================================================

# 当前代码放在 LTR-DQN-main/code/ 下运行
CODE_DIR = Path(__file__).resolve().parent

dapan_code = "0060"
test_batch = 123
train_year = 3

train_start = 20181206
train_end = 20211206
test_start = 20211207
test_end = 20230303

shouxufei = 0.0003
yinhaushui = 0.001
initial_capital = 1_000_000

SAMPLING_RATES = [0.5, 0.6, 0.7, 0.8, 0.9]
SAMPLING_LABELS = ["50%", "60%", "70%", "80%", "90%"]

# True：优先用 GPU，与原始代码 tree_method='gpu_hist' 保持一致；
# 如果你的环境没有 GPU，代码会自动退回 tree_method='hist'。
USE_GPU = True

# 如果想临时少跑一点用于调试，可以改成 5 或 10；
# 正式画图保持 None，表示使用 seed_summary.csv 里全部 seed。
MAX_SEEDS_PER_CELL = None

# 默认不打印每个 seed 的结果。
VERBOSE = False


# ============================================================
# 路径
# ============================================================

data_path = CODE_DIR / f"data/{dapan_code}merge_open_close_final.csv"
select_path = CODE_DIR / f"temp/oc/batch{test_batch}/meiri_xuanze.csv"
seed_path = CODE_DIR / "temp/seed_summary.csv"


# ============================================================
# 列名
# ============================================================

col_name = [
    "stock_code", "page", "advance_reaction", "star_analyst", "title_len", "num_sentence",
    "avg_sentence_len", "sd_sentence_len",
    "num_authors", "analyst_coverage", "rm_rf", "smb", "hml", "rmw", "cma", "broker_size", "listed",
    "prior_performance_avg", "prior_performance_sd", "broker_status", "qid_date", "real_return",
    "ind_1", "ind_2", "ind_3", "ind_4", "ind_5", "ind_6",
    "close", "pclose", "volume"
]

Xcol_name = [
    "page", "advance_reaction", "star_analyst", "title_len", "num_sentence",
    "avg_sentence_len", "sd_sentence_len",
    "num_authors", "analyst_coverage", "rm_rf", "smb", "hml", "rmw", "cma", "broker_size", "listed",
    "prior_performance_avg", "prior_performance_sd", "broker_status", "qid_date",
    "ind_1", "ind_2", "ind_3", "ind_4", "ind_5", "ind_6"
]

Ycol_name = ["real_return"]


# ============================================================
# 工具函数
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def to_int_date_series(s):
    raw = s.astype(str).str.strip()
    raw = raw.str.replace(r"\.0$", "", regex=True)
    raw = raw.str.replace("-", "", regex=False)
    raw = raw.str.replace("/", "", regex=False)
    return pd.to_numeric(raw, errors="coerce").astype("Int64")


def sample_or_keep(group, sampling_rate, seed):
    if len(group) > 1 and sampling_rate < 1.0:
        return group.sample(frac=sampling_rate, random_state=seed)
    return group


def transform_column(column):
    return column.applymap(lambda x: x)


def safe_normalize_features(yuan_all_df):
    features_to_normalize = yuan_all_df.drop(columns=["qid_date", "stock_code"])
    min_vals = features_to_normalize.min()
    max_vals = features_to_normalize.max()

    denom = max_vals - min_vals
    denom = denom.replace(0, np.nan)

    normalized_features = 2 * (features_to_normalize - min_vals) / denom - 1
    normalized_features = normalized_features.replace([np.inf, -np.inf], np.nan).fillna(0)

    return yuan_all_df[["qid_date"]].join(normalized_features)


def get_group_sizes(df):
    return (
        df.groupby("qid_date")
        .size()
        .to_frame("size")
        .sort_values(by="qid_date")["size"]
        .to_numpy()
        .tolist()
    )


def trade_one_day(top_stocks, current_capital):
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

        sell_value = (
            shares_bought * close
            - shares_bought * close * (shouxufei + yinhaushui)
            + yu
        )

        day_total_profit += sell_value

    day_return = (day_total_profit - current_capital) / current_capital
    return day_total_profit, day_return


def annualized_return_from_curve(results_df):
    if len(results_df) == 0:
        return np.nan

    trading_days = results_df.shape[0]
    return (results_df.iloc[-1]["total_profit"] / initial_capital) ** (242 / trading_days) - 1


# ============================================================
# seed 读取
# ============================================================

def load_seed_table():
    if not seed_path.exists():
        raise FileNotFoundError(f"找不到 seed_summary.csv：{seed_path}")

    return pd.read_csv(seed_path, index_col=0)


def fallback_seed_list(seed_df):
    values = []

    for idx in seed_df.index:
        if str(idx).startswith("T6M"):
            values.extend(seed_df.loc[idx].dropna().tolist())

    values = [int(x) for x in values if pd.notna(x)]
    values = sorted(list(dict.fromkeys(values)))

    if len(values) == 0:
        values = [1795]

    return values


def get_seed_list(seed_df, sampling_rate, suffix):
    """
    suffix:
    - "1"：LambdaRank，即 T6M5_1/T6M6_1/...
    - "2"：LambdaMART 与 LTR-DQN，即 T6M5_2/T6M6_2/...
    """
    rate_num = int(round(sampling_rate * 10))
    candidates = [
        f"T6M{rate_num}_{suffix}",
        f"T6M{rate_num}_{suffix}.py",
    ]

    for row_name in candidates:
        if row_name in seed_df.index:
            seeds = seed_df.loc[row_name].dropna().tolist()
            seeds = [int(x) for x in seeds if pd.notna(x)]

            if MAX_SEEDS_PER_CELL is not None:
                seeds = seeds[:MAX_SEEDS_PER_CELL]

            return seeds

    # 如果 seed_summary 里没有对应行，则退回到全部 T6M 行的 seed 池。
    seeds = fallback_seed_list(seed_df)

    if MAX_SEEDS_PER_CELL is not None:
        seeds = seeds[:MAX_SEEDS_PER_CELL]

    return seeds


# ============================================================
# 模型与回测
# ============================================================

def prepare_sampled_data(all_df, sampling_rate, seed):
    set_seed(seed)

    if sampling_rate < 1.0:
        yuan_all_df = (
            all_df.groupby("qid_date")
            .apply(lambda g: sample_or_keep(g, sampling_rate, seed))
            .reset_index(drop=True)
        )
    else:
        yuan_all_df = copy.deepcopy(all_df)

    yuan_all_df[Ycol_name] = transform_column(yuan_all_df[Ycol_name])

    df_normalized = safe_normalize_features(yuan_all_df)

    train_df = df_normalized[
        (train_start <= df_normalized["qid_date"]) &
        (df_normalized["qid_date"] <= train_end)
    ]

    yuan_train_df = yuan_all_df[
        (train_start <= yuan_all_df["qid_date"]) &
        (yuan_all_df["qid_date"] <= train_end)
    ]

    test_df = df_normalized[
        (test_start <= df_normalized["qid_date"]) &
        (df_normalized["qid_date"] <= test_end)
    ]

    yuan_test_df = yuan_all_df[
        (test_start <= yuan_all_df["qid_date"]) &
        (yuan_all_df["qid_date"] <= test_end)
    ]

    train_groups = get_group_sizes(train_df)
    test_groups = get_group_sizes(test_df)

    X_train = train_df[Xcol_name]
    Y_train = train_df[Ycol_name]
    X_test = test_df[Xcol_name]

    return X_train, Y_train, X_test, yuan_test_df, train_groups, test_groups


def make_model(model_name, seed, tree_method):
    """
    model_name:
    - LambdaRank: rank:pairwise, learning_rate=0.01
    - LambdaMART: rank:map, learning_rate=0.001, max_depth=5, n_estimators=1000
    """
    if model_name == "LambdaRank":
        return xgb.XGBRanker(
            tree_method=tree_method,
            lambdarank_num_pair_per_sample=8,
            booster="gbtree",
            eval_metric="ndcg",
            objective="rank:pairwise",
            learning_rate=0.01,
            lambdarank_pair_method="topk",
            random_state=seed,
        )

    if model_name == "LambdaMART":
        return xgb.XGBRanker(
            tree_method=tree_method,
            lambdarank_num_pair_per_sample=8,
            booster="gbtree",
            eval_metric="ndcg",
            objective="rank:map",
            learning_rate=0.001,
            max_depth=5,
            n_estimators=1000,
            lambdarank_pair_method="topk",
            random_state=seed,
        )

    raise ValueError(f"Unknown model_name: {model_name}")


def fit_ranker_with_fallback(model_name, seed, x_train, y_train, train_groups):
    tree_method = "gpu_hist" if USE_GPU else "hist"
    model = make_model(model_name, seed, tree_method)

    try:
        return model.fit(x_train, y_train, group=train_groups)
    except Exception as e:
        if USE_GPU:
            # 无 GPU 或 xgboost 版本不支持 gpu_hist 时自动回退
            model = make_model(model_name, seed, "hist")
            return model.fit(x_train, y_train, group=train_groups)
        raise e


def train_predict_temp(all_df, sampling_rate, seed, model_name):
    X_train, Y_train, X_test, yuan_test_df, train_groups, test_groups = prepare_sampled_data(
        all_df=all_df,
        sampling_rate=sampling_rate,
        seed=seed
    )

    x_train = X_train.drop(["qid_date"], axis=1)
    y_train = Y_train.to_numpy()

    ranker = fit_ranker_with_fallback(
        model_name=model_name,
        seed=seed,
        x_train=x_train,
        y_train=y_train,
        train_groups=train_groups
    )

    x_test = X_test.drop(["qid_date"], axis=1)

    predictions = []
    start_idx = 0

    for group_size in test_groups:
        group_data = x_test.iloc[start_idx:start_idx + group_size]
        group_predictions = ranker.predict(group_data)
        predictions.extend(group_predictions)
        start_idx += group_size

    temp = yuan_test_df[["qid_date", "stock_code", "real_return", "close", "pclose", "volume"]].copy()
    temp.loc[:, "prediction"] = copy.deepcopy(predictions)

    return temp


def backtest_top4(temp):
    df = temp[
        (test_start <= temp["qid_date"]) &
        (temp["qid_date"] <= test_end)
    ].copy()

    current_capital = initial_capital
    daily_results = []

    for qid_date, group in df.groupby("qid_date"):
        top_stocks = group.nlargest(min(4, len(group)), "prediction")

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

    return annualized_return_from_curve(pd.DataFrame(daily_results))


def backtest_ltr_dqn(temp, select_map):
    df = temp[
        (test_start <= temp["qid_date"]) &
        (temp["qid_date"] <= test_end)
    ].copy()

    current_capital = initial_capital
    daily_results = []

    for qid_date, group in df.groupby("qid_date"):
        top_n = int(select_map.get(int(qid_date), 0))

        if top_n <= 0:
            top_stocks = group.iloc[0:0]
        else:
            top_stocks = group.nlargest(min(top_n, len(group)), "prediction")

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

    return annualized_return_from_curve(pd.DataFrame(daily_results))


def load_all_data():
    all_df = pd.read_csv(data_path, usecols=col_name)

    all_df["qid_date"] = to_int_date_series(all_df["qid_date"])
    all_df = all_df.dropna(subset=["qid_date"]).copy()
    all_df["qid_date"] = all_df["qid_date"].astype(int)

    all_df = all_df[
        (train_start <= all_df["qid_date"]) &
        (all_df["qid_date"] <= test_end)
    ].copy()

    return all_df


def load_select_map():
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


def collect_boxplot_data():
    all_df = load_all_data()
    select_map = load_select_map()
    seed_df = load_seed_table()

    rows = []

    for sampling_rate, sampling_label in zip(SAMPLING_RATES, SAMPLING_LABELS):

        # LambdaRank：T6M*_1 逻辑
        seeds_rank = get_seed_list(seed_df, sampling_rate, suffix="1")

        for seed in seeds_rank:
            temp = train_predict_temp(
                all_df=all_df,
                sampling_rate=sampling_rate,
                seed=seed,
                model_name="LambdaRank"
            )

            arr = backtest_top4(temp)

            rows.append({
                "Sampling Rate": sampling_label,
                "Model": "LambdaRank",
                "Annualized Return": arr
            })

            if VERBOSE:
                print(sampling_label, "LambdaRank", seed, arr)

        # LambdaMART 与 LTR-DQN：T6M*_2 逻辑
        seeds_mart = get_seed_list(seed_df, sampling_rate, suffix="2")

        for seed in seeds_mart:
            temp = train_predict_temp(
                all_df=all_df,
                sampling_rate=sampling_rate,
                seed=seed,
                model_name="LambdaMART"
            )

            arr_mart = backtest_top4(temp)
            arr_dqn = backtest_ltr_dqn(temp, select_map)

            rows.append({
                "Sampling Rate": sampling_label,
                "Model": "LambdaMART",
                "Annualized Return": arr_mart
            })

            rows.append({
                "Sampling Rate": sampling_label,
                "Model": "LTR-DQN",
                "Annualized Return": arr_dqn
            })

            if VERBOSE:
                print(sampling_label, "LambdaMART", seed, arr_mart)
                print(sampling_label, "LTR-DQN", seed, arr_dqn)

    return pd.DataFrame(rows)


# ============================================================
# 画图
# ============================================================

def plot_main_sampling_boxplot(df):
    models = ["LambdaRank", "LambdaMART", "LTR-DQN"]
    colors = {
        "LambdaRank": "#22A7CF",
        "LambdaMART": "#ED7D31",
        "LTR-DQN": "#BFBFBF",
    }

    base_positions = np.arange(len(SAMPLING_LABELS)) + 1
    offsets = {
        "LambdaRank": -0.24,
        "LambdaMART": 0.0,
        "LTR-DQN": 0.24,
    }

    fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=120)

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for model in models:
        data = []

        for label in SAMPLING_LABELS:
            values = df[
                (df["Sampling Rate"] == label) &
                (df["Model"] == model)
            ]["Annualized Return"].dropna().values

            data.append(values)

        bp = ax.boxplot(
            data,
            positions=base_positions + offsets[model],
            widths=0.22,
            patch_artist=True,
            showfliers=True,
            whis=1.5,
            medianprops={
                "color": "black",
                "linewidth": 0.9,
            },
            boxprops={
                "linewidth": 0.8,
                "color": "#555555",
            },
            whiskerprops={
                "linewidth": 0.8,
                "color": "#555555",
            },
            capprops={
                "linewidth": 0.8,
                "color": "#555555",
            },
            flierprops={
                "marker": "d",
                "markersize": 2.2,
                "markerfacecolor": "#555555",
                "markeredgecolor": "#555555",
                "alpha": 0.9,
            },
        )

        for patch in bp["boxes"]:
            patch.set_facecolor(colors[model])

    ax.set_xticks(base_positions)
    ax.set_xticklabels(SAMPLING_LABELS, fontsize=8, fontweight="bold")

    ax.set_xlabel("Sampling Rate", fontsize=9, fontweight="bold")
    ax.set_ylabel("Annualized Return", fontsize=9, fontweight="bold")

    ax.tick_params(axis="y", labelsize=8)

    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.grid(axis="x", visible=False)

    for spine in ax.spines.values():
        spine.set_color("#A0A0A0")
        spine.set_linewidth(0.8)

    legend_handles = [
        mpatches.Patch(facecolor=colors[m], edgecolor="#555555", label=m)
        for m in models
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=7,
        frameon=True,
        framealpha=1,
        edgecolor="#D9D9D9"
    )

    # 参考图主板纵轴范围
    ax.set_ylim(-3, 8.5)
    ax.set_yticks(np.arange(-2, 9, 2))

    plt.tight_layout()
    plt.show()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    box_df = collect_boxplot_data()
    plot_main_sampling_boxplot(box_df)
