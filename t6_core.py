"""Sampling-rate robustness experiment used for paper Table 6.

This module is a consolidated form of the original Mbox.py/Cbox.py logic.
The market-specific values are configuration data; sampling, ranking and
backtest operations intentionally follow the source scripts.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


CODE_DIR = Path(__file__).resolve().parent
T6_REPLICATIONS = 20
SAMPLING_RATES = [0.5, 0.6, 0.7, 0.8, 0.9]
SAMPLING_LABELS = ["50%", "60%", "70%", "80%", "90%"]
TRAIN_START = 20181206
TRAIN_END = 20211206
TEST_START = 20211207
TEST_END = 20230303
FEES = 0.0003
STAMP_TAX = 0.001
INITIAL_CAPITAL = 1_000_000

MARKET_CONFIG = {
    "Main": {
        "code": "0060", "prefix": "T6M", "action_column": "60",
        "rank_learning_rate": 0.01, "mart_objective": "rank:map",
        "mart_learning_rate": 0.001, "mart_depth": 5,
    },
    "ChiNext": {
        "code": "3068", "prefix": "T6C", "action_column": "3068",
        "rank_learning_rate": 0.1, "mart_objective": "rank:ndcg",
        "mart_learning_rate": 0.1, "mart_depth": 6,
    },
}

COL_NAME = [
    "stock_code", "page", "advance_reaction", "star_analyst", "title_len", "num_sentence",
    "avg_sentence_len", "sd_sentence_len", "num_authors", "analyst_coverage", "rm_rf", "smb",
    "hml", "rmw", "cma", "broker_size", "listed", "prior_performance_avg",
    "prior_performance_sd", "broker_status", "qid_date", "real_return", "ind_1", "ind_2",
    "ind_3", "ind_4", "ind_5", "ind_6", "close", "pclose", "volume",
]
XCOL_NAME = [
    "page", "advance_reaction", "star_analyst", "title_len", "num_sentence", "avg_sentence_len",
    "sd_sentence_len", "num_authors", "analyst_coverage", "rm_rf", "smb", "hml", "rmw", "cma",
    "broker_size", "listed", "prior_performance_avg", "prior_performance_sd", "broker_status",
    "qid_date", "ind_1", "ind_2", "ind_3", "ind_4", "ind_5", "ind_6",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def to_int_date_series(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    raw = raw.str.replace(r"\.0$", "", regex=True).str.replace("-", "", regex=False)
    raw = raw.str.replace("/", "", regex=False)
    return pd.to_numeric(raw, errors="coerce").astype("Int64")


def sample_or_keep(group: pd.DataFrame, rate: float, seed: int) -> pd.DataFrame:
    if len(group) > 1 and rate < 1.0:
        return group.sample(frac=rate, random_state=seed)
    return group


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.drop(columns=["qid_date", "stock_code"])
    minimum = features.min()
    maximum = features.max()
    denominator = (maximum - minimum).replace(0, np.nan)
    normalized = 2 * (features - minimum) / denominator - 1
    normalized = normalized.replace([np.inf, -np.inf], np.nan).fillna(0)
    return frame[["qid_date"]].join(normalized)


def group_sizes(frame: pd.DataFrame) -> list[int]:
    return frame.groupby("qid_date").size().sort_index().tolist()


def trade_one_day(top_stocks: pd.DataFrame, capital: float) -> tuple[float, float]:
    if len(top_stocks) == 0:
        return capital, 0.0
    per_stock = capital / len(top_stocks)
    total = 0.0
    for _, row in top_stocks.iterrows():
        pclose, close = row["pclose"], row["close"]
        if pd.isna(pclose) or pd.isna(close) or pclose <= 0:
            total += per_stock
            continue
        lots = int(per_stock / (100 * pclose))
        fee = lots * 100 * pclose * FEES
        shares = int((per_stock - fee) / (100 * pclose)) * 100
        cash = per_stock - fee - shares * pclose
        if shares == 0:
            cash = per_stock
        total += shares * close - shares * close * (FEES + STAMP_TAX) + cash
    return total, (total - capital) / capital


def annualized(curve: pd.DataFrame) -> float:
    if curve.empty:
        return float("nan")
    return float((curve.iloc[-1]["total_profit"] / INITIAL_CAPITAL) ** (242 / len(curve)) - 1)


def load_seed_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"T6 seed summary not found: {path}")
    return pd.read_csv(path, index_col=0)


def load_dqn_seed_table(path: Path) -> pd.DataFrame:
    """Load the independent DQN seed ledger.

    ``seed_summary.csv`` belongs to the ranker sampling experiment.  DQN
    training/evaluation randomness is recorded in its own ledger because the
    current DQN implementation no longer shares the original execution path.
    """
    if not path.is_file():
        raise FileNotFoundError(f"DQN seed summary not found: {path}")
    return pd.read_csv(path, index_col=0)


def get_seed_list(seed_df: pd.DataFrame, market: str, rate: float, suffix: str, max_seeds: int | None) -> list[int]:
    config = MARKET_CONFIG[market]
    key = f"{config['prefix']}{int(round(rate * 10))}_{suffix}"
    if key not in seed_df.index and f"{key}.py" in seed_df.index:
        key = f"{key}.py"
    if key in seed_df.index:
        values = seed_df.loc[key].dropna().tolist()
    else:
        values = []
        for index in seed_df.index:
            if str(index).startswith(config["prefix"]):
                values.extend(seed_df.loc[index].dropna().tolist())
        values = sorted(set(values))
    values = [int(value) for value in values if pd.notna(value)]
    if max_seeds is not None:
        values = values[:max_seeds]
    return values or [1795]


def get_dqn_seed_list(seed_df: pd.DataFrame, market: str, rate: float, max_seeds: int | None) -> list[int]:
    """Return the recorded DQN seeds for a T6 market/rate cell."""
    config = MARKET_CONFIG[market]
    key = f"{config['prefix']}{int(round(rate * 10))}_DQN"
    if key not in seed_df.index:
        raise KeyError(f"Missing DQN seed row {key} in {seed_df.index.name or 'seed summary'}")
    values = [int(value) for value in seed_df.loc[key].dropna().tolist()]
    if max_seeds is not None:
        values = values[:max_seeds]
    return values or [1795]


def load_all_data(market: str, data_dir: Path) -> pd.DataFrame:
    code = MARKET_CONFIG[market]["code"]
    path = data_dir / f"{code}merge_open_close_final.csv"
    frame = pd.read_csv(path, usecols=COL_NAME)
    frame["qid_date"] = to_int_date_series(frame["qid_date"])
    frame = frame.dropna(subset=["qid_date"]).copy()
    frame["qid_date"] = frame["qid_date"].astype(int)
    return frame[(frame.qid_date >= TRAIN_START) & (frame.qid_date <= TEST_END)].copy()


def make_ranker(market: str, model_name: str, seed: int, tree_method: str) -> xgb.XGBRanker:
    cfg = MARKET_CONFIG[market]
    if model_name == "LambdaRank":
        return xgb.XGBRanker(
            tree_method=tree_method, lambdarank_num_pair_per_sample=8,
            booster="gbtree", eval_metric="ndcg", objective="rank:pairwise",
            learning_rate=cfg["rank_learning_rate"], lambdarank_pair_method="topk",
            random_state=seed, n_jobs=1,
        )
    if model_name == "LambdaMART":
        return xgb.XGBRanker(
            tree_method=tree_method, lambdarank_num_pair_per_sample=8,
            booster="gbtree", eval_metric="ndcg", objective=cfg["mart_objective"],
            learning_rate=cfg["mart_learning_rate"], max_depth=cfg["mart_depth"],
            n_estimators=1000, lambdarank_pair_method="topk", random_state=seed,
            n_jobs=1,
        )
    raise ValueError(f"Unknown T6 model: {model_name}")


def train_predict_temp(
    all_df: pd.DataFrame, market: str, rate: float, seed: int, model_name: str,
    use_gpu: bool = True, require_gpu: bool = False,
) -> pd.DataFrame:
    set_seed(seed)
    if rate < 1.0:
        sampled = all_df.groupby("qid_date").apply(
            lambda group: sample_or_keep(group, rate, seed)
        ).reset_index(drop=True)
    else:
        sampled = copy.deepcopy(all_df)
    normalized = normalize(sampled)
    train = normalized[(normalized.qid_date >= TRAIN_START) & (normalized.qid_date <= TRAIN_END)]
    test = normalized[(normalized.qid_date >= TEST_START) & (normalized.qid_date <= TEST_END)]
    raw_test = sampled[(sampled.qid_date >= TEST_START) & (sampled.qid_date <= TEST_END)]
    x_train = train[XCOL_NAME].drop(columns=["qid_date"])
    y_train = train[["real_return"]].to_numpy()
    methods = ["gpu_hist"] if require_gpu else (["gpu_hist", "hist"] if use_gpu else ["hist"])
    last_error = None
    for method in methods:
        try:
            ranker = make_ranker(market, model_name, seed, method)
            ranker.fit(x_train, y_train, group=group_sizes(train))
            break
        except Exception as exc:
            last_error = exc
    else:
        raise last_error
    x_test = test[XCOL_NAME].drop(columns=["qid_date"])
    predictions = []
    start = 0
    for size in group_sizes(test):
        predictions.extend(ranker.predict(x_test.iloc[start:start + size]))
        start += size
    result = raw_test[["qid_date", "stock_code", "real_return", "close", "pclose", "volume"]].copy()
    result["prediction"] = copy.deepcopy(predictions)
    return result


def backtest_top4(temp: pd.DataFrame) -> float:
    capital = INITIAL_CAPITAL
    rows = []
    for date, group in temp.groupby("qid_date"):
        chosen = group.nlargest(min(4, len(group)), "prediction")
        total, day_return = trade_one_day(chosen, capital)
        capital = total
        rows.append({"qid_date": date, "total_profit": total, "day_return": day_return})
    return annualized(pd.DataFrame(rows))


def backtest_ltr_dqn(temp: pd.DataFrame, select_map: dict[int, int]) -> float:
    capital = INITIAL_CAPITAL
    rows = []
    for date, group in temp.groupby("qid_date"):
        top_n = int(select_map.get(int(date), 0))
        chosen = group.iloc[0:0] if top_n <= 0 else group.nlargest(min(top_n, len(group)), "prediction")
        total, day_return = trade_one_day(chosen, capital)
        capital = total
        rows.append({"qid_date": date, "total_profit": total, "day_return": day_return})
    return annualized(pd.DataFrame(rows))


def load_select_map(path: Path, market: str) -> dict[int, int]:
    column = MARKET_CONFIG[market]["action_column"]
    frame = pd.read_csv(path, usecols=["qid_date", column])
    frame["qid_date"] = to_int_date_series(frame["qid_date"])
    frame = frame.dropna(subset=["qid_date"]).copy()
    frame["qid_date"] = frame["qid_date"].astype(int)
    frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)
    return dict(zip(frame.qid_date, frame[column]))


def run_sampling(
    data_dir: Path, seed_path: Path, select_map_path: Path,
    output_path: Path, markets: list[str] | None = None,
    max_seeds: int | None = None, use_gpu: bool = True,
    resume: bool = True, include_full_rate: bool = True,
    dqn_seed_path: Path | None = None, require_gpu: bool = False,
    full_rate_path: Path | None = None,
) -> pd.DataFrame:
    markets = ["Main", "ChiNext"] if markets is None else markets
    seed_df = load_seed_table(seed_path)
    dqn_seed_path = dqn_seed_path or (CODE_DIR / "temp" / "dqn_seed_summary.csv")
    dqn_seed_df = load_dqn_seed_table(dqn_seed_path)
    existing = pd.read_csv(output_path) if resume and output_path.is_file() else pd.DataFrame()
    done = set(zip(existing.get("market", []), existing.get("sampling_rate", []), existing.get("model", []), existing.get("seed", [])))
    rows = existing.to_dict(orient="records")
    # Upgrade raw files written before the independent DQN ledger existed.
    # No ARR is recomputed; this only adds the missing provenance field.
    upgraded = False
    for record in rows:
        if record.get("model") != "LTR-DQN" or pd.notna(record.get("dqn_seed")):
            continue
        market = str(record["market"])
        rate = float(record["sampling_rate"])
        rank_seed = int(record["seed"])
        lookup_rate = 0.9 if rate == 1.0 else rate
        rank_seeds = get_seed_list(seed_df, market, lookup_rate, "2", None)
        try:
            position = rank_seeds.index(rank_seed)
        except ValueError:
            position = 0
        dqn_seeds = get_dqn_seed_list(dqn_seed_df, market, lookup_rate, None)
        record["dqn_seed"] = dqn_seeds[position % len(dqn_seeds)]
        upgraded = True
    if upgraded:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(
            output_path, index=False, encoding="utf-8-sig", float_format="%.17g"
        )
    if include_full_rate:
        if full_rate_path is None or not full_rate_path.is_file():
            raise FileNotFoundError(
                "T6 100% rows must come from the same run's T4 reference: "
                f"{full_rate_path}"
            )
        full_reference = pd.read_csv(full_rate_path, float_precision="round_trip")
        required = {"market", "model", "ARR", "seed"}
        missing = sorted(required - set(full_reference.columns))
        if missing:
            raise ValueError(f"T4 reference is missing columns: {missing}")
        full_reference = full_reference[
            full_reference.market.isin(markets)
            & full_reference.model.isin(("LambdaRank", "LambdaMART", "LTR-DQN"))
        ].copy()
        counts = full_reference.groupby(["market", "model"]).size()
        if len(counts) != len(markets) * 3 or not (counts == 1).all():
            raise ValueError(f"T4 reference must contain one row per market/model:\n{counts}")
        for record in full_reference.to_dict(orient="records"):
            rows.append({
                "market": record["market"], "sampling_rate": 1.0,
                "sampling_label": "100%", "model": record["model"],
                "seed": int(record["seed"]), "ARR": float(record["ARR"]),
                "dqn_seed": record.get("dqn_seed", np.nan),
                "source": "same_run_T4",
            })

    for market in markets:
        all_df = load_all_data(market, data_dir)
        select_map = load_select_map(select_map_path, market)
        for rate, label in zip(SAMPLING_RATES, SAMPLING_LABELS):
            for suffix, model_name in (("1", "LambdaRank"), ("2", "LambdaMART")):
                seeds = get_seed_list(seed_df, market, rate, suffix, max_seeds)
                dqn_seeds = get_dqn_seed_list(dqn_seed_df, market, rate, max_seeds)
                for position, seed in enumerate(seeds):
                    key = (market, rate, model_name, seed)
                    if key in done:
                        continue
                    temp = train_predict_temp(
                        all_df, market, rate, seed, model_name,
                        use_gpu=use_gpu, require_gpu=require_gpu,
                    )
                    arr = backtest_top4(temp)
                    rows.append({"market": market, "sampling_rate": rate, "sampling_label": label, "model": model_name, "seed": seed, "ARR": arr})
                    if model_name == "LambdaMART":
                        dqn_seed = dqn_seeds[position % len(dqn_seeds)]
                        rows.append({"market": market, "sampling_rate": rate, "sampling_label": label, "model": "LTR-DQN", "seed": seed, "dqn_seed": dqn_seed, "ARR": backtest_ltr_dqn(temp, select_map)})
                    done.add(key)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(rows).to_csv(
                        output_path,
                        index=False,
                        encoding="utf-8-sig",
                        float_format="%.17g",
                    )
    return pd.DataFrame(rows)


def summarize_sampling(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rates = [(1.0, "100%"), (0.9, "90%"), (0.8, "80%"), (0.7, "70%"), (0.6, "60%"), (0.5, "50%")]
    markets = [market for market in ("Main", "ChiNext") if market in set(raw.get("market", []))]
    for market in markets:
        for model in ("LambdaRank", "LambdaMART", "LTR-DQN"):
            for statistic in ("Mean", "Std."):
                row = {"market": market, "model": model, "statistic": statistic}
                for rate, label in rates:
                    values = raw[(raw.market == market) & (raw.model == model) & (raw.sampling_rate == rate)]["ARR"].dropna()
                    if statistic == "Mean":
                        row[label] = float(values.mean()) if len(values) else np.nan
                    elif rate == 1.0:
                        row[label] = "-"
                    else:
                        row[label] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
                rows.append(row)
    return pd.DataFrame(rows)
