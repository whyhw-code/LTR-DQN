"""Reproduce the empirical appendix figures (Figures C1-C5).

This is intentionally separate from ``Fig_main.py``.  Every plotted value is
derived from the current data/artifacts, and the underlying CSVs plus SHA-256
manifest are retained next to the PNG files.

Figure C2 needs a brokerage identifier.  The distributed clean data do not
contain brokerage names/IDs, so the default run uses ``broker_size`` as an
explicitly labelled proxy grouping.  Pass ``--broker_file`` and
``--broker_column`` when the original brokerage identifier is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_core import (
    CODE_DIR,
    DATA_DIR,
    DQN_RANKER,
    FEES,
    MARKETS,
    STAMP_TAX,
    TEST_END,
    TEST_START,
    artifact_dir,
    runtime_versions,
    sha256,
    validate_runtime,
)
from t6_core import T6_REPLICATIONS


MARKET_ORDER = ("Main", "ChiNext")
MARKET_TITLES = {"Main": "Main board market", "ChiNext": "ChiNext market"}
INDEX_NAMES = {"Main": "CSI 300 Index", "ChiNext": "ChiNext Index"}
MARKET_CODES = {"Main": "0060", "ChiNext": "3068"}
INITIAL_CAPITAL = 1_000_000.0
LONG_START = 20171206
MODELS = ("LambdaRank", "LambdaMART", "LTR-DQN")
RATES = (0.5, 0.6, 0.7, 0.8, 0.9)
COLORS = {
    "Main": "#2F5597",
    "ChiNext": "#D28E00",
    "index": "#777777",
    "Baseline portfolio": "#4472C4",
    "No ESG": "#C00000",
    "NS 25%": "#E6A700",
    "NS 50%": "#ED7D31",
    "PI 25%": "#70AD47",
    "PI 50%": "#264478",
    "LambdaRank": "#2AA6C8",
    "LambdaMART": "#ED7D31",
    "LTR-DQN": "#A6A6A6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute and export appendix Figures C1-C5"
    )
    parser.add_argument(
        "--run_dir", type=Path, default=CODE_DIR,
        help="Artifacts created by train.py; default uses code_1_final/temp and model",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=CODE_DIR / "results" / "appendix_figures",
    )
    parser.add_argument(
        "--figures", default="C1,C2,C3,C4,C5",
        help="Comma-separated subset of C1,C2,C3,C4,C5",
    )
    parser.add_argument(
        "--t6_csv", type=Path, default=CODE_DIR / "temp" / "t6_runs" / "t6_raw.csv",
        help=f"{T6_REPLICATIONS}-replication raw results used by Figure C4",
    )
    parser.add_argument(
        "--broker_file", type=Path, default=None,
        help="Optional report-level CSV containing a true brokerage identifier",
    )
    parser.add_argument(
        "--broker_column", default=None,
        help="Brokerage identifier column; default is broker_id when present, else broker_size proxy",
    )
    parser.add_argument(
        "--min_broker_reports", type=int, default=100,
        help="Minimum report observations required for one Figure C2 group",
    )
    parser.add_argument("--force", action="store_true", help="Ignore cached appendix data")
    return parser.parse_args()


def selected_figures(value: str) -> list[str]:
    result = []
    for item in value.split(","):
        label = item.strip().upper()
        if label and not label.startswith("C"):
            label = f"C{label}"
        if label and label not in result:
            result.append(label)
    invalid = sorted(set(result) - {"C1", "C2", "C3", "C4", "C5"})
    if not result or invalid:
        raise ValueError(f"figures must be a subset of C1,C2,C3,C4,C5; invalid={invalid}")
    return result


def require_file(path: Path, purpose: str) -> Path:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{purpose} not found: {path}")
    return path


def digest_text(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_signature(paths: Iterable[Path]) -> str:
    resolved = [require_file(path, "appendix figure source") for path in paths]
    return digest_text({str(path): sha256(path) for path in resolved})


def implementation_paths() -> list[Path]:
    return [CODE_DIR / "Appendix_Fig_main.py", CODE_DIR / "experiment_core.py"]


def cached_csv(path: Path, source_signature: str, force: bool) -> pd.DataFrame | None:
    if not path.is_file() or force:
        return None
    frame = pd.read_csv(path)
    if frame.empty or "source_signature" not in frame.columns:
        return None
    if not (frame.source_signature.astype(str) == source_signature).all():
        print(f"Ignoring appendix cache generated from different sources: {path}")
        return None
    print(f"Using cached appendix data: {path}")
    return frame


def save_csv(frame: pd.DataFrame, path: Path, source_signature: str) -> pd.DataFrame:
    result = frame.copy()
    result["source_signature"] = source_signature
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, encoding="utf-8-sig")
    return result


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path.resolve()}")


def style_axis(ax, grid_axis: str = "y") -> None:
    ax.set_facecolor("white")
    ax.grid(axis=grid_axis, color="#D9D9D9", linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#A6A6A6")
        spine.set_linewidth(0.8)


def to_int_dates(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_numeric(series.dt.strftime("%Y%m%d"), errors="coerce")
    raw = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    numeric = pd.to_numeric(raw, errors="coerce")
    serial = numeric.dropna()
    if not serial.empty and serial.between(30000, 60000).mean() > 0.8:
        dates = pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
        return pd.to_numeric(dates.dt.strftime("%Y%m%d"), errors="coerce")
    compact = raw.str.replace("-", "", regex=False).str.replace("/", "", regex=False)
    eight_digit = compact.str.fullmatch(r"\d{8}", na=False)
    result = pd.Series(np.nan, index=series.index, dtype="float64")
    result.loc[eight_digit] = pd.to_numeric(compact.loc[eight_digit], errors="coerce")
    remaining = ~eight_digit
    parsed = pd.to_datetime(raw.loc[remaining], errors="coerce")
    result.loc[remaining] = pd.to_numeric(parsed.dt.strftime("%Y%m%d"), errors="coerce")
    return result


def as_datetime(series: pd.Series) -> pd.Series:
    raw = pd.to_numeric(series, errors="coerce").astype("Int64").astype(str)
    return pd.to_datetime(raw, format="%Y%m%d", errors="coerce")


def normalize_funds(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[["qid_date", "funds"]].dropna().sort_values("qid_date", kind="mergesort")
    if result.empty:
        return result
    first = float(result.funds.iloc[0])
    if first != 0:
        result["funds"] = result.funds / first * INITIAL_CAPITAL
    return result


def trade_selected(
    selected: pd.DataFrame,
    capital: float,
    commission: float = FEES,
    stamp_tax: float = STAMP_TAX,
) -> tuple[float, int, int]:
    if selected.empty:
        return capital, 0, 0
    allocation = capital / len(selected)
    total = 0.0
    wins = 0
    traded = 0
    for _, row in selected.iterrows():
        pclose = pd.to_numeric(row.get("pclose"), errors="coerce")
        close = pd.to_numeric(row.get("close"), errors="coerce")
        if pd.isna(pclose) or pd.isna(close) or pclose <= 0:
            total += allocation
            continue
        lots = int(allocation / (100 * pclose))
        purchase_fee = lots * 100 * pclose * commission
        shares = int((allocation - purchase_fee) / (100 * pclose)) * 100
        cash = allocation - purchase_fee - shares * pclose
        sell = shares * close - shares * close * (commission + stamp_tax) + cash
        total += sell
        traded += 1
        wins += int(sell > allocation)
    return total, wins, traded


def backtest(
    frame: pd.DataFrame,
    *,
    selection: str,
    actions: dict[int, int] | None = None,
    commission: float = FEES,
    stamp_tax: float = STAMP_TAX,
) -> tuple[pd.DataFrame, int, int]:
    capital = INITIAL_CAPITAL
    rows = []
    total_wins = total_trades = 0
    ordered = frame.sort_values("qid_date", kind="mergesort")
    for date, group in ordered.groupby("qid_date", sort=True):
        if selection == "all":
            selected = group
        elif selection == "top4":
            selected = group.nlargest(min(4, len(group)), "prediction")
        elif selection == "actions":
            top_n = int((actions or {}).get(int(date), 0))
            selected = (
                group.iloc[0:0]
                if top_n <= 0
                else group.nlargest(min(top_n, len(group)), "prediction")
            )
        else:
            raise ValueError(f"Unknown selection mode: {selection}")
        before = capital
        capital, wins, trades = trade_selected(selected, capital, commission, stamp_tax)
        total_wins += wins
        total_trades += trades
        rows.append({
            "qid_date": int(date), "funds": capital,
            "day_return": (capital - before) / before if before else np.nan,
            "number_of_stocks": len(selected),
        })
    return pd.DataFrame(rows), total_wins, total_trades


def curve_metrics(curve: pd.DataFrame, wins: int, trades: int) -> dict[str, float]:
    if curve.empty:
        return {name: np.nan for name in ("ARR", "MDR", "CR", "SR", "WR")}
    arr = (curve.funds.iloc[-1] / INITIAL_CAPITAL) ** (242 / len(curve)) - 1
    drawdown = (curve.funds - curve.funds.cummax()) / curve.funds.cummax()
    mdr = -float(drawdown.min())
    cr = arr / mdr if mdr else np.nan
    std = curve.day_return.std()
    sr = (((1 + curve.day_return.mean()) ** 242 - 1 - 0.025) / (std * 242 ** 0.5)) if std else np.nan
    wr = wins / trades if trades else np.nan
    return {"ARR": float(arr), "MDR": mdr, "CR": float(cr), "SR": float(sr), "WR": float(wr)}


def load_actions(run_dir: Path, market: str) -> tuple[dict[int, int], Path]:
    path = require_file(
        artifact_dir(run_dir, "actions") / f"{market}_DQN_actions3.csv",
        f"{market} DQN actions (run main.py/T7main.py first if missing)",
    )
    frame = pd.read_csv(path)
    action_col = "action" if "action" in frame.columns else "real_action"
    if action_col not in frame.columns:
        raise ValueError(f"No action column in {path}: {frame.columns.tolist()}")
    frame["qid_date"] = to_int_dates(frame.qid_date)
    frame[action_col] = pd.to_numeric(frame[action_col], errors="coerce").fillna(0).astype(int)
    return dict(zip(frame.qid_date.dropna().astype(int), frame.loc[frame.qid_date.notna(), action_col])), path


def dqn_ranking_path(run_dir: Path, market: str) -> Path:
    return require_file(
        artifact_dir(run_dir, "rankings") / f"{market}_{DQN_RANKER}_test3.csv",
        f"{market} {DQN_RANKER} test ranking used by DQN",
    )


def index_curve(market: str, start: int, end: int) -> tuple[pd.DataFrame, Path]:
    code = MARKET_CODES[market]
    candidates = [DATA_DIR / f"{code}merge.csv", DATA_DIR / "dapan" / f"{code}merge.csv"]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Index curve source not found for {market}: {candidates}")
    frame = pd.read_csv(path)
    date_col = "qid_date" if "qid_date" in frame.columns else "trade_date"
    frame["qid_date"] = to_int_dates(frame[date_col])
    fund_col = "total_profit" if "total_profit" in frame.columns else None
    if fund_col is None:
        raise ValueError(f"Index source lacks total_profit: {path}")
    frame["funds"] = pd.to_numeric(frame[fund_col], errors="coerce")
    frame = frame[(frame.qid_date >= start) & (frame.qid_date <= end)]
    return normalize_funds(frame), path


def baseline_curve(market: str, start: int, end: int) -> tuple[pd.DataFrame, Path]:
    path = require_file(DATA_DIR / f"{MARKET_CODES[market]}merge_open_close_final.csv", "stock data")
    frame = pd.read_csv(path, usecols=["qid_date", "stock_code", "close", "pclose"])
    frame["qid_date"] = to_int_dates(frame.qid_date)
    frame = frame[(frame.qid_date >= start) & (frame.qid_date <= end)]
    curve, _, _ = backtest(frame, selection="all")
    return normalize_funds(curve), path


def compute_c1(data_path: Path, force: bool) -> pd.DataFrame:
    sources = [
        DATA_DIR / f"{code}merge.csv" for code in MARKET_CODES.values()
    ] + [
        DATA_DIR / f"{code}merge_open_close_final.csv" for code in MARKET_CODES.values()
    ] + implementation_paths()
    signature = file_signature(sources)
    cached = cached_csv(data_path, signature, force)
    if cached is not None:
        return cached
    rows = []
    for market in MARKET_ORDER:
        index, _ = index_curve(market, LONG_START, TEST_END)
        baseline, _ = baseline_curve(market, LONG_START, TEST_END)
        for model, curve in ((INDEX_NAMES[market], index), ("Baseline portfolio", baseline)):
            part = curve.copy()
            part["market"] = market
            part["model"] = model
            rows.extend(part.to_dict(orient="records"))
    return save_csv(pd.DataFrame(rows), data_path, signature)


def plot_c1(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 6.8), sharex=False)
    for ax, market, panel in zip(axes, MARKET_ORDER, ("(a)", "(b)")):
        subset = frame[frame.market == market]
        for model, group in subset.groupby("model", sort=False):
            color = COLORS["Baseline portfolio"] if model == "Baseline portfolio" else COLORS["index"]
            ax.plot(as_datetime(group.qid_date), group.funds / 1_000_000, label=model, linewidth=1.7, color=color)
        ax.set_title(f"{panel} {MARKET_TITLES[market]}", loc="left", fontsize=11)
        ax.set_ylabel("Total fund (million)")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=30)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        style_axis(ax)
    axes[-1].set_xlabel("Trading day")
    fig.tight_layout()
    save_figure(fig, path)


def broker_source(args: argparse.Namespace, market: str) -> tuple[pd.DataFrame, Path, str, str]:
    path = args.broker_file or (DATA_DIR / f"{MARKET_CODES[market]}merge_open_close_final.csv")
    path = require_file(path, "brokerage report data")
    frame = pd.read_csv(path)
    if "market" in frame.columns and args.broker_file:
        frame = frame[frame.market.astype(str).str.lower().str.contains(market.lower())]
    requested = args.broker_column
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"Broker column {requested!r} not found in {path}")
        column = requested
        mode = "true_identifier" if requested not in {"broker_size", "broker_status"} else "proxy"
    elif "broker_id" in frame.columns:
        column, mode = "broker_id", "true_identifier"
    elif "brokerage_id" in frame.columns:
        column, mode = "brokerage_id", "true_identifier"
    elif "broker_size" in frame.columns:
        column, mode = "broker_size", "proxy"
    else:
        raise ValueError(
            f"No brokerage identifier in {path}. Pass --broker_file and --broker_column."
        )
    return frame, path, column, mode


def compute_c2(data_path: Path, args: argparse.Namespace) -> pd.DataFrame:
    source_paths = []
    source_meta = []
    loaded = {}
    for market in MARKET_ORDER:
        frame, path, column, mode = broker_source(args, market)
        loaded[market] = (frame, column, mode)
        source_paths.append(path)
        source_meta.append((market, column, mode, args.min_broker_reports))
    signature = digest_text({
        "files": file_signature([*source_paths, *implementation_paths()]),
        "settings": source_meta,
    })
    cached = cached_csv(data_path, signature, args.force)
    if cached is not None:
        return cached
    rows = []
    required = {"qid_date", "stock_code", "close", "pclose"}
    for market in MARKET_ORDER:
        frame, broker_column, mode = loaded[market]
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"C2 source for {market} is missing {missing}")
        frame = frame.copy()
        frame["qid_date"] = to_int_dates(frame.qid_date)
        frame = frame[(frame.qid_date >= LONG_START) & (frame.qid_date <= TEST_END)]
        frame = frame.dropna(subset=[broker_column, "qid_date", "pclose", "close"])
        counts = frame[broker_column].value_counts()
        keep = counts[counts >= args.min_broker_reports].index
        for broker, group in frame[frame[broker_column].isin(keep)].groupby(broker_column, sort=True):
            curve, wins, trades = backtest(group, selection="all")
            metrics = curve_metrics(curve, wins, trades)
            rows.append({
                "market": market,
                "broker_group": str(broker),
                "broker_column": broker_column,
                "broker_grouping_mode": mode,
                "n_reports": len(group),
                **metrics,
            })
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No brokerage group satisfies --min_broker_reports")
    return save_csv(result, data_path, signature)


def plot_c2(frame: pd.DataFrame, path: Path) -> None:
    metrics = [("ARR", 1.0), ("MDR", 10.0), ("CR", 1.0), ("SR", 1.0), ("WR", 10.0)]
    positions = np.arange(len(metrics), dtype=float)
    width = 0.28
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for offset, market, color in ((-width / 1.5, "Main", "#8EC5BD"), (width / 1.5, "ChiNext", "#F2EFA6")):
        values = [
            pd.to_numeric(frame.loc[frame.market == market, name], errors="coerce").dropna() * scale
            for name, scale in metrics
        ]
        bp = ax.boxplot(
            values, positions=positions + offset, widths=width, patch_artist=True,
            manage_ticks=False, showfliers=True,
        )
        for box in bp["boxes"]:
            box.set_facecolor(color)
            box.set_edgecolor("#666666")
        for element in ("whiskers", "caps", "medians"):
            for artist in bp[element]:
                artist.set_color("#666666")
        bp["boxes"][0].set_label(MARKET_TITLES[market])
    ax.set_xticks(positions, ["ARR", "MDRx10", "CR", "SR", "WRx10"])
    ax.set_xlabel("Evaluation metrics")
    ax.set_ylabel("Value")
    mode = ", ".join(sorted(frame.broker_grouping_mode.unique()))
    if mode != "true_identifier":
        ax.set_title("Brokerage-performance proxy groups (broker_size)", loc="left", fontsize=10)
    ax.legend(frameon=True, fontsize=8, loc="upper left")
    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, path)


def compute_c3(run_dir: Path, data_path: Path, force: bool) -> pd.DataFrame:
    sources = []
    for market in MARKET_ORDER:
        _, action_path = load_actions(run_dir, market)
        sources.extend([action_path, dqn_ranking_path(run_dir, market)])
    signature = file_signature([*sources, *implementation_paths()])
    cached = cached_csv(data_path, signature, force)
    if cached is not None:
        return cached
    settings = (
        ("fee=0.00%, tax=0.00%", 0.0, 0.0),
        ("fee=0.01%, tax=0.10%", 0.0001, 0.001),
        ("fee=0.03%, tax=0.10%", 0.0003, 0.001),
        ("fee=0.05%, tax=0.10%", 0.0005, 0.001),
    )
    rows = []
    for market in MARKET_ORDER:
        actions, _ = load_actions(run_dir, market)
        ranked = pd.read_csv(dqn_ranking_path(run_dir, market))
        ranked["qid_date"] = to_int_dates(ranked.qid_date)
        ranked = ranked[(ranked.qid_date >= TEST_START) & (ranked.qid_date <= TEST_END)]
        for label, commission, tax in settings:
            curve, _, _ = backtest(
                ranked, selection="actions", actions=actions,
                commission=commission, stamp_tax=tax,
            )
            curve["market"] = market
            curve["scenario"] = label
            curve["commission"] = commission
            curve["stamp_tax"] = tax
            rows.extend(curve.to_dict(orient="records"))
    return save_csv(pd.DataFrame(rows), data_path, signature)


def plot_c3(frame: pd.DataFrame, path: Path) -> None:
    scenario_colors = ("#FFC000", "#A5A5A5", "#4472C4", "#ED7D31")
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.8), sharex=False)
    for ax, market, panel in zip(axes, MARKET_ORDER, ("(a)", "(b)")):
        subset = frame[frame.market == market]
        for color, (scenario, group) in zip(scenario_colors, subset.groupby("scenario", sort=False)):
            ax.plot(as_datetime(group.qid_date), group.funds / 1_000_000, label=scenario, color=color, linewidth=1.5)
        ax.set_title(f"{panel} {MARKET_TITLES[market]}", loc="left", fontsize=11)
        ax.set_ylabel("Total fund (million)")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=30)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        style_axis(ax)
    axes[-1].set_xlabel("Trading day")
    fig.tight_layout()
    save_figure(fig, path)


def compute_c4(t6_csv: Path, data_path: Path, force: bool) -> pd.DataFrame:
    t6_csv = require_file(t6_csv, f"T6 {T6_REPLICATIONS}-replication results")
    signature = file_signature([t6_csv, *implementation_paths()])
    cached = cached_csv(data_path, signature, force)
    if cached is not None:
        return cached
    frame = pd.read_csv(t6_csv, float_precision="round_trip")
    if "sampling_rate" not in frame.columns and "rate" in frame.columns:
        frame["sampling_rate"] = pd.to_numeric(frame.rate, errors="coerce")
    required = {"market", "sampling_rate", "model", "seed", "ARR"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"T6 input is missing {missing}: {t6_csv}")
    frame["sampling_rate"] = pd.to_numeric(
        frame["sampling_rate"], errors="coerce"
    ).round(10)
    frame = frame[
        frame.market.isin(MARKET_ORDER)
        & frame.model.isin(MODELS)
        & frame.sampling_rate.isin(RATES)
    ].copy()
    counts = frame.groupby(["market", "sampling_rate", "model"]).size()
    incomplete = counts[counts < T6_REPLICATIONS]
    if not incomplete.empty:
        raise ValueError(
            f"Figure C4 requires {T6_REPLICATIONS} results per cell; incomplete:\n{incomplete}"
        )
    return save_csv(frame, data_path, signature)


def plot_c4(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.6, 7.2), sharex=False)
    positions = np.arange(len(RATES), dtype=float)
    width = 0.22
    offsets = (-width, 0.0, width)
    for ax, market, panel in zip(axes, MARKET_ORDER, ("(a)", "(b)")):
        subset = frame[frame.market == market]
        for model, offset in zip(MODELS, offsets):
            values = [
                pd.to_numeric(
                    subset[(subset.model == model) & (subset.sampling_rate == rate)].ARR,
                    errors="coerce",
                ).dropna()
                for rate in RATES
            ]
            bp = ax.boxplot(
                values, positions=positions + offset, widths=width * 0.9,
                patch_artist=True, manage_ticks=False, showfliers=True,
                flierprops={"markersize": 2.2, "markerfacecolor": "#555555", "markeredgecolor": "#555555"},
            )
            for box in bp["boxes"]:
                box.set_facecolor(COLORS[model])
                box.set_edgecolor("#555555")
            for element in ("whiskers", "caps", "medians"):
                for artist in bp[element]:
                    artist.set_color("#555555")
            bp["boxes"][0].set_label(model)
        ax.set_xticks(positions, [f"{int(rate * 100)}%" for rate in RATES])
        ax.set_title(f"{panel} {MARKET_TITLES[market]}", loc="left", fontsize=11)
        ax.set_ylabel("Annualized return")
        ax.legend(frameon=True, fontsize=8, loc="upper right")
        style_axis(ax)
    axes[-1].set_xlabel("Sampling rate")
    fig.tight_layout()
    save_figure(fig, path)


def esg_curve(
    frame: pd.DataFrame,
    actions: dict[int, int],
    *,
    threshold: float | None,
    prefilter: bool,
) -> pd.DataFrame:
    capital = INITIAL_CAPITAL
    rows = []
    for date, group in frame.sort_values("qid_date", kind="mergesort").groupby("qid_date", sort=True):
        top_n = int(actions.get(int(date), 0))
        if top_n <= 0:
            selected = group.iloc[0:0]
        elif threshold is None:
            selected = group.nlargest(min(top_n, len(group)), "prediction")
        elif prefilter:
            eligible = group[group.ESG >= threshold]
            selected = eligible.nlargest(min(top_n, len(eligible)), "prediction")
        else:
            selected = group.nlargest(min(top_n, len(group)), "prediction")
            selected = selected[selected.ESG >= threshold]
        before = capital
        capital, _, _ = trade_selected(selected, capital)
        rows.append({
            "qid_date": int(date), "funds": capital,
            "day_return": (capital - before) / before if before else np.nan,
            "number_of_stocks": len(selected),
        })
    return pd.DataFrame(rows)


def compute_c5(run_dir: Path, data_path: Path, force: bool) -> pd.DataFrame:
    sources = []
    for market in MARKET_ORDER:
        _, action_path = load_actions(run_dir, market)
        sources.extend([
            action_path,
            DATA_DIR / "ESG" / f"{MARKET_CODES[market]}temp_test_ndcg_train3_esg.csv",
            DATA_DIR / f"{MARKET_CODES[market]}merge.csv",
            DATA_DIR / f"{MARKET_CODES[market]}merge_open_close_final.csv",
        ])
    signature = file_signature([*sources, *implementation_paths()])
    cached = cached_csv(data_path, signature, force)
    if cached is not None:
        return cached
    rows = []
    for market in MARKET_ORDER:
        actions, _ = load_actions(run_dir, market)
        esg_path = require_file(
            DATA_DIR / "ESG" / f"{MARKET_CODES[market]}temp_test_ndcg_train3_esg.csv",
            f"{market} ESG ranking data",
        )
        esg = pd.read_csv(esg_path)
        esg["qid_date"] = to_int_dates(esg.qid_date)
        esg = esg[(esg.qid_date >= TEST_START) & (esg.qid_date <= TEST_END)].copy()
        index, _ = index_curve(market, TEST_START, TEST_END)
        baseline, _ = baseline_curve(market, TEST_START, TEST_END)
        curves = {
            INDEX_NAMES[market]: index,
            "Baseline portfolio": baseline,
            "No ESG": esg_curve(esg, actions, threshold=None, prefilter=False),
            "NS 25%": esg_curve(esg, actions, threshold=5.52, prefilter=False),
            "NS 50%": esg_curve(esg, actions, threshold=6.02, prefilter=False),
            "PI 25%": esg_curve(esg, actions, threshold=5.52, prefilter=True),
            "PI 50%": esg_curve(esg, actions, threshold=6.02, prefilter=True),
        }
        for strategy, curve in curves.items():
            part = normalize_funds(curve)
            part["market"] = market
            part["strategy"] = strategy
            rows.extend(part.to_dict(orient="records"))
    return save_csv(pd.DataFrame(rows), data_path, signature)


def plot_c5(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.0), sharex=False)
    for ax, market, panel in zip(axes, MARKET_ORDER, ("(a)", "(b)")):
        subset = frame[frame.market == market]
        for strategy, group in subset.groupby("strategy", sort=False):
            color = COLORS.get(strategy, COLORS["index"])
            ax.plot(as_datetime(group.qid_date), group.funds / 1_000_000, label=strategy, color=color, linewidth=1.5)
        ax.set_title(f"{panel} {MARKET_TITLES[market]}", loc="left", fontsize=11)
        ax.set_ylabel("Total fund (million)")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=30)
        ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper left")
        style_axis(ax)
    axes[-1].set_xlabel("Trading day")
    fig.tight_layout()
    save_figure(fig, path)


def main() -> None:
    args = parse_args()
    validate_runtime()
    figures = selected_figures(args.figures)
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    data_dir = output_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    data_outputs: list[Path] = []
    notes = {
        "C1": "Long-horizon index and all-report baseline portfolio curves.",
        "C2": "Uses a true brokerage identifier when supplied; otherwise broker_size is an explicitly labelled proxy.",
        "C3": "Current DQN daily stock counts and current LambdaMART ranking, re-backtested under four fee settings.",
        "C4": f"Uses the fixed {T6_REPLICATIONS}-replication-per-cell T6 result ledger.",
        "C5": "Current DQN daily stock counts applied to the supplied ESG ranking data.",
    }

    if "C1" in figures:
        data_path = data_dir / "FigC1_long_horizon_curves.csv"
        frame = compute_c1(data_path, args.force)
        path = output_dir / "FigC1_baseline_portfolio_and_indices.png"
        plot_c1(frame, path)
        outputs.append(path); data_outputs.append(data_path)
    if "C2" in figures:
        data_path = data_dir / "FigC2_brokerage_performance.csv"
        frame = compute_c2(data_path, args)
        path = output_dir / "FigC2_brokerage_performance_boxplots.png"
        plot_c2(frame, path)
        outputs.append(path); data_outputs.append(data_path)
        notes["C2_grouping_mode"] = sorted(frame.broker_grouping_mode.unique().tolist())
        notes["C2_grouping_column"] = sorted(frame.broker_column.unique().tolist())
    if "C3" in figures:
        data_path = data_dir / "FigC3_transaction_cost_curves.csv"
        frame = compute_c3(run_dir, data_path, args.force)
        path = output_dir / "FigC3_transaction_cost_sensitivity.png"
        plot_c3(frame, path)
        outputs.append(path); data_outputs.append(data_path)
    if "C4" in figures:
        data_path = data_dir / "FigC4_sampling_ARR.csv"
        frame = compute_c4(args.t6_csv, data_path, args.force)
        path = output_dir / "FigC4_sampling_robustness_boxplots.png"
        plot_c4(frame, path)
        outputs.append(path); data_outputs.append(data_path)
    if "C5" in figures:
        data_path = data_dir / "FigC5_ESG_curves.csv"
        frame = compute_c5(run_dir, data_path, args.force)
        path = output_dir / "FigC5_ESG_strategy_curves.png"
        plot_c5(frame, path)
        outputs.append(path); data_outputs.append(data_path)

    manifest = {
        "scope": "appendix empirical Figures C1-C5",
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "figures": figures,
        "runtime": runtime_versions(),
        "notes": {key: value for key, value in notes.items() if key in figures or key.startswith("C2_")},
        "outputs": {path.name: sha256(path) for path in outputs},
        "data": {path.name: sha256(path) for path in data_outputs},
    }
    manifest_path = output_dir / "appendix_figures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
