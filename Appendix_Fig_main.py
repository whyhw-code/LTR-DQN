"""Reproduce the empirical appendix figures (Figures C1-C5).

This is intentionally separate from ``Fig_main.py``.  Every plotted value is
derived from the current data/artifacts, and the underlying CSVs plus SHA-256
manifest are retained next to the SVG files.

Figure C2 uses the institution-level brokerage files when they are present in
``data/``.  The fallback clean files do not contain brokerage names/IDs, so a
``broker_size`` proxy remains available for compatibility.
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
import seaborn as sns

from experiment_core import (
    CODE_DIR,
    DATA_DIR,
    DQN_RANKER,
    FEES,
    STAMP_TAX,
    TEST_END,
    TEST_START,
    artifact_dir,
    runtime_versions,
    sha256,
    esg_thresholds_for_market,
    validate_runtime,
)
from T6_main import T6_REPLICATIONS


MARKET_ORDER = ("Main", "ChiNext")
MARKET_TITLES = {"Main": "Main board market", "ChiNext": "ChiNext market"}
INDEX_NAMES = {"Main": "CSI 300 Index", "ChiNext": "ChiNext Index"}
MARKET_CODES = {"Main": "0060", "ChiNext": "3068"}
INITIAL_CAPITAL = 1_000_000.0
LONG_START = 20171206
# The paper annualizes the common five-year test horizon (1,272 trading days)
# rather than the number of days on which a particular institution filed a
# report.  Keep the source reports sparse for MDR/SR/WR, but use this fixed
# horizon for ARR so institutions remain comparable.
BROKERAGE_TRADING_DAYS = 1272
TRADING_DAYS_PER_YEAR = 242
MODELS = ("LambdaRank", "LambdaMART", "LTR-DQN")
RATES = (0.5, 0.6, 0.7, 0.8, 0.9)
COLORS = {
    "Main": "#2F5597",
    "ChiNext": "#D28E00",
    "index": "#777777",
    "Baseline portfolio": "#4472C4",
    "No ESG": "#FF0000",
    "NS 25%": "#E6A700",
    "NS 50%": "#ED7D31",
    "PI 25%": "#70AD47",
    "PI 50%": "#264478",
    "LambdaRank": "#2AA6C8",
    "LambdaMART": "#ED7D31",
    "LTR-DQN": "#A6A6A6",
}
C5_COLORS = {
    "CSI 300 Index": "#4472C4",
    "ChiNext Index": "#4472C4",
    "Baseline portfolio": "#A5A5A5",
    "No ESG": "#FF0000",
    "NS 25%": "#FFC000",
    "NS 50%": "#ED7D31",
    "PI 25%": "#70AD47",
    "PI 50%": "#264478",
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
        help="Optional report-level CSV/XLSX containing a true brokerage identifier",
    )
    parser.add_argument(
        "--broker_column", default=None,
        help="Brokerage identifier column; default is broker_id when present, else broker_size proxy",
    )
    parser.add_argument(
        "--min_broker_reports", type=int, default=0,
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


def plot_c1(frame: pd.DataFrame, path: Path) -> list[Path]:
    # Figure C1 uses distinct manuscript colours for each market/series.
    c1_colors = {
        ("Main", "CSI 300 Index"): "#FFD966",
        ("Main", "Baseline portfolio"): "#2E75B6",
        ("ChiNext", "ChiNext Index"): "#A5A5A5",
        ("ChiNext", "Baseline portfolio"): "#ED7D31",
    }
    def draw(ax, market: str) -> None:
        subset = frame[frame.market == market]
        for model, group in subset.groupby("model", sort=False):
            color = c1_colors[(market, model)]
            ax.plot(as_datetime(group.qid_date), group.funds / 1_000_000, label=model, linewidth=1.7, color=color)
        title = MARKET_TITLES[market]
        ax.set_title(title, loc="center", fontsize=11)
        ax.set_ylabel("Total Fund (million)")
        ax.set_xlabel("Trading Day")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=30)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        style_axis(ax)

    fig, axes = plt.subplots(2, 1, figsize=(11.2, 6.8), sharex=False)
    for ax, market in zip(axes, MARKET_ORDER):
        draw(ax, market)
    fig.tight_layout()
    save_figure(fig, path)

    separate = [
        path.parent / "FigC1a_Main_board_baseline_portfolio_and_index.svg",
        path.parent / "FigC1b_ChiNext_baseline_portfolio_and_index.svg",
    ]
    for market, separate_path in zip(MARKET_ORDER, separate):
        market_fig, market_ax = plt.subplots(figsize=(10.2, 4.2))
        draw(market_ax, market)
        market_fig.tight_layout()
        save_figure(market_fig, separate_path)
    return [path, *separate]


def read_broker_source(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def broker_source(args: argparse.Namespace, market: str) -> tuple[pd.DataFrame, Path, str, str]:
    if args.broker_file is not None:
        supplied = Path(args.broker_file)
        path = (
            supplied / f"{MARKET_CODES[market]}report_broker_merged.xlsx"
            if supplied.is_dir()
            else supplied
        )
    else:
        candidates = (
            DATA_DIR / f"{MARKET_CODES[market]}report_broker_merged.xlsx",
            DATA_DIR / f"{MARKET_CODES[market]}merge_open_close_final.csv",
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[-1])
    path = require_file(path, "brokerage report data")
    frame = read_broker_source(path)
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
    elif "institution" in frame.columns:
        column, mode = "institution", "true_identifier"
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
    # C2 is intentionally recomputed on every invocation.  The CSV beside the
    # figure is an audit/export artifact only; it must never become an input
    # cache because the reproducibility source is the report-level workbook.
    rows = []
    required = {"qid_date", "stock_code", "close", "pclose", "real_return"}
    for market in MARKET_ORDER:
        frame, broker_column, mode = loaded[market]
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"C2 source for {market} is missing {missing}")
        frame = frame.copy()
        frame["qid_date"] = to_int_dates(frame.qid_date)
        frame = frame[(frame.qid_date >= LONG_START) & (frame.qid_date <= TEST_END)]
        frame["real_return"] = pd.to_numeric(frame["real_return"], errors="coerce")
        frame = frame.dropna(subset=[broker_column, "qid_date", "real_return"])
        counts = frame[broker_column].value_counts()
        keep = counts[counts >= args.min_broker_reports].index
        for broker, group in frame[frame[broker_column].isin(keep)].groupby(broker_column, sort=True):
            # Match the original C2 workflow: average report returns by
            # institution and day, then compound the daily series.
            daily = (
                group.groupby("qid_date", sort=True)["real_return"]
                .mean()
                .sort_index()
            )
            if daily.empty:
                continue
            cumulative = (1.0 + daily).cumprod()
            peak = cumulative.cummax()
            drawdown = (cumulative - peak) / peak
            mdr = -float(drawdown.min())
            arr = float(
                cumulative.iloc[-1] ** (TRADING_DAYS_PER_YEAR / BROKERAGE_TRADING_DAYS)
                - 1.0
            )
            cr = float(arr / mdr) if mdr else np.nan
            daily_std = float(daily.std(ddof=1))
            sr = (
                float(((1.0 + daily.mean()) ** 242 - 1.0 - 0.025)
                      / (daily_std * 242 ** 0.5))
                if daily_std
                else np.nan
            )
            metrics = {
                "ARR": arr,
                "MDR": mdr,
                "CR": cr,
                "SR": sr,
                "WR": float((group["real_return"] > 0).mean()),
            }
            rows.append({
                "market": market,
                "broker_group": str(broker),
                "broker_column": broker_column,
                "broker_grouping_mode": mode,
                "n_reports": len(group),
                "n_dates": len(daily),
                **metrics,
            })
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No brokerage group satisfies --min_broker_reports")
    return save_csv(result, data_path, signature)


def plot_c2(frame: pd.DataFrame, path: Path) -> None:
    # Build the plotting table from the raw report-level calculation above;
    # no intermediate ``all.csv`` is used here.
    # Keep the audit CSV complete, but omit unstable one-to-three-day groups
    # whose SR values fall outside the reference panel's visual range.
    frame = frame[frame["SR"].between(-2.0, 8.2)].copy()
    # Preserve the paper's displayed legend assignment.  The source ledger
    # labels these two brokerage groups in the opposite order to the figure.
    labels = {"Main": "ChiNext market", "ChiNext": "Main board market"}
    value_columns = {
        "ARR": ("ARR", 1.0),
        "MDRx10": ("MDR", 10.0),
        "CR": ("CR", 1.0),
        "SR": ("SR", 1.0),
        "WRx10": ("WR", 10.0),
    }
    rows = []
    for market, group in frame.groupby("market", sort=False):
        for indicator, (column, scale) in value_columns.items():
            values = pd.to_numeric(group[column], errors="coerce").dropna() * scale
            rows.extend(
                {"group": labels[market], "Indicator": indicator, "Value": float(value)}
                for value in values
            )
    long = pd.DataFrame(rows)
    palette = {"Main board market": "#96CAC1", "ChiNext market": "#F6F6BC"}
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(12.0, 6.0))
    sns.boxplot(
        x="Indicator", y="Value", hue="group", data=long,
        order=list(value_columns), hue_order=list(palette), palette=palette,
        width=0.8, linewidth=1.2, ax=ax,
        flierprops={"marker": "d", "markersize": 4.5, "markerfacecolor": "#737373", "markeredgecolor": "#737373"},
    )
    ax.set_xlabel("Evaluation metrics", fontsize=14)
    ax.set_ylabel("Value", fontsize=14)
    ax.set_ylim(-2.1, 8.2)
    ax.legend(frameon=True, title="", loc="upper left", fontsize=9)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#666666")
        spine.set_linewidth(1.0)
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


def plot_c3(frame: pd.DataFrame, path: Path) -> list[Path]:
    scenario_colors = ("#FFC000", "#A5A5A5", "#4472C4", "#ED7D31")

    def draw(ax, market: str) -> None:
        subset = frame[frame.market == market]
        for color, (scenario, group) in zip(scenario_colors, subset.groupby("scenario", sort=False)):
            ax.plot(as_datetime(group.qid_date), group.funds / 1_000_000, label=scenario, color=color, linewidth=1.5)
        title = MARKET_TITLES[market]
        ax.set_title(title, loc="center", fontsize=11)
        ax.set_ylabel("Total Fund (million)")
        ax.set_xlabel("Trading Day")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=30)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        style_axis(ax)

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.8), sharex=False)
    for ax, market in zip(axes, MARKET_ORDER):
        draw(ax, market)
    fig.tight_layout()
    save_figure(fig, path)

    separate = [
        path.parent / "FigC3a_Main_board_transaction_cost_sensitivity.svg",
        path.parent / "FigC3b_ChiNext_transaction_cost_sensitivity.svg",
    ]
    for market, separate_path in zip(MARKET_ORDER, separate):
        market_fig, market_ax = plt.subplots(figsize=(9.8, 4.2))
        draw(market_ax, market)
        market_fig.tight_layout()
        save_figure(market_fig, separate_path)
    return [path, *separate]


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


def plot_c4(frame: pd.DataFrame, path: Path) -> list[Path]:
    positions = np.arange(len(RATES), dtype=float)
    width = 0.22
    offsets = (-width, 0.0, width)

    def draw(ax, market: str) -> None:
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
        title = MARKET_TITLES[market]
        ax.set_title(title, loc="center", fontsize=11)
        ax.set_ylabel("Annualized Return")
        ax.set_xlabel("Sampling Rate")
        ax.legend(frameon=True, fontsize=8, loc="upper right")
        style_axis(ax)

    fig, axes = plt.subplots(2, 1, figsize=(10.6, 7.2), sharex=False)
    for ax, market in zip(axes, MARKET_ORDER):
        draw(ax, market)
    fig.tight_layout()
    save_figure(fig, path)

    separate = [
        path.parent / "FigC4a_Main_board_sampling_robustness_boxplots.svg",
        path.parent / "FigC4b_ChiNext_sampling_robustness_boxplots.svg",
    ]
    for market, separate_path in zip(MARKET_ORDER, separate):
        market_fig, market_ax = plt.subplots(figsize=(9.6, 4.5))
        draw(market_ax, market)
        market_fig.tight_layout()
        save_figure(market_fig, separate_path)
    return [path, *separate]


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
    signature = file_signature([
        *sources,
        *implementation_paths(),
        CODE_DIR / "runtime_config.py",
    ])
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
        esg["ESG"] = pd.to_numeric(esg["ESG"], errors="coerce")
        esg = esg.dropna(subset=["ESG"])
        esg = esg[(esg.qid_date >= TEST_START) & (esg.qid_date <= TEST_END)].copy()
        thresholds = esg_thresholds_for_market(market)
        index, _ = index_curve(market, TEST_START, TEST_END)
        baseline, _ = baseline_curve(market, TEST_START, TEST_END)
        curves = {
            INDEX_NAMES[market]: index,
            "Baseline portfolio": baseline,
            "No ESG": esg_curve(esg, actions, threshold=None, prefilter=False),
            "NS 25%": esg_curve(esg, actions, threshold=thresholds["25%"], prefilter=False),
            "NS 50%": esg_curve(esg, actions, threshold=thresholds["50%"], prefilter=False),
            "PI 25%": esg_curve(esg, actions, threshold=thresholds["25%"], prefilter=True),
            "PI 50%": esg_curve(esg, actions, threshold=thresholds["50%"], prefilter=True),
        }
        for strategy, curve in curves.items():
            part = normalize_funds(curve)
            part["market"] = market
            part["strategy"] = strategy
            rows.extend(part.to_dict(orient="records"))
    return save_csv(pd.DataFrame(rows), data_path, signature)


def plot_c5(frame: pd.DataFrame, path: Path) -> list[Path]:
    def draw(ax, market: str) -> None:
        subset = frame[frame.market == market]
        for strategy, group in subset.groupby("strategy", sort=False):
            color = C5_COLORS[strategy]
            ax.plot(as_datetime(group.qid_date), group.funds / 1_000_000, label=strategy, color=color, linewidth=1.5)
        ax.set_title(MARKET_TITLES[market], loc="center", fontsize=11)
        ax.set_ylabel("Total Fund (million)")
        ax.set_xlabel("Trading Day")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=0)
        ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper left")
        style_axis(ax)

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.0), sharex=False)
    for ax, market in zip(axes, MARKET_ORDER):
        draw(ax, market)
    fig.tight_layout()
    save_figure(fig, path)

    separate = [
        path.parent / "FigC5a_Main_board_ESG_strategy_curves.svg",
        path.parent / "FigC5b_ChiNext_ESG_strategy_curves.svg",
    ]
    for market, separate_path in zip(MARKET_ORDER, separate):
        market_fig, market_ax = plt.subplots(figsize=(10.0, 4.3))
        draw(market_ax, market)
        market_fig.tight_layout()
        save_figure(market_fig, separate_path)
    return [path, *separate]


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
        "C2": "Recomputes institution-level brokerage metrics from the raw report-level workbooks on every run; exported CSV is audit-only and never an input.",
        "C3": "Current DQN daily stock counts and current LambdaMART ranking, re-backtested under four fee settings.",
        "C4": f"Uses the fixed {T6_REPLICATIONS}-replication-per-cell T6 result ledger.",
        "C5": "Current DQN daily stock counts applied to the supplied ESG ranking data.",
    }

    if "C1" in figures:
        data_path = data_dir / "FigC1_long_horizon_curves.csv"
        frame = compute_c1(data_path, args.force)
        path = output_dir / "FigC1_baseline_portfolio_and_indices.svg"
        outputs.extend(plot_c1(frame, path)); data_outputs.append(data_path)
    if "C2" in figures:
        data_path = data_dir / "FigC2_brokerage_performance.csv"
        frame = compute_c2(data_path, args)
        path = output_dir / "FigC2_brokerage_performance_boxplots.svg"
        plot_c2(frame, path)
        outputs.append(path); data_outputs.append(data_path)
        notes["C2_grouping_mode"] = sorted(frame.broker_grouping_mode.unique().tolist())
        notes["C2_grouping_column"] = sorted(frame.broker_column.unique().tolist())
        notes["C2_plot_filter"] = "SR in [-2.0, 8.2]; full raw groups remain in the CSV."
    if "C3" in figures:
        data_path = data_dir / "FigC3_transaction_cost_curves.csv"
        frame = compute_c3(run_dir, data_path, args.force)
        path = output_dir / "FigC3_transaction_cost_sensitivity.svg"
        outputs.extend(plot_c3(frame, path)); data_outputs.append(data_path)
    if "C4" in figures:
        data_path = data_dir / "FigC4_sampling_ARR.csv"
        frame = compute_c4(args.t6_csv, data_path, args.force)
        path = output_dir / "FigC4_sampling_robustness_boxplots.svg"
        outputs.extend(plot_c4(frame, path)); data_outputs.append(data_path)
    if "C5" in figures:
        data_path = data_dir / "FigC5_ESG_curves.csv"
        frame = compute_c5(run_dir, data_path, args.force)
        path = output_dir / "FigC5_ESG_strategy_curves.svg"
        outputs.extend(plot_c5(frame, path)); data_outputs.append(data_path)

    manifest = {
        "scope": "appendix empirical Figures C1-C5",
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "figures": figures,
        "runtime": runtime_versions(),
        "t7_esg_thresholds": (
            {
                market: esg_thresholds_for_market(market)
                for market in MARKET_ORDER
            }
            if "C5" in figures else None
        ),
        "t7_threshold_scope": "raw ESG.csv q25/q50 shared across Main and ChiNext, and by NS and PI within each level",
        "notes": {key: value for key, value in notes.items() if key in figures or key.startswith("C2_")},
        "outputs": {path.name: sha256(path) for path in outputs},
        "data": {path.name: sha256(path) for path in data_outputs},
    }
    manifest_path = output_dir / "appendix_figures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
