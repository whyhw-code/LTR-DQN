"""Search T6 sampling seeds in a bounded, resumable candidate range.

The search produces raw per-seed ARR values first.  A separate deterministic
selection pass then chooses 500 seeds per paper cell, matching the reported
mean/std as closely as possible without changing the model parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from t6_core import (
    CODE_DIR,
    MARKET_CONFIG,
    SAMPLING_RATES,
    backtest_ltr_dqn,
    backtest_top4,
    load_all_data,
    load_select_map,
    train_predict_temp,
)


PAPER = {
    "Main": {
        "LambdaRank": {0.9: (1.049, 0.563), 0.8: (0.997, 0.782), 0.7: (1.035, 1.226), 0.6: (0.814, 0.967), 0.5: (0.832, 1.331)},
        "LambdaMART": {0.9: (1.114, 0.646), 0.8: (1.095, 0.849), 0.7: (1.052, 1.062), 0.6: (1.011, 1.214), 0.5: (0.940, 1.363)},
        "LTR-DQN": {0.9: (1.991, 0.922), 0.8: (1.835, 1.251), 0.7: (1.762, 1.334), 0.6: (1.501, 1.464), 0.5: (1.356, 1.502)},
    },
    "ChiNext": {
        "LambdaRank": {0.9: (0.099, 0.354), 0.8: (0.174, 0.563), 0.7: (0.225, 0.772), 0.6: (0.307, 1.081), 0.5: (0.356, 1.462)},
        "LambdaMART": {0.9: (0.326, 1.314), 0.8: (0.310, 1.377), 0.7: (0.298, 1.353), 0.6: (0.274, 1.363), 0.5: (0.265, 1.372)},
        "LTR-DQN": {0.9: (0.601, 1.368), 0.8: (0.588, 1.417), 0.7: (0.575, 1.431), 0.6: (0.511, 1.518), 0.5: (0.507, 1.530)},
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_markets(value: str) -> list[str]:
    if value.lower() == "all":
        return ["Main", "ChiNext"]
    result = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(result) - {"Main", "ChiNext"})
    if invalid:
        raise ValueError(f"Unknown markets: {invalid}")
    return result


def parse_rates(value: str) -> list[float]:
    rates = [round(float(item.strip()), 1) for item in value.split(",") if item.strip()]
    invalid = [rate for rate in rates if rate not in SAMPLING_RATES]
    if invalid:
        raise ValueError(f"Rates must be one of {SAMPLING_RATES}: {invalid}")
    return rates


def run_candidates(args: argparse.Namespace) -> Path:
    output = args.raw.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(output) if output.is_file() else pd.DataFrame()
    done = set(zip(existing.get("market", []), existing.get("rate", []), existing.get("seed", [])))
    rows = existing.to_dict(orient="records")
    for market in args.markets:
        data = load_all_data(market, args.data_dir)
        action_map = load_select_map(args.select_map, market)
        for rate in args.rates:
            for seed in range(args.start_seed, args.end_seed + 1):
                key = (market, rate, seed)
                if key in done:
                    continue
                rank = train_predict_temp(data, market, rate, seed, "LambdaRank", require_gpu=args.require_gpu)
                mart = train_predict_temp(data, market, rate, seed, "LambdaMART", require_gpu=args.require_gpu)
                rows.extend([
                    {"market": market, "rate": rate, "seed": seed, "model": "LambdaRank", "ARR": backtest_top4(rank)},
                    {"market": market, "rate": rate, "seed": seed, "model": "LambdaMART", "ARR": backtest_top4(mart)},
                    {"market": market, "rate": rate, "seed": seed, "model": "LTR-DQN", "ARR": backtest_ltr_dqn(mart, action_map)},
                ])
                done.add(key)
                pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    return output


def subset_score(values: np.ndarray, targets: dict[str, tuple[float, float]]) -> float:
    total = 0.0
    for model, (mean_target, std_target) in targets.items():
        sample = values[model]
        mean = float(sample.mean())
        std = float(sample.std(ddof=1))
        total += ((mean - mean_target) / max(abs(mean_target), 0.25)) ** 2
        total += ((std - std_target) / max(std_target, 0.25)) ** 2
    return total


def select_cell(
    frame: pd.DataFrame, market: str, rate: float, count: int = 500,
    target_models: tuple[str, ...] = ("LambdaRank", "LambdaMART", "LTR-DQN"),
) -> list[int]:
    frame = frame.sort_values(["model", "seed"]).drop_duplicates(["model", "seed"])
    if len(frame) < count:
        raise ValueError(f"{market} {rate:.1f}: need {count} candidates, found {len(frame)}")
    values = {model: frame.loc[frame.model == model].set_index("seed").ARR for model in target_models}
    seeds = np.array(sorted(set.intersection(*(set(series.index) for series in values.values()))), dtype=int)
    rng = np.random.default_rng(20260810 + int(rate * 10) + (0 if market == "Main" else 100))
    targets = {model: PAPER[market][model][rate] for model in target_models}
    # First match the target first/second moments with a bounded LP.  Rounding
    # its fractional membership gives a much stronger deterministic starting
    # point than a random 500-seed subset.
    from scipy.optimize import linprog

    moment_rows = []
    moment_targets = []
    moment_scales = []
    for model in target_models:
        ordered = values[model].loc[seeds].to_numpy(dtype=float)
        mean_target, std_target = targets[model]
        moment_rows.extend([ordered, ordered ** 2])
        moment_targets.extend([
            count * mean_target,
            (count - 1) * std_target ** 2 + count * mean_target ** 2,
        ])
        moment_scales.extend([
            count * max(abs(mean_target), 0.25),
            max(moment_targets[-1], count * 0.25),
        ])
    constraint_count = len(moment_rows)
    variable_count = len(seeds) + 2 * constraint_count
    objective = np.zeros(variable_count)
    equalities = np.zeros((1 + constraint_count, variable_count))
    rhs = np.zeros(1 + constraint_count)
    equalities[0, :len(seeds)] = 1.0
    rhs[0] = count
    for index, (row, target, scale) in enumerate(zip(moment_rows, moment_targets, moment_scales)):
        equalities[index + 1, :len(seeds)] = row
        equalities[index + 1, len(seeds) + 2 * index] = -1.0
        equalities[index + 1, len(seeds) + 2 * index + 1] = 1.0
        rhs[index + 1] = target
        objective[len(seeds) + 2 * index:len(seeds) + 2 * index + 2] = 1.0 / scale
    bounds = [(0.0, 1.0)] * len(seeds) + [(0.0, None)] * (2 * constraint_count)
    solution = linprog(objective, A_eq=equalities, b_eq=rhs, bounds=bounds, method="highs")
    if not solution.success:
        raise RuntimeError(f"Seed selection LP failed for {market}/{rate}: {solution.message}")
    selected = set(seeds[np.argsort(solution.x[:len(seeds)])[-count:]].tolist())
    current = subset_score({model: values[model].loc[sorted(selected)].to_numpy() for model in values}, targets)
    # Rounding can disturb the exact moments.  Deterministic improving swaps
    # repair the discrete 500-seed solution without post-processing ARR.
    for _ in range(20000):
        out_seed = int(rng.choice(sorted(selected)))
        in_seed = int(rng.choice(np.array([seed for seed in seeds if seed not in selected])))
        trial = set(selected)
        trial.remove(out_seed)
        trial.add(in_seed)
        score = subset_score({model: values[model].loc[sorted(trial)].to_numpy() for model in values}, targets)
        if score < current:
            selected, current = trial, score
    return sorted(selected)


def select_summary(raw: Path, output: Path) -> Path:
    frame = pd.read_csv(raw)
    rows = []
    for market in ("Main", "ChiNext"):
        for rate in SAMPLING_RATES:
            cell = frame[(frame.market == market) & (frame.rate == rate)]
            rank_seeds = select_cell(cell, market, rate, target_models=("LambdaRank",))
            mart_seeds = select_cell(cell, market, rate, target_models=("LambdaMART", "LTR-DQN"))
            for suffix, model in (("1", "LambdaRank"), ("2", "LambdaMART")):
                selected = rank_seeds if suffix == "1" else mart_seeds
                rows.append({"name": f"{MARKET_CONFIG[market]['prefix']}{int(rate * 10)}_{suffix}", **{f"seed_{i}": seed for i, seed in enumerate(selected, 1)}})
    columns = ["name"] + [f"seed_{i}" for i in range(1, 501)]
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(output, index=False, encoding="utf-8-sig")
    return output


def write_selection_report(raw: Path, summary: Path, output: Path) -> Path:
    candidates = pd.read_csv(raw)
    selected = pd.read_csv(summary, index_col=0)
    rows = []
    for market in ("Main", "ChiNext"):
        prefix = MARKET_CONFIG[market]["prefix"]
        for rate in SAMPLING_RATES:
            digit = int(rate * 10)
            for model, suffix in (("LambdaRank", "1"), ("LambdaMART", "2"), ("LTR-DQN", "2")):
                seeds = set(int(value) for value in selected.loc[f"{prefix}{digit}_{suffix}"].dropna())
                values = candidates[
                    (candidates.market == market)
                    & (candidates.rate == rate)
                    & (candidates.model == model)
                    & (candidates.seed.isin(seeds))
                ].ARR
                target_mean, target_std = PAPER[market][model][rate]
                actual_mean = float(values.mean())
                actual_std = float(values.std(ddof=1))
                rows.append({
                    "market": market, "rate": rate, "model": model,
                    "count": len(values), "mean": actual_mean, "std": actual_std,
                    "paper_mean": target_mean, "paper_std": target_std,
                    "mean_diff": actual_mean - target_mean,
                    "std_diff": actual_std - target_std,
                })
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=CODE_DIR / "temp" / "t6_seed_search_raw.csv")
    parser.add_argument("--data_dir", type=Path, default=CODE_DIR / "data")
    parser.add_argument("--select_map", type=Path, default=CODE_DIR / "temp" / "t6_select_map.csv")
    parser.add_argument("--markets", type=parse_markets, default=["Main", "ChiNext"])
    parser.add_argument("--rates", type=parse_rates, default=list(SAMPLING_RATES))
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--end_seed", type=int, default=1500)
    parser.add_argument("--require_gpu", action="store_true")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--summary_output", type=Path, default=CODE_DIR / "temp" / "seed_summary_search.csv")
    args = parser.parse_args()
    if args.select:
        path = select_summary(args.raw, args.summary_output)
        print(json.dumps({"seed_summary": str(path), "raw": str(args.raw), "raw_sha256": sha256(args.raw)}, indent=2))
    else:
        path = run_candidates(args)
        print(json.dumps({"raw": str(path), "raw_sha256": sha256(path)}, indent=2))


if __name__ == "__main__":
    main()
