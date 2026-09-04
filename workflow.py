"""Unified test, baseline, table and export implementation.

Run this after train.py. Baselines are fitted here because their runtime is
short; rankers and DQN are loaded from the selected run directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_core import (
    CODE_DIR,
    DATA_DIR,
    DQN_RANKER,
    MARKETS,
    METRIC_NAMES,
    all_stock_metrics,
    artifact_dir,
    backtest_predictions,
    dqn_ranking_path,
    esg_metrics,
    evaluate_dqn,
    fit_baseline,
    index_metrics,
    metric_record,
    runtime_versions,
    sha256,
    validate_runtime,
    write_results,
)
from runtime_config import load_stage_seed_config, set_global_determinism, stage_seed


BASELINES = ("LR", "MLP_R", "SVM_R", "XGB_R", "SVM_C", "MLP_C", "XGB_C")
PRIMARY = ("LambdaRank", "LambdaMART", "LTR-DQN")
BASELINE_CACHE_VERSION = "paper-t5-source-config-v2"


def baseline_ranking(
    run_dir: Path, market: str, year: int, model: str, seed: int
):
    """Fit a baseline once and retain its seeded predictions for later consumers."""
    import pandas as pd

    path = artifact_dir(run_dir, "rankings") / f"{market}_{model}_test{year}.csv"
    if path.is_file():
        cached = pd.read_csv(path)
        if (
            "baseline_seed" in cached.columns
            and "baseline_cache_version" in cached.columns
            and not cached.empty
            and (pd.to_numeric(cached.baseline_seed, errors="coerce") == seed).all()
            and (cached.baseline_cache_version == BASELINE_CACHE_VERSION).all()
        ):
            return cached
    ranked = fit_baseline(market, year, model, seed=seed)
    ranked["baseline_seed"] = seed
    ranked["baseline_cache_version"] = BASELINE_CACHE_VERSION
    path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(path, index=False)
    return ranked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate experiments and export one multi-sheet workbook")
    parser.add_argument(
        "--run_dir", type=Path, default=CODE_DIR,
        help="Artifact root; omit for the clean temp/model/results layout",
    )
    parser.add_argument(
        "--tables", default="T3,T4,T5,T7",
        help="Comma-separated subset of T3,T4,T5,T7; T6 uses T6_main.py",
    )
    parser.add_argument("--markets", default="all", help="Main,ChiNext or all")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Optional global seed override for every stage",
    )
    parser.add_argument(
        "--seed_config", type=Path, default=None,
        help="Optional JSON map of independent market/year/stage seeds",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--export_csvs", action="store_true",
        help="Also export results_long.csv and one paper-format CSV per table",
    )
    parser.add_argument(
        "--no_baselines", action="store_false", dest="include_baselines", default=True,
        help="Skip the fast baseline train-and-test step",
    )
    parser.add_argument(
        "--dqn_eval_mode", choices=["dqn"], default="dqn",
        help="Generate actions from the trained DQN policy deterministically.",
    )
    return parser.parse_args()


def selected_markets(value: str) -> list[str]:
    if value.lower() == "all":
        return list(MARKETS)
    result = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(result) - set(MARKETS))
    if invalid:
        raise ValueError(f"Unknown markets: {invalid}")
    return result


def require_file(path: Path, purpose: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"{purpose} not found: {path}. Run train.py first to create long-running model artifacts."
        )
    return path


def validate_dqn_training_input(
    run_dir: Path, market: str, year: int, test_ranking: Path
) -> None:
    """Reject DQN checkpoints trained from a different ranker or run.

    DQN state features must come from the LambdaMART ranking generated for
    this run.  Older artifacts may still contain a ``ranking_model`` entry of
    ``LambdaRank``; accepting them would silently produce a different
    experiment while the result file still looks valid.
    """
    root = Path(run_dir).resolve()
    manifest_path = (
        CODE_DIR / "temp" / "train_manifest.json"
        if root == CODE_DIR.resolve()
        else root / "train_manifest.json"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"DQN training manifest not found: {manifest_path}. "
            "Run train.py after generating LambdaMART rankings."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [
        item for item in manifest.get("dqn", [])
        if item.get("market") == market and int(item.get("train_year", -1)) == int(year)
    ]
    if len(records) != 1:
        raise ValueError(
            f"Expected one DQN manifest record for {market}/year {year}; found {len(records)}."
        )
    record = records[0]
    if record.get("ranking_model") != DQN_RANKER:
        raise ValueError(
            f"DQN checkpoint for {market}/year {year} was trained from "
            f"{record.get('ranking_model')!r}, not {DQN_RANKER!r}. Retrain DQN from LambdaMART."
        )
    expected_train = artifact_dir(root, "rankings") / f"{market}_{DQN_RANKER}_train{year}.csv"
    if not expected_train.is_file():
        raise FileNotFoundError(f"DQN LambdaMART training ranking not found: {expected_train}")
    declared_name = Path(str(record.get("ranking_input", ""))).name
    if declared_name != expected_train.name:
        raise ValueError(
            f"DQN manifest ranking input is {declared_name!r}; expected {expected_train.name!r}."
        )
    declared_hash = record.get("ranking_input_sha256")
    if declared_hash and declared_hash != sha256(expected_train):
        raise ValueError(
            f"DQN LambdaMART training ranking hash mismatch for {market}/year {year}."
        )
    if test_ranking.name != f"{market}_{DQN_RANKER}_test{year}.csv":
        raise ValueError(f"DQN test ranking must be a LambdaMART file, got {test_ranking.name!r}.")


def evaluate_ranker(run_dir: Path, market: str, year: int, model: str) -> dict[str, float]:
    path = require_file(
        artifact_dir(run_dir, "rankings") / f"{market}_{model}_test{year}.csv",
        f"{model} ranking output",
    )
    ranked = __import__("pandas").read_csv(path)
    return backtest_predictions(ranked)[0]


def evaluate_dqn_model(
    run_dir: Path, market: str, year: int, seed_config: dict, seed_override: int | None,
    *, fixed_actions: bool = False,
) -> dict[str, float]:
    ranking = require_file(
        dqn_ranking_path(run_dir, market, year, "test"),
        f"{DQN_RANKER} ranking input for DQN",
    )
    model_path = require_file(
        artifact_dir(run_dir, "models") / f"{market}_DQN_train{year}.pt", "DQN model"
    )
    validate_dqn_training_input(run_dir, market, year, ranking)
    metrics, daily = evaluate_dqn(
        market, year, ranking, model_path,
        seed=stage_seed(MARKETS[market], year, "evaluation", seed_config, seed_override),
        return_daily=True,
        fixed_actions=fixed_actions,
    )
    actions_dir = artifact_dir(run_dir, "actions")
    actions_dir.mkdir(parents=True, exist_ok=True)
    action_path = actions_dir / f"{market}_DQN_actions{year}.csv"
    daily.loc[daily.qid_date >= 20211207, ["qid_date", "real_action"]].rename(
        columns={"real_action": "action"}
    ).to_csv(action_path, index=False)
    provenance = {
        "market": market,
        "train_year": year,
        "evaluation_mode": "deterministic_greedy_online_updates",
        "evaluation_seed": stage_seed(
            MARKETS[market], year, "evaluation", seed_config, seed_override
        ),
        "ranking": str(ranking),
        "ranking_sha256": sha256(ranking),
        "checkpoint": str(model_path),
        "checkpoint_sha256": sha256(model_path),
        "actions": str(action_path),
        "actions_sha256": sha256(action_path),
    }
    action_path.with_suffix(".json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return metrics


def evaluate_dqn_with_actions(
    run_dir: Path, market: str, year: int, seed_config: dict, seed_override: int | None
) -> tuple[dict[str, float], object]:
    ranking = require_file(
        dqn_ranking_path(run_dir, market, year, "test"),
        f"{DQN_RANKER} ranking input for DQN",
    )
    model_path = require_file(
        artifact_dir(run_dir, "models") / f"{market}_DQN_train{year}.pt", "DQN model"
    )
    validate_dqn_training_input(run_dir, market, year, ranking)
    evaluation_seed = stage_seed(
        MARKETS[market], year, "evaluation", seed_config, seed_override
    )
    metrics, daily = evaluate_dqn(
        market, year, ranking, model_path,
        seed=evaluation_seed,
        return_daily=True,
        fixed_actions=False,
    )
    actions_dir = artifact_dir(run_dir, "actions")
    actions_dir.mkdir(parents=True, exist_ok=True)
    action_path = actions_dir / f"{market}_DQN_actions{year}.csv"
    daily.loc[daily.qid_date >= 20211207, ["qid_date", "real_action"]].rename(
        columns={"real_action": "action"}
    ).to_csv(action_path, index=False)
    provenance = {
        "market": market,
        "train_year": year,
        "evaluation_mode": "deterministic_greedy_online_updates",
        "evaluation_seed": evaluation_seed,
        "ranking": str(ranking),
        "ranking_sha256": sha256(ranking),
        "checkpoint": str(model_path),
        "checkpoint_sha256": sha256(model_path),
        "actions": str(action_path),
        "actions_sha256": sha256(action_path),
        "consumer": "T7",
    }
    action_path.with_suffix(".json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return metrics, daily


def run_t3(records: list[dict], markets: list[str]) -> None:
    for market in markets:
        code = MARKETS[market]
        index_path = DATA_DIR / f"{code}merge.csv"
        records.append(metric_record("T3", market, "Market indices", index_metrics(index_path)))
        records.append(metric_record("T3", market, "Baseline portfolios", all_stock_metrics(market)))


def run_table4(
    records: list[dict], run_dir: Path, markets: list[str],
    include_baselines: bool, seed_config: dict, seed_override: int | None,
    dqn_fixed_actions: bool = False,
) -> None:
    for market in markets:
        code = MARKETS[market]
        records.append(metric_record(
            "T4", market, "Market indices", index_metrics(DATA_DIR / f"{code}merge_T4.csv"), 3
        ))
        records.append(metric_record(
            "T4", market, "Baseline portfolios", all_stock_metrics(market, start_date=20211206), 3
        ))
        if include_baselines:
            for model in BASELINES:
                metrics, _ = backtest_predictions(
                    baseline_ranking(
                        run_dir, market, 3, model,
                        stage_seed(MARKETS[market], 3, "baseline", seed_config, seed_override),
                    ),
                    classifier=model.endswith("_C"),
                )
                records.append(metric_record("T4", market, model, metrics, 3))
        primary_metrics = evaluate_table4_primary(
            run_dir, market, seed_config, seed_override,
            dqn_fixed_actions=dqn_fixed_actions,
        )
        for model, metrics in primary_metrics.items():
            records.append(metric_record("T4", market, model, metrics, 3))


def evaluate_table4_primary(
    run_dir: Path, market: str, seed_config: dict, seed_override: int | None,
    *, dqn_fixed_actions: bool = False,
) -> dict[str, dict[str, float]]:
    """Evaluate the three T4 primary models from one run's 3-year artifacts."""
    return {
        model: (
            evaluate_dqn_model(
                run_dir, market, 3, seed_config, seed_override,
                fixed_actions=dqn_fixed_actions,
            )
            if model == "LTR-DQN"
            else evaluate_ranker(run_dir, market, 3, model)
        )
        for model in PRIMARY
    }


def run_table5(
    records: list[dict], run_dir: Path, markets: list[str],
    include_baselines: bool, seed_config: dict, seed_override: int | None,
    dqn_fixed_actions: bool = False,
) -> None:
    for year in (2, 4):
        for market in markets:
            if include_baselines:
                for model in BASELINES:
                    metrics, _ = backtest_predictions(
                        baseline_ranking(
                            run_dir, market, year, model,
                            stage_seed(MARKETS[market], year, "baseline", seed_config, seed_override),
                        ),
                        classifier=model.endswith("_C"),
                    )
                    records.append(metric_record("T5", market, model, metrics, year))
            for model in PRIMARY:
                metrics = evaluate_dqn_model(
                    run_dir, market, year, seed_config, seed_override,
                    fixed_actions=dqn_fixed_actions,
                ) if model == "LTR-DQN" else evaluate_ranker(run_dir, market, year, model)
                records.append(metric_record("T5", market, model, metrics, year))


def run_table7(
    records: list[dict], run_dir: Path, markets: list[str],
    seed_config: dict, seed_override: int | None,
) -> None:
    for market in markets:
        code = MARKETS[market]
        dqn_metrics, actions = evaluate_dqn_with_actions(
            run_dir, market, 3, seed_config, seed_override
        )
        records.append(metric_record(
            "T7", market, "Market indices", index_metrics(DATA_DIR / f"{code}merge_T4.csv"), 3
        ))
        records.append(metric_record(
            "T7", market, "Baseline portfolios", all_stock_metrics(market, start_date=20211206), 3
        ))
        records.append(metric_record("T7", market, "LTR-DQN without ESG", dqn_metrics, 3))
        for label, threshold, prefilter in (
            ("NS 25%", 5.52, False),
            ("NS 50%", 6.02, False),
            ("PI 25%", 5.52, True),
            ("PI 50%", 6.02, True),
        ):
            records.append(metric_record(
                "T7", market, label,
                esg_metrics(market, actions, threshold, prefilter),
                3,
            ))


def run_t6(records: list[dict], path: Path | None, raw_path: Path | None) -> None:
    import pandas as pd
    from t6_core import summarize_sampling

    selected = path or raw_path or (CODE_DIR / "temp" / "t6_runs" / "t6_raw.csv")
    if not selected.is_file():
        raise FileNotFoundError(
            f"T6 input not found: {selected}. Run train.py --t6 first, or pass --t6_csv."
        )
    frame = pd.read_csv(selected)
    if {"market", "sampling_rate", "model", "seed", "ARR"}.issubset(frame.columns):
        counts = frame.groupby(["market", "sampling_rate", "model"], dropna=False).size()
        incomplete = counts[counts < 500]
        if not incomplete.empty:
            print(
                "WARNING: T6 summary is provisional; incomplete cells have fewer "
                "than 500 seeds. Run train.py --t6 without --t6_max_seeds for the "
                "paper-comparable Std."
            )
        frame = summarize_sampling(frame)
    records.extend({"table": "T6", **row} for row in frame.to_dict(orient="records"))


def main() -> None:
    args = parse_args()
    validate_runtime()
    seed_config = load_stage_seed_config(args.seed_config)
    if args.seed is not None:
        set_global_determinism(args.seed)
    run_dir = args.run_dir.resolve()
    markets = selected_markets(args.markets)
    tables = {item.strip().upper() for item in args.tables.split(",") if item.strip()}
    invalid_tables = sorted(tables - {"T3", "T4", "T5", "T7"})
    if invalid_tables:
        raise ValueError(
            f"main.py supports T3,T4,T5,T7 only; invalid tables: {invalid_tables}. "
            "Use T6_main.py for T6."
        )
    records: list[dict] = []
    if "T3" in tables:
        run_t3(records, markets)
    if "T4" in tables:
        run_table4(
            records, run_dir, markets, args.include_baselines, seed_config, args.seed,
            False,
        )
    if "T5" in tables:
        run_table5(
            records, run_dir, markets, args.include_baselines, seed_config, args.seed,
            False,
        )
    if "T7" in tables:
        run_table7(records, run_dir, markets, seed_config, args.seed)
    output_dir = (args.output_dir or artifact_dir(run_dir, "results") / "combined").resolve()
    csv_path = output_dir / "results_long.csv" if args.export_csvs else None
    excel_path = output_dir / "results.xlsx"
    paper_csvs = write_results(
        records,
        csv_path,
        excel_path,
        write_table_csvs=args.export_csvs,
    )
    manifest = {
        "run_dir": str(run_dir),
        "tables": sorted(tables),
        "markets": markets,
        "seed": args.seed,
        "seed_config": str(args.seed_config.resolve()) if args.seed_config else "built-in",
        "runtime": runtime_versions(),
        "rows": len(records),
        "dqn_ranking_mode": "fresh_run",
        "dqn_evaluation_mode": args.dqn_eval_mode,
        "action_files": {
            path.name: sha256(path)
            for path in sorted(artifact_dir(run_dir, "actions").glob("*.csv"))
        },
        "long_csv": str(csv_path) if csv_path else None,
        "paper_csvs": paper_csvs,
        "excel": str(excel_path),
    }
    (output_dir / "main_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
