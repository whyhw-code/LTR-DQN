"""Unified long-running training entry point.

Baselines are intentionally excluded: main.py fits and evaluates them in one
short-lived test job. This command trains only rankers and DQN models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model import (
    CODE_DIR,
    artifact_dir,
    canonicalize_ranking,
    dqn_ranking_path,
    DQN_RANKER,
    evaluate_dqn,
    MARKETS,
    PAPER_HYPERPARAMETERS,
    T4_MART_HYPERPARAMETERS,
    fit_ranker,
    runtime_versions,
    sha256,
    train_dqn,
    validate_runtime,
)
from runtime_config import (
    load_mart_config,
    load_rank_config,
    load_stage_seed_config,
    set_global_determinism,
    stage_seed,
)
from t6_core import T6_REPLICATIONS


def parse_years(value: str) -> list[int]:
    years = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not years or any(year not in (2, 3, 4) for year in years):
        raise argparse.ArgumentTypeError("years must be a comma-separated subset of 2,3,4")
    return years


def parse_markets(value: str) -> list[str]:
    if value.lower() == "all":
        return list(MARKETS)
    result = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(result) - set(MARKETS))
    if invalid:
        raise argparse.ArgumentTypeError(f"unknown markets: {invalid}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LambdaRank, LambdaMART and DQN models")
    parser.add_argument(
        "--run_dir", type=Path, default=CODE_DIR,
        help="Artifact root; default writes rankings to temp/ and checkpoints to model/",
    )
    parser.add_argument("--models", default="all", help="rankers,dqn,all")
    parser.add_argument("--markets", type=parse_markets, default=list(MARKETS))
    parser.add_argument("--years", type=parse_years, default=[2, 3, 4])
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Optional global seed override for every stage",
    )
    parser.add_argument(
        "--seed_config", type=Path, default=None,
        help="Optional JSON map of independent market/year/stage seeds",
    )
    parser.add_argument(
        "--rank_config", type=Path, default=None,
        help="Optional JSON map for unreported LambdaRank max_depth/n_estimators only",
    )
    parser.add_argument(
        "--mart_config", type=Path, default=None,
        help="Optional JSON map for CPU LambdaMART max_bin only",
    )
    parser.add_argument("--n_games", type=int, default=31)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.03)
    parser.add_argument("--eps_dec", type=float, default=0.00015)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_mem_size", type=int, default=100)
    parser.add_argument("--replace_target_iter", type=int, default=8)
    parser.add_argument(
        "--ranker_tree_method", choices=["hist", "exact", "approx"], default="approx",
        help="Single-CPU ranker tree builder; approx is the verified default",
    )
    parser.add_argument(
        "--t6", action="store_true",
        help=f"Also run the {T6_REPLICATIONS}-replication CPU sampling-rate robustness experiment",
    )
    parser.add_argument(
        "--t6_markets", default="all",
        help="Markets for T6 sampling: Main,ChiNext or all",
    )
    parser.add_argument(
        "--t6_max_seeds", type=int, default=T6_REPLICATIONS,
        help=f"Fresh CPU replications per T6 sampling cell (default: {T6_REPLICATIONS})",
    )
    parser.add_argument(
        "--t6_seed_summary", type=Path, default=None,
        help="Ranker/MART seed ledger; defaults to data/reproducibility/t6_cpu20_seed_summary.csv",
    )
    parser.add_argument(
        "--t6_dqn_seed_summary", type=Path, default=None,
        help="DQN provenance ledger; defaults to data/reproducibility/t6_cpu20_dqn_seed_summary.csv",
    )
    return parser.parse_args()


def upsert(records: list[dict], new_record: dict, keys: tuple[str, ...]) -> None:
    records[:] = [
        record for record in records
        if any(record.get(key) != new_record.get(key) for key in keys)
    ]
    records.append(new_record)


def generate_t6_select_map(
    run_dir: Path,
    markets: list[str],
    seed_config: dict,
    seed_override: int | None,
) -> Path:
    """Regenerate the T6 daily selection counts from freshly trained DQN models."""
    merged = None
    for market in markets:
        ranking = dqn_ranking_path(run_dir, market, 3, "test")
        model_path = artifact_dir(run_dir, "models") / f"{market}_DQN_train3.pt"
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Missing DQN checkpoint required for T6 actions: {model_path}"
            )
        _, daily = evaluate_dqn(
            market,
            3,
            ranking,
            model_path,
            seed=stage_seed(MARKETS[market], 3, "evaluation", seed_config, seed_override),
            return_daily=True,
            fixed_actions=False,
        )
        action_column = "60" if MARKETS[market] == "0060" else "3068"
        current = daily[["qid_date", "real_action"]].rename(
            columns={"real_action": action_column}
        )
        merged = current if merged is None else merged.merge(current, on="qid_date", how="outer")
    output = CODE_DIR / "temp" / "t6_select_map.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    merged = merged.sort_values("qid_date", kind="mergesort").fillna(0)
    for column in ("60", "3068"):
        if column not in merged:
            merged[column] = 0
        merged[column] = merged[column].astype(int)
    merged[["qid_date", "3068", "60"]].to_csv(output, index=False)
    return output


def main() -> None:
    args = parse_args()
    validate_runtime()
    paper_dqn_lr = PAPER_HYPERPARAMETERS["LTR-DQN"]["learning_rate"]
    if args.lr != paper_dqn_lr:
        raise ValueError(
            f"LTR-DQN learning rate is fixed by the paper at {paper_dqn_lr}; got {args.lr}"
        )
    seed_config = load_stage_seed_config(args.seed_config)
    rank_config = load_rank_config(args.rank_config)
    mart_config = load_mart_config(args.mart_config)
    if args.seed is not None:
        set_global_determinism(args.seed)
    run_dir = args.run_dir.resolve()
    rankings_dir = artifact_dir(run_dir, "rankings")
    models_dir = artifact_dir(run_dir, "models")
    rankings_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    selected = {item.strip().lower() for item in args.models.split(",")}
    train_rankers = "all" in selected or "rankers" in selected
    train_dqns = "all" in selected or "dqn" in selected
    if train_dqns and not train_rankers:
        raise ValueError(
            "DQN training requires fresh LambdaMART training in the same invocation. "
            "Use --models all (or --models rankers,dqn); stale ranking files are not accepted."
        )
    ranker_tree_method = args.ranker_tree_method
    effective_tree_method = ranker_tree_method
    manifest_path = (
        CODE_DIR / "temp" / "train_manifest.json"
        if run_dir == CODE_DIR.resolve()
        else run_dir / "train_manifest.json"
    )
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"rankers": [], "dqn": [], "invocations": []}
    manifest.setdefault("rankers", [])
    manifest.setdefault("dqn", [])
    manifest.setdefault("invocations", [])
    manifest["markets"] = sorted(set(manifest.get("markets", [])) | set(args.markets))
    manifest["years"] = sorted(set(manifest.get("years", [])) | set(args.years))
    manifest["n_games"] = args.n_games
    manifest["runtime"] = runtime_versions()
    manifest["invocations"].append({
        "models": sorted(selected),
        "markets": args.markets,
        "years": args.years,
        "seed_override": args.seed,
        "seed_config": str(args.seed_config.resolve()) if args.seed_config else "built-in",
        "rank_config": str(args.rank_config.resolve()) if args.rank_config else "built-in",
        "mart_config": str(args.mart_config.resolve()) if args.mart_config else "built-in",
        "n_games": args.n_games,
        "lr": args.lr,
        "gamma": args.gamma,
        "epsilon": args.epsilon,
        "eps_end": args.eps_end,
        "eps_dec": args.eps_dec,
        "batch_size": args.batch_size,
        "max_mem_size": args.max_mem_size,
        "replace_target_iter": args.replace_target_iter,
        "ranker_tree_method": args.ranker_tree_method,
        "effective_ranker_tree_method": effective_tree_method,
    })

    for market in args.markets:
        for year in args.years:
            rank_train_path = rankings_dir / f"{market}_{DQN_RANKER}_train{year}.csv"
            if train_rankers:
                for model_name in ("LambdaRank", "LambdaMART"):
                    rank_params = rank_config[MARKETS[market]][str(year)]
                    mart_params = mart_config[MARKETS[market]][str(year)]
                    model_seed = stage_seed(
                        MARKETS[market], year,
                        "rank" if model_name == "LambdaRank" else "mart",
                        seed_config, args.seed,
                    )
                    model, train_ranked, test_ranked = fit_ranker(
                        market, year, model_name, seed=model_seed,
                        tree_method=ranker_tree_method,
                        rank_max_depth=rank_params["max_depth"],
                        rank_n_estimators=rank_params["n_estimators"],
                        mart_max_bin=mart_params["max_bin"],
                        mart_min_child_weight=mart_params["min_child_weight"],
                    )
                    train_path = rankings_dir / f"{market}_{model_name}_train{year}.csv"
                    test_path = rankings_dir / f"{market}_{model_name}_test{year}.csv"
                    # Preserve the ranker's raw predictions. DQN consumes the
                    # same LambdaMART scores used by the ranking backtest.
                    train_ranked.to_csv(train_path, index=False)
                    test_ranked.to_csv(test_path, index=False)
                    record = {
                        "market": market,
                        "train_year": year,
                        "model": model_name,
                        "seed": model_seed,
                        "tree_method": effective_tree_method,
                        "paper_parameters": PAPER_HYPERPARAMETERS[model_name][MARKETS[market]],
                        "source_entrypoint_parameters": (
                            T4_MART_HYPERPARAMETERS[MARKETS[market]]
                            if model_name == "LambdaMART" and year == 3 else None
                        ),
                        "unreported_parameters": (
                            rank_params
                            if model_name == "LambdaRank" else mart_params
                        ),
                        "train_output": str(train_path),
                        "test_output": str(test_path),
                        "train_sha256": sha256(train_path),
                        "test_sha256": sha256(test_path),
                    }
                    upsert(manifest["rankers"], record, ("market", "train_year", "model"))

            if train_dqns:
                if not rank_train_path.is_file():
                    raise FileNotFoundError(
                        f"Missing {DQN_RANKER} ranking input: {rank_train_path}. "
                        "Run train.py with --models rankers first."
                    )
                model_path = models_dir / f"{market}_DQN_train{year}.pt"
                dqn_seed = stage_seed(MARKETS[market], year, "dqn", seed_config, args.seed)
                state_hash = train_dqn(
                    market,
                    year,
                    rank_train_path,
                    model_path,
                    lr=args.lr,
                    n_games=args.n_games,
                    seed=dqn_seed,
                    gamma=args.gamma,
                    epsilon=args.epsilon,
                    eps_end=args.eps_end,
                    eps_dec=args.eps_dec,
                    batch_size=args.batch_size,
                    max_mem_size=args.max_mem_size,
                    replace_target_iter=args.replace_target_iter,
                )
                record = {
                    "market": market,
                    "train_year": year,
                    "seed": dqn_seed,
                    "ranking_model": DQN_RANKER,
                    "ranking_input": str(rank_train_path),
                    "ranking_input_sha256": sha256(rank_train_path),
                    "model": str(model_path),
                    "sha256": sha256(model_path),
                    "model_state_sha256": state_hash,
                    "gamma": args.gamma,
                    "epsilon": args.epsilon,
                    "eps_end": args.eps_end,
                    "eps_dec": args.eps_dec,
                    "batch_size": args.batch_size,
                    "max_mem_size": args.max_mem_size,
                    "replace_target_iter": args.replace_target_iter,
                }
                upsert(manifest["dqn"], record, ("market", "train_year"))

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps({"run_dir": str(run_dir), "manifest": str(manifest_path)}, indent=2))

    if args.t6:
        from t6_core import run_sampling
        from workflow import evaluate_table4_primary

        if args.t6_markets.lower() == "all":
            t6_markets = ["Main", "ChiNext"]
        else:
            t6_markets = [item.strip() for item in args.t6_markets.split(",") if item.strip()]
            invalid = sorted(set(t6_markets) - {"Main", "ChiNext"})
            if invalid:
                raise ValueError(f"Unknown T6 markets: {invalid}")
        seed_path = args.t6_seed_summary or (
            CODE_DIR / "data" / "reproducibility" / "t6_cpu20_seed_summary.csv"
        )
        if not seed_path.is_file():
            raise FileNotFoundError(f"T6 seed summary not found: {seed_path}")
        dqn_seed_path = args.t6_dqn_seed_summary or (
            CODE_DIR / "data" / "reproducibility" / "t6_cpu20_dqn_seed_summary.csv"
        )
        if not dqn_seed_path.is_file():
            raise FileNotFoundError(
                f"T6 DQN seed summary not found: {dqn_seed_path}. "
                "Run make_dqn_seed_summary.py first."
            )
        # Always regenerate the action map from the freshly trained DQN.  A
        # checked-in or user-supplied map would be an intermediate-result
        # shortcut and is intentionally not part of the reproduction API.
        select_path = generate_t6_select_map(
            run_dir, t6_markets, seed_config, args.seed
        )
        if not select_path.is_file():
            raise FileNotFoundError(
                f"T6 daily action map was not generated: {select_path}"
            )
        t6_run_dir = CODE_DIR / "temp" / "t6_runs" if run_dir == CODE_DIR.resolve() else run_dir / "t6_runs"
        t6_run_dir.mkdir(parents=True, exist_ok=True)
        t4_reference_path = t6_run_dir / "t4_primary_reference.csv"
        t4_rows = []
        for market in t6_markets:
            metrics_by_model = evaluate_table4_primary(
                run_dir, market, seed_config, args.seed
            )
            for model_name, metrics in metrics_by_model.items():
                stage = "dqn" if model_name == "LTR-DQN" else (
                    "rank" if model_name == "LambdaRank" else "mart"
                )
                row = {
                    "market": market, "model": model_name,
                    "ARR": metrics["ARR"],
                    "seed": stage_seed(
                        MARKETS[market], 3, stage, seed_config, args.seed
                    ),
                }
                if model_name == "LTR-DQN":
                    row["dqn_seed"] = row["seed"]
                t4_rows.append(row)
        __import__("pandas").DataFrame(t4_rows).to_csv(
            t4_reference_path, index=False, float_format="%.17g"
        )
        t6_output = t6_run_dir / "t6_raw.csv"
        raw = run_sampling(
            data_dir=CODE_DIR / "data", seed_path=seed_path,
            select_map_path=select_path, output_path=t6_output,
            markets=t6_markets, max_seeds=args.t6_max_seeds,
            # Recompute every cell from the raw data on each invocation.  A
            # prior t6_raw.csv is never used as a shortcut.
            use_gpu=False, resume=False,
            dqn_seed_path=dqn_seed_path, require_gpu=False,
            full_rate_path=t4_reference_path,
        )
        t6_manifest = {
            "markets": t6_markets,
            "max_seeds_per_cell": args.t6_max_seeds,
            "seed_summary": str(seed_path),
            "seed_summary_sha256": sha256(seed_path),
            "dqn_seed_summary": str(dqn_seed_path),
            "dqn_seed_summary_sha256": sha256(dqn_seed_path),
            "require_gpu": False,
            "select_map": str(select_path),
            "select_map_sha256": sha256(select_path),
            "full_rate_source": "same_run_T4",
            "t4_reference": str(t4_reference_path),
            "t4_reference_sha256": sha256(t4_reference_path),
            "raw_csv": str(t6_output),
            "raw_csv_sha256": sha256(t6_output),
            "rows": len(raw),
            "sampling_rates": ["100%", "50%", "60%", "70%", "80%", "90%"],
            "resume": False,
        }
        t6_manifest_path = t6_output.with_name("t6_manifest.json")
        t6_manifest_path.write_text(
            json.dumps(t6_manifest, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        print(json.dumps({"t6_raw": str(t6_output), "t6_manifest": str(t6_manifest_path), "t6_rows": len(raw)}, indent=2))


if __name__ == "__main__":
    main()
