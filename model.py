"""Shared experiment and table-generation API for the T3/T4/T5/T7 entry points."""

from __future__ import annotations

import json
from pathlib import Path

# Public model classes for callers that want to inspect or reuse the DQN
# implementation. Their behavior remains implemented in dl_dqn2.py.
from dl_dqn2 import Agent, DeepQNetwork, Environment

from experiment_core import (
    CODE_DIR,
    DQN_RANKER,
    MARKETS,
    PAPER_HYPERPARAMETERS,
    T4_MART_HYPERPARAMETERS,
    artifact_dir,
    canonicalize_ranking,
    dqn_ranking_path,
    evaluate_dqn,
    fit_ranker,
    runtime_versions,
    sha256,
    train_dqn,
    validate_runtime,
    write_results,
)
from workflow import (
    run_t3,
    run_table4,
    run_table5,
    run_table7,
    selected_markets,
)
from runtime_config import load_stage_seed_config, set_global_determinism


def generate_table(
    table: str,
    *,
    run_dir: Path,
    markets: str = "all",
    output_dir: Path | None = None,
    seed: int | None = None,
    seed_config: Path | None = None,
    include_baselines: bool = True,
    dqn_eval_mode: str = "dqn",
) -> dict:
    """Evaluate one table and write its long CSV, paper CSV and workbook."""
    validate_runtime()
    table = table.upper()
    if table not in {"T3", "T4", "T5", "T7"}:
        raise ValueError("table must be one of T3, T4, T5 or T7; T6 is separate")
    if dqn_eval_mode != "dqn":
        raise ValueError("DQN result generation must use the trained policy; fixed action maps are disabled")
    seed_map = load_stage_seed_config(seed_config)
    if seed is not None:
        set_global_determinism(seed)
    run_dir = run_dir.resolve()
    selected = selected_markets(markets)
    output_dir = (output_dir or artifact_dir(run_dir, "results") / table).resolve()
    records: list[dict] = []
    if table == "T3":
        run_t3(records, selected)
    elif table == "T4":
        run_table4(
            records, run_dir, selected, include_baselines, seed_map, seed,
            dqn_fixed_actions=False,
        )
    elif table == "T5":
        run_table5(
            records, run_dir, selected, include_baselines, seed_map, seed,
            dqn_fixed_actions=False,
        )
    else:
        run_table7(records, run_dir, selected, seed_map, seed)
    paper_csvs = write_results(
        records,
        output_dir / f"{table}_results_long.csv",
        output_dir / f"{table}.xlsx",
    )
    manifest = {
        "table": table,
        "run_dir": str(run_dir),
        "markets": selected,
        "seed": seed,
        "seed_config": str(seed_config.resolve()) if seed_config else "built-in",
        "dqn_evaluation_mode": dqn_eval_mode,
        "runtime": runtime_versions(),
        "long_csv": str(output_dir / f"{table}_results_long.csv"),
        "paper_csv": paper_csvs.get(table),
        "workbook": str(output_dir / f"{table}.xlsx"),
        "include_baselines": include_baselines,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "main_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return manifest


def default_run_dir() -> Path:
    return CODE_DIR
