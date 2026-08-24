"""Run one deterministic Table 6 seed shard for GitHub Actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_core import CODE_DIR, runtime_versions, sha256, validate_runtime
from runtime_config import load_stage_seed_config
from t6_core import run_sampling
from train import generate_t6_select_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one deterministic T6 seed shard")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--shard_index", type=int, required=True)
    parser.add_argument("--shard_count", type=int, required=True)
    parser.add_argument("--markets", default="all")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seed_config", type=Path, default=None)
    parser.add_argument(
        "--ranker_tree_method",
        choices=["hist", "gpu_hist"],
        default="hist",
    )
    return parser.parse_args()


def selected_markets(value: str) -> list[str]:
    if value.lower() == "all":
        return ["Main", "ChiNext"]
    markets = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(markets) - {"Main", "ChiNext"})
    if invalid:
        raise ValueError(f"Unknown T6 markets: {invalid}")
    return markets


def main() -> None:
    args = parse_args()
    validate_runtime()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")

    run_dir = args.run_dir.resolve()
    markets = selected_markets(args.markets)
    seed_config = load_stage_seed_config(args.seed_config)
    seed_path = CODE_DIR / "data" / "reproducibility" / "seed_summary.csv"
    dqn_seed_path = CODE_DIR / "data" / "reproducibility" / "dqn_seed_summary.csv"
    select_path = generate_t6_select_map(run_dir, markets, seed_config, args.seed)
    output_path = run_dir / "t6_runs" / "t6_raw.csv"

    raw = run_sampling(
        data_dir=CODE_DIR / "data",
        seed_path=seed_path,
        select_map_path=select_path,
        output_path=output_path,
        markets=markets,
        use_gpu=args.ranker_tree_method == "gpu_hist",
        resume=True,
        include_full_rate=args.shard_index == 0,
        dqn_seed_path=dqn_seed_path,
        require_gpu=args.ranker_tree_method == "gpu_hist",
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )

    manifest = {
        "runtime": runtime_versions(),
        "markets": markets,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "ranker_tree_method": args.ranker_tree_method,
        "seed_summary": str(seed_path),
        "seed_summary_sha256": sha256(seed_path),
        "dqn_seed_summary": str(dqn_seed_path),
        "dqn_seed_summary_sha256": sha256(dqn_seed_path),
        "select_map": str(select_path),
        "select_map_sha256": sha256(select_path),
        "raw_csv": str(output_path),
        "raw_csv_sha256": sha256(output_path),
        "rows": len(raw),
    }
    manifest_path = output_path.with_name("t6_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
