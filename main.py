"""Generate the paper tables and all non-T6 empirical figures."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export T3/T4/T5/T7 and the empirical paper figures"
    )
    parser.add_argument("--run_dir", type=Path, default=CODE_DIR)
    parser.add_argument("--tables", default="T3,T4,T5,T7")
    parser.add_argument("--export_csvs", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seed_config", type=Path, default=None)
    parser.add_argument(
        "--ranker_tree_method",
        choices=["auto", "hist", "exact", "approx", "gpu_hist"],
        default="auto",
    )
    parser.add_argument("--no_baselines", action="store_true")
    parser.add_argument(
        "--dqn_eval_mode", choices=["dqn", "fixed"], default="dqn"
    )
    parser.add_argument("--figures", default="3,4,5,6,7")
    parser.add_argument("--appendix_figures", default="C1,C2,C3,C5")
    parser.add_argument("--force_figures", action="store_true")
    parser.add_argument(
        "--skip_figures",
        action="store_true",
        help="Generate tables only; useful for a quick workflow check",
    )
    return parser.parse_args()


def run_stage(label: str, script: str, arguments: list[str]) -> None:
    command = [sys.executable, str(CODE_DIR / script), *arguments]
    print(f"\n== {label} ==", flush=True)
    subprocess.run(command, cwd=CODE_DIR, check=True)


def shared_arguments(args: argparse.Namespace) -> list[str]:
    result = ["--run_dir", str(args.run_dir.resolve())]
    if args.seed is not None:
        result.extend(["--seed", str(args.seed)])
    if args.seed_config is not None:
        result.extend(["--seed_config", str(args.seed_config.resolve())])
    return result


def main() -> None:
    args = parse_args()
    common = shared_arguments(args)
    table_args = [*common, "--tables", args.tables, "--dqn_eval_mode", args.dqn_eval_mode]
    if args.export_csvs:
        table_args.append("--export_csvs")
    if args.no_baselines:
        table_args.append("--no_baselines")
    run_stage("Tables T3/T4/T5/T7", "workflow.py", table_args)

    if args.skip_figures:
        return

    figure_args = [
        *common,
        "--figures",
        args.figures,
        "--ranker_tree_method",
        args.ranker_tree_method,
    ]
    if args.force_figures:
        figure_args.append("--force")
    run_stage("Main-text Figures 3-7", "Fig_main.py", figure_args)

    appendix_args = [
        "--run_dir",
        str(args.run_dir.resolve()),
        "--figures",
        args.appendix_figures,
    ]
    if args.force_figures:
        appendix_args.append("--force")
    run_stage("Appendix Figures C1/C2/C3/C5", "Appendix_Fig_main.py", appendix_args)


if __name__ == "__main__":
    main()
