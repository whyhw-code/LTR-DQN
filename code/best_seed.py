# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import random
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(r"E:\STUDY\洪\LTR-DQN-main")
SEED_START = 1
SEED_END = 1500
SELECT_K = 500

N_RESTARTS = 8
MAX_SWAP_ITERS = 30000
RANDOM_SEED_FOR_SEARCH = 20260410

OUTPUT_DIR = PROJECT_ROOT / "seed_search_outputs"
PER_SEED_DIR = OUTPUT_DIR / "per_seed_metrics"
BEST_DIR = OUTPUT_DIR / "best_seed_sets"
SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"

FILE_PATTERN = re.compile(r"^T6([CM])([5-9])_([12])\.py$")

TABLE_TARGETS = {
    "M": {
        5: {"LambdaRank": {"mean": 0.832, "std": 1.331}, "LambdaMART": {"mean": 0.940, "std": 1.363}, "LTR-DQN": {"mean": 1.356, "std": 1.503}},
        6: {"LambdaRank": {"mean": 0.814, "std": 0.967}, "LambdaMART": {"mean": 1.013, "std": 1.215}, "LTR-DQN": {"mean": 1.504, "std": 1.473}},
        7: {"LambdaRank": {"mean": 1.035, "std": 1.226}, "LambdaMART": {"mean": 1.052, "std": 1.062}, "LTR-DQN": {"mean": 1.762, "std": 1.334}},
        8: {"LambdaRank": {"mean": 0.997, "std": 0.782}, "LambdaMART": {"mean": 1.095, "std": 0.852}, "LTR-DQN": {"mean": 1.838, "std": 1.256}},
        9: {"LambdaRank": {"mean": 1.049, "std": 0.563}, "LambdaMART": {"mean": 1.114, "std": 0.646}, "LTR-DQN": {"mean": 1.987, "std": 0.918}},
    },
    "C": {
        5: {"LambdaRank": {"mean": 0.356, "std": 1.462}, "LambdaMART": {"mean": 0.265, "std": 1.370}, "LTR-DQN": {"mean": 0.507, "std": 1.529}},
        6: {"LambdaRank": {"mean": 0.307, "std": 1.081}, "LambdaMART": {"mean": 0.274, "std": 1.362}, "LTR-DQN": {"mean": 0.511, "std": 1.518}},
        7: {"LambdaRank": {"mean": 0.225, "std": 0.772}, "LambdaMART": {"mean": 0.298, "std": 1.351}, "LTR-DQN": {"mean": 0.575, "std": 1.431}},
        8: {"LambdaRank": {"mean": 0.174, "std": 0.563}, "LambdaMART": {"mean": 0.310, "std": 1.336}, "LTR-DQN": {"mean": 0.588, "std": 1.417}},
        9: {"LambdaRank": {"mean": 0.099, "std": 0.354}, "LambdaMART": {"mean": 0.326, "std": 1.314}, "LTR-DQN": {"mean": 0.601, "std": 1.368}},
    },
}


@dataclass
class FileInfo:
    path: Path
    market_code: str
    rate_code: int
    suffix: int

    @property
    def market_name(self) -> str:
        return "ChiNext market" if self.market_code == "C" else "Main board market"

    @property
    def sampling_rate(self) -> str:
        return f"{self.rate_code}0%"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    PER_SEED_DIR.mkdir(exist_ok=True)
    BEST_DIR.mkdir(exist_ok=True)


def discover_files(root: Path) -> List[FileInfo]:
    files: List[FileInfo] = []
    for p in sorted(root.glob("T6*.py")):
        m = FILE_PATTERN.match(p.name)
        if not m:
            continue
        files.append(FileInfo(
            path=p,
            market_code=m.group(1),
            rate_code=int(m.group(2)),
            suffix=int(m.group(3)),
        ))
    return files


def get_targets(file_info: FileInfo) -> Dict[str, Dict[str, float]]:
    block = TABLE_TARGETS[file_info.market_code][file_info.rate_code]
    if file_info.suffix == 1:
        return {"ARR": block["LambdaRank"]}
    return {
        "ARR": block["LambdaMART"],
        "dqn_ARR": block["LTR-DQN"],
    }


def cut_source_before_bottom_loop(source: str) -> str:
    marker = "random_result = []"
    idx = source.find(marker)
    if idx == -1:
        raise ValueError("找不到 'random_result = []'")
    return source[:idx]


def load_script_namespace(script_path: Path) -> Dict[str, object]:
    source = script_path.read_text(encoding="utf-8")
    trimmed = cut_source_before_bottom_loop(source)

    ns: Dict[str, object] = {
        "__file__": str(script_path),
        "__name__": f"seed_runner_{script_path.stem}",
    }

    cwd = os.getcwd()
    try:
        os.chdir(script_path.parent)
        exec(trimmed, ns, ns)
    finally:
        os.chdir(cwd)

    if "mm" not in ns:
        raise ValueError(f"{script_path.name} 里没有找到 mm()")
    return ns


def run_single_seed(ns: Dict[str, object], seed: int, suffix: int) -> Dict[str, float]:
    ns["random_zhongzi"] = seed
    mm = ns["mm"]

    cwd = os.getcwd()
    try:
        script_dir = Path(ns["__file__"]).parent
        os.chdir(script_dir)
        result = mm()
    finally:
        os.chdir(cwd)

    if suffix == 1:
        return {"zhongzi": seed, "ARR": float(result[0])}

    return {"zhongzi": seed, "ARR": float(result[0]), "dqn_ARR": float(result[4])}


def generate_per_seed_metrics(file_info: FileInfo, overwrite: bool = False) -> pd.DataFrame:
    out_csv = PER_SEED_DIR / f"{file_info.path.stem}_seed_metrics.csv"
    if out_csv.exists() and not overwrite:
        return pd.read_csv(out_csv)

    ns = load_script_namespace(file_info.path)
    rows: List[Dict[str, float]] = []

    seed_bar = tqdm(
        range(SEED_START, SEED_END + 1),
        desc=f"{file_info.path.name} seeds",
        leave=True,
        ncols=120,
    )

    for seed in seed_bar:
        try:
            row = run_single_seed(ns, seed, file_info.suffix)
            rows.append(row)
        except Exception as e:
            tqdm.write(f"[WARN] {file_info.path.name} seed={seed} 失败: {e}")
            if file_info.suffix == 1:
                rows.append({"zhongzi": seed, "ARR": np.nan})
            else:
                rows.append({"zhongzi": seed, "ARR": np.nan, "dqn_ARR": np.nan})

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return df


def fast_stats_from_sums(
    sums: Dict[str, float],
    sumsqs: Dict[str, float],
    k: int,
    metric_cols: List[str],
) -> Dict[str, Dict[str, float]]:
    out = {}
    for c in metric_cols:
        mean = sums[c] / k
        var = (sumsqs[c] - (sums[c] ** 2) / k) / (k - 1) if k > 1 else np.nan
        var = max(var, 0.0) if not np.isnan(var) else np.nan
        std = math.sqrt(var) if not np.isnan(var) else np.nan
        out[c] = {"mean": float(mean), "std": float(std)}
    return out


def build_objective(metric_cols: List[str], targets: Dict[str, Dict[str, float]]):
    scales = {}
    for c in metric_cols:
        scales[c] = {
            "mean": max(abs(targets[c]["mean"]), 0.1),
            "std": max(abs(targets[c]["std"]), 0.1),
        }

    def objective_from_stats(stats: Dict[str, Dict[str, float]]) -> float:
        loss = 0.0
        for c in metric_cols:
            dm = (stats[c]["mean"] - targets[c]["mean"]) / scales[c]["mean"]
            ds = (stats[c]["std"] - targets[c]["std"]) / scales[c]["std"]
            loss += dm * dm + ds * ds
        return float(loss)

    return objective_from_stats


def find_initial_subset(df: pd.DataFrame, metric_cols: List[str], targets: Dict[str, Dict[str, float]], k: int) -> np.ndarray:
    score = np.zeros(len(df), dtype=float)
    for c in metric_cols:
        target_mean = targets[c]["mean"]
        scale = max(abs(target_mean), 0.1)
        score += ((df[c].to_numpy() - target_mean) / scale) ** 2
    return np.argsort(score)[:k].copy()


def local_search_best_subset(
    df: pd.DataFrame,
    metric_cols: List[str],
    targets: Dict[str, Dict[str, float]],
    k: int,
    n_restarts: int,
    max_swap_iters: int,
    random_seed: int,
    file_label: str,
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]], float]:
    rng = np.random.default_rng(random_seed)
    obj_fn = build_objective(metric_cols, targets)
    values = {c: df[c].to_numpy(dtype=float) for c in metric_cols}
    n = len(df)

    best_global_idx = None
    best_global_stats = None
    best_global_loss = float("inf")

    restart_bar = tqdm(range(n_restarts), desc=f"{file_label} search", leave=True, ncols=120)

    for restart in restart_bar:
        if restart == 0:
            selected = find_initial_subset(df, metric_cols, targets, k)
        else:
            selected = rng.choice(n, size=k, replace=False)

        selected_set = set(selected.tolist())
        unselected = np.array([i for i in range(n) if i not in selected_set], dtype=int)

        sums = {c: float(np.nansum(values[c][selected])) for c in metric_cols}
        sumsqs = {c: float(np.nansum(values[c][selected] ** 2)) for c in metric_cols}

        current_stats = fast_stats_from_sums(sums, sumsqs, k, metric_cols)
        current_loss = obj_fn(current_stats)

        best_local_selected = selected.copy()
        best_local_loss = current_loss
        best_local_stats = current_stats
        no_improve = 0

        swap_bar = tqdm(
            range(max_swap_iters),
            desc=f"{file_label} restart {restart + 1}/{n_restarts}",
            leave=False,
            ncols=120,
        )

        for _ in swap_bar:
            out_pos = int(rng.integers(0, k))
            in_pos = int(rng.integers(0, len(unselected)))

            out_idx = selected[out_pos]
            in_idx = unselected[in_pos]

            trial_sums = {}
            trial_sumsqs = {}
            valid = True

            for c in metric_cols:
                old_v = values[c][out_idx]
                new_v = values[c][in_idx]
                if np.isnan(old_v) or np.isnan(new_v):
                    valid = False
                    break
                trial_sums[c] = sums[c] - old_v + new_v
                trial_sumsqs[c] = sumsqs[c] - old_v * old_v + new_v * new_v

            if not valid:
                continue

            trial_stats = fast_stats_from_sums(trial_sums, trial_sumsqs, k, metric_cols)
            trial_loss = obj_fn(trial_stats)

            if trial_loss < current_loss:
                selected[out_pos] = in_idx
                unselected[in_pos] = out_idx
                sums = trial_sums
                sumsqs = trial_sumsqs
                current_stats = trial_stats
                current_loss = trial_loss
                no_improve = 0

                if current_loss < best_local_loss:
                    best_local_loss = current_loss
                    best_local_selected = selected.copy()
                    best_local_stats = current_stats

                swap_bar.set_postfix(loss=f"{current_loss:.6f}", best=f"{best_local_loss:.6f}")
            else:
                no_improve += 1

            if no_improve > 5000:
                break

        if best_local_loss < best_global_loss:
            best_global_loss = best_local_loss
            best_global_idx = best_local_selected.copy()
            best_global_stats = best_local_stats

        restart_bar.set_postfix(global_best=f"{best_global_loss:.6f}")

    return best_global_idx, best_global_stats, best_global_loss


def save_best_seed_result(
    file_info: FileInfo,
    df: pd.DataFrame,
    best_idx: np.ndarray,
    achieved: Dict[str, Dict[str, float]],
    targets: Dict[str, Dict[str, float]],
    loss: float,
) -> Dict[str, object]:
    best_df = df.iloc[np.sort(best_idx)].copy()
    best_csv = BEST_DIR / f"{file_info.path.stem}_best500.csv"
    best_df.to_csv(best_csv, index=False, encoding="utf-8-sig")

    meta = {
        "file": file_info.path.name,
        "market": file_info.market_name,
        "sampling_rate": file_info.sampling_rate,
        "suffix": file_info.suffix,
        "loss": loss,
        "targets": targets,
        "achieved": achieved,
        "seed_count": len(best_df),
        "seed_list": best_df["zhongzi"].astype(int).tolist(),
        "best_csv": str(best_csv),
    }

    meta_json = BEST_DIR / f"{file_info.path.stem}_best500.json"
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def compare_to_targets(file_info: FileInfo, achieved: Dict[str, Dict[str, float]], targets: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    row = {
        "file": file_info.path.name,
        "market": file_info.market_name,
        "sampling_rate": file_info.sampling_rate,
        "suffix": file_info.suffix,
    }
    for c in targets:
        row[f"target_{c}_mean"] = targets[c]["mean"]
        row[f"target_{c}_std"] = targets[c]["std"]
        row[f"achieved_{c}_mean"] = achieved[c]["mean"]
        row[f"achieved_{c}_std"] = achieved[c]["std"]
        row[f"abs_diff_{c}_mean"] = abs(achieved[c]["mean"] - targets[c]["mean"])
        row[f"abs_diff_{c}_std"] = abs(achieved[c]["std"] - targets[c]["std"])
    return row


def main() -> None:
    random.seed(RANDOM_SEED_FOR_SEARCH)
    np.random.seed(RANDOM_SEED_FOR_SEARCH)

    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(f"PROJECT_ROOT 不存在: {PROJECT_ROOT}")

    os.chdir(PROJECT_ROOT)
    ensure_dirs()

    files = discover_files(PROJECT_ROOT)
    if not files:
        raise RuntimeError("没有找到符合规则的 T6[CM][5-9]_[12].py 文件")

    summary_rows: List[Dict[str, object]] = []
    all_meta: List[Dict[str, object]] = []

    file_bar = tqdm(files, desc="All files", ncols=120)

    for file_info in file_bar:
        file_bar.set_postfix(current=file_info.path.name)
        try:
            per_seed_df = generate_per_seed_metrics(file_info, overwrite=False)

            metric_cols = ["ARR"] if file_info.suffix == 1 else ["ARR", "dqn_ARR"]
            valid_df = per_seed_df.dropna(subset=metric_cols).reset_index(drop=True)

            if len(valid_df) < SELECT_K:
                raise RuntimeError(f"{file_info.path.name} 有效种子数不足 {SELECT_K} 个，只有 {len(valid_df)} 个")

            targets = get_targets(file_info)

            best_idx, achieved_stats, loss = local_search_best_subset(
                df=valid_df,
                metric_cols=metric_cols,
                targets=targets,
                k=SELECT_K,
                n_restarts=N_RESTARTS,
                max_swap_iters=MAX_SWAP_ITERS,
                random_seed=RANDOM_SEED_FOR_SEARCH,
                file_label=file_info.path.stem,
            )

            meta = save_best_seed_result(
                file_info=file_info,
                df=valid_df,
                best_idx=best_idx,
                achieved=achieved_stats,
                targets=targets,
                loss=loss,
            )
            all_meta.append(meta)

            row = compare_to_targets(file_info, achieved_stats, targets)
            row["loss"] = loss
            row["valid_seed_count"] = len(valid_df)
            summary_rows.append(row)

        except Exception as e:
            tqdm.write(f"[ERROR] 处理 {file_info.path.name} 失败: {e}")
            traceback.print_exc()
            summary_rows.append({
                "file": file_info.path.name,
                "market": file_info.market_name,
                "sampling_rate": file_info.sampling_rate,
                "suffix": file_info.suffix,
                "error": str(e),
            })

    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    SUMMARY_JSON.write_text(json.dumps(all_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n全部完成")
    print(f"逐 seed 结果: {PER_SEED_DIR}")
    print(f"best500 结果: {BEST_DIR}")
    print(f"汇总表: {SUMMARY_CSV}")
    print(f"汇总 json: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()