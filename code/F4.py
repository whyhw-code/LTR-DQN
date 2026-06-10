# F4.py
import argparse
import re
import sys
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 基础配置
# ============================================================

# 当前代码位于 LTR-DQN-main/code/
CODE_DIR = Path(__file__).resolve().parent

# F4 横轴
LEARNING_RATES = [0.0001, 0.001, 0.002, 0.01, 0.1, 0.2]
N_ESTIMATORS_LIST = [800, 900, 1000, 1100, 1200]
MAX_DEPTH_LIST = [4, 5, 6, 7, 8]

# 写入 Excel 时的行顺序，匹配 F3F4.xlsx：
# A2 = C，表示 ChiNext market
# A3 = M，表示 Main board market
F4_ROW_ORDER = ["C", "M"]

F4_SCRIPTS = {
    "M": "T4M11.py",
    "C": "T4C11.py",
}

# LambdaMART 基准参数。
# F4a：只变 learning_rate，其余参数用这里的 max_depth / n_estimators
# F4b：只变 n_estimators，其余参数用这里的 learning_rate / max_depth
# F4c：只变 max_depth，其余参数用这里的 learning_rate / n_estimators
F4_BASE_PARAMS = {
    "M": {
        "learning_rate": 0.001,
        "max_depth": 8,
        "n_estimators": 1000,
        "shouxufei": 0.0003,
        "yinhaushui": 0.001,
    },
    "C": {
        "learning_rate": 0.1,
        "max_depth": 4,
        "n_estimators": 1100,
        "shouxufei": 0.0003,
        "yinhaushui": 0.001,
    },
}


# ============================================================
# 命令行参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_batch", type=int, default=123)
    parser.add_argument("--train_year", type=int, default=3)

    # 默认：不调用 T4M11/T4C11，直接读取已有 F3F4.xlsx 画图
    # 需要重新计算时，手动加 --run-analysis
    parser.add_argument(
        "--run-analysis",
        dest="run_analysis",
        action="store_true",
        help="调用 T4M11/T4C11 重新计算 F4a/F4b/F4c"
    )
    parser.set_defaults(run_analysis=False)

    # 默认：不写回 F3F4.xlsx
    # 需要写回时，手动加 --write-excel
    parser.add_argument(
        "--write-excel",
        dest="write_excel",
        action="store_true",
        help="把 F4a/F4b/F4c 结果写回 F3F4.xlsx"
    )
    parser.set_defaults(write_excel=False)

    # 默认：画图
    parser.add_argument(
        "--no-plot",
        dest="plot_figure",
        action="store_false",
        help="不画图，只计算/写入 Excel"
    )
    parser.set_defaults(plot_figure=True)

    # 默认：不保存图片，只 plt.show()
    parser.add_argument(
        "--save-fig",
        action="store_true",
        help="保存 F4a.png、F4b.png、F4c.png 到 result/batchxxx/"
    )

    # 默认：不输出中间过程
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印每次调用 T4 脚本的命令和 ARR"
    )

    # 默认：遇到错误直接停止
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="某个参数运行失败时写 NaN 并继续后续参数"
    )

    return parser.parse_args()


# ============================================================
# 工具函数
# ============================================================

def set_seed(seed=1795):
    random.seed(seed)
    np.random.seed(seed)


def fmt_num(x):
    if isinstance(x, int):
        return str(x)

    xf = float(x)
    if xf.is_integer():
        return str(int(xf))

    return f"{xf:.10f}".rstrip("0").rstrip(".")


def check_script_exists(script_name):
    script_path = CODE_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(
            f"找不到脚本：{script_path}\n"
            f"请确认 {script_name} 位于 LTR-DQN-main/code/ 目录下。"
        )
    return script_path


def parse_arr_from_output(output):
    patterns = [
        r"年化收益率\s*\(ARR\)\s*[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"\(\s*ARR\s*\)\s*[:：]\s*([-+]?\d+(?:\.\d+)?)",
        r"ARR\s*[:：]\s*([-+]?\d+(?:\.\d+)?)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, output)
        if matches:
            return float(matches[-1])

    return np.nan


def run_python_script(script_path, script_args, verbose=False, continue_on_error=False):
    cmd = [sys.executable, str(script_path)] + list(script_args)

    proc = subprocess.run(
        cmd,
        cwd=CODE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )

    output = proc.stdout

    if proc.returncode != 0:
        message = (
            f"{script_path.name} 运行失败，returncode={proc.returncode}.\n"
            f"命令：{' '.join(cmd)}\n"
            f"末尾输出：\n{output[-2000:]}"
        )

        if continue_on_error:
            if verbose:
                print(message)
            return np.nan

        raise RuntimeError(message)

    arr = parse_arr_from_output(output)

    if pd.isna(arr):
        message = (
            f"没有从 {script_path.name} 的输出中解析到 ARR。\n"
            f"命令：{' '.join(cmd)}\n"
            f"末尾输出：\n{output[-2000:]}"
        )

        if continue_on_error:
            if verbose:
                print(message)
            return np.nan

        raise RuntimeError(message)

    if verbose:
        print(f"{script_path.name} {' '.join(script_args)} -> ARR={arr}")

    return arr


def reorder_df(df, col_order):
    for row in F4_ROW_ORDER:
        if row not in df.index:
            df.loc[row] = np.nan

    for col in col_order:
        if col not in df.columns:
            df[col] = np.nan

    return df.loc[F4_ROW_ORDER, col_order]


def make_t4_args(
    args,
    base_params,
    learning_rate,
    max_depth,
    n_estimators
):
    return [
        "--train_or_test", "test",
        "--train_year", str(args.train_year),
        "--test_batch", str(args.test_batch),
        "--shouxufei", fmt_num(base_params["shouxufei"]),
        "--yinhaushui", fmt_num(base_params["yinhaushui"]),
        "--learning_rate", fmt_num(learning_rate),
        "--max_depth", str(max_depth),
        "--n_estimators", str(n_estimators),
    ]


# ============================================================
# F4a / F4b / F4c：调用 T4M11.py / T4C11.py
# ============================================================

def run_f4a_lambdamart_lr_sensitivity(args):
    result = {}

    for row_name in F4_ROW_ORDER:
        script_path = check_script_exists(F4_SCRIPTS[row_name])
        base_params = F4_BASE_PARAMS[row_name]
        arr_values = []

        for lr_value in LEARNING_RATES:
            script_args = make_t4_args(
                args=args,
                base_params=base_params,
                learning_rate=lr_value,
                max_depth=base_params["max_depth"],
                n_estimators=base_params["n_estimators"],
            )

            arr = run_python_script(
                script_path=script_path,
                script_args=script_args,
                verbose=args.verbose,
                continue_on_error=args.continue_on_error,
            )
            arr_values.append(arr)

        result[row_name] = arr_values

    df = pd.DataFrame(
        result,
        index=[fmt_num(x) for x in LEARNING_RATES]
    ).T

    return reorder_df(df, [fmt_num(x) for x in LEARNING_RATES])


def run_f4b_weak_learners_sensitivity(args):
    result = {}

    for row_name in F4_ROW_ORDER:
        script_path = check_script_exists(F4_SCRIPTS[row_name])
        base_params = F4_BASE_PARAMS[row_name]
        arr_values = []

        for n_estimators in N_ESTIMATORS_LIST:
            script_args = make_t4_args(
                args=args,
                base_params=base_params,
                learning_rate=base_params["learning_rate"],
                max_depth=base_params["max_depth"],
                n_estimators=n_estimators,
            )

            arr = run_python_script(
                script_path=script_path,
                script_args=script_args,
                verbose=args.verbose,
                continue_on_error=args.continue_on_error,
            )
            arr_values.append(arr)

        result[row_name] = arr_values

    df = pd.DataFrame(
        result,
        index=[fmt_num(x) for x in N_ESTIMATORS_LIST]
    ).T

    return reorder_df(df, [fmt_num(x) for x in N_ESTIMATORS_LIST])


def run_f4c_max_depth_sensitivity(args):
    result = {}

    for row_name in F4_ROW_ORDER:
        script_path = check_script_exists(F4_SCRIPTS[row_name])
        base_params = F4_BASE_PARAMS[row_name]
        arr_values = []

        for max_depth in MAX_DEPTH_LIST:
            script_args = make_t4_args(
                args=args,
                base_params=base_params,
                learning_rate=base_params["learning_rate"],
                max_depth=max_depth,
                n_estimators=base_params["n_estimators"],
            )

            arr = run_python_script(
                script_path=script_path,
                script_args=script_args,
                verbose=args.verbose,
                continue_on_error=args.continue_on_error,
            )
            arr_values.append(arr)

        result[row_name] = arr_values

    df = pd.DataFrame(
        result,
        index=[fmt_num(x) for x in MAX_DEPTH_LIST]
    ).T

    return reorder_df(df, [fmt_num(x) for x in MAX_DEPTH_LIST])


# ============================================================
# 写入 F3F4.xlsx
# ============================================================

def get_result_dir(test_batch):
    return (CODE_DIR / f"../result/batch{test_batch}").resolve()


def get_f3f4_path(test_batch):
    return get_result_dir(test_batch) / "F3F4.xlsx"


def write_f4_results_to_excel(f4a_df, f4b_df, f4c_df, file_path):
    file_path.parent.mkdir(parents=True, exist_ok=True)

    f4a_df = reorder_df(f4a_df, [fmt_num(x) for x in LEARNING_RATES])
    f4b_df = reorder_df(f4b_df, [fmt_num(x) for x in N_ESTIMATORS_LIST])
    f4c_df = reorder_df(f4c_df, [fmt_num(x) for x in MAX_DEPTH_LIST])

    if file_path.exists():
        with pd.ExcelWriter(
            file_path,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace"
        ) as writer:
            f4a_df.to_excel(writer, sheet_name="F4a")
            f4b_df.to_excel(writer, sheet_name="F4b")
            f4c_df.to_excel(writer, sheet_name="F4c")
    else:
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="w") as writer:
            f4a_df.to_excel(writer, sheet_name="F4a")
            f4b_df.to_excel(writer, sheet_name="F4b")
            f4c_df.to_excel(writer, sheet_name="F4c")


# ============================================================
# 画图
# ============================================================

def read_f4_sheet(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)

    # 只保留 M 和 C，不读取 M1、C1
    df = df.loc[["M", "C"]]

    x = [str(i) for i in df.columns]
    y_main = df.loc["M"].values
    y_chinext = df.loc["C"].values

    return x, y_main, y_chinext


def plot_f4(file_path, sheet_name, xlabel, save_path=None):
    x, y_main, y_chinext = read_f4_sheet(file_path, sheet_name)

    fig, ax = plt.subplots(figsize=(4.6, 2.8), dpi=120)

    # 白色背景 + 灰色横向网格线
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    ax.plot(
        x,
        y_main,
        color="#4472C4",
        marker="^",
        markersize=4,
        linewidth=1.2,
        label="Main board market"
    )

    ax.plot(
        x,
        y_chinext,
        color="#FFC000",
        marker="s",
        markersize=4,
        linewidth=1.2,
        label="ChiNext market"
    )

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Annualized Return", fontsize=9)

    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)

    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.grid(axis="x", visible=False)

    for spine in ax.spines.values():
        spine.set_color("#A0A0A0")
        spine.set_linewidth(0.8)

    ax.legend(
        loc="upper right",
        fontsize=7,
        frameon=False,
        handlelength=2.0
    )

    # 自动 y 轴范围，避免曲线被截断，同时减少刻度密度
    y_all = np.concatenate([
        np.asarray(y_main, dtype=float),
        np.asarray(y_chinext, dtype=float)
    ])

    y_max = np.nanmax(y_all)
    step = 0.4 if y_max > 2.0 else 0.2
    upper = max(step, np.ceil((y_max * 1.10) / step) * step)

    ax.set_ylim(0, upper)
    ax.set_yticks(np.arange(0, upper + 0.001, step))

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    set_seed(1795)

    args = parse_args()
    file_path = get_f3f4_path(args.test_batch)

    if args.run_analysis:
        f4a_df = run_f4a_lambdamart_lr_sensitivity(args)
        f4b_df = run_f4b_weak_learners_sensitivity(args)
        f4c_df = run_f4c_max_depth_sensitivity(args)

        if args.write_excel:
            # 保留 --write-excel 开关，但无论是否传入该参数，都不执行 Excel 写入。
            # 如需恢复写入，只需取消下一行注释并删除 pass。
            # write_f4_results_to_excel(f4a_df, f4b_df, f4c_df, file_path)
            pass

    if args.plot_figure:
        result_dir = get_result_dir(args.test_batch)

        if args.save_fig:
            result_dir.mkdir(parents=True, exist_ok=True)
            f4a_save_path = result_dir / "F4a.png"
            f4b_save_path = result_dir / "F4b.png"
            f4c_save_path = result_dir / "F4c.png"
        else:
            f4a_save_path = None
            f4b_save_path = None
            f4c_save_path = None

        plot_f4(
            file_path=file_path,
            sheet_name="F4a",
            xlabel="Learning rate",
            save_path=f4a_save_path
        )

        plot_f4(
            file_path=file_path,
            sheet_name="F4b",
            xlabel="Number of weak learners",
            save_path=f4b_save_path
        )

        plot_f4(
            file_path=file_path,
            sheet_name="F4c",
            xlabel="Maximum depth of the tree",
            save_path=f4c_save_path
        )
