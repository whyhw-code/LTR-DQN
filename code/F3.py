# F3.py
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
# 默认只读取已有 F3F4.xlsx 并画图
# 重新计算：python F3.py --run-analysis
# 重新计算并写入：python F3.py --run-analysis --write-excel
# 不画图：python F3.py --no-plot
#
# ============================================================

# 当前代码位于 LTR-DQN-main/code/
CODE_DIR = Path(__file__).resolve().parent

LEARNING_RATES = [0.0001, 0.001, 0.002, 0.01, 0.1, 0.2]

# 写入 Excel 时的行顺序，匹配 F3F4.xlsx：
# A2 = C，表示 ChiNext market
# A3 = M，表示 Main board market
F3_ROW_ORDER = ["C", "M"]

F3A_SCRIPTS = {
    "M": "T4M10.py",
    "C": "T4C10.py",
}

F3B_SCRIPTS = {
    "M": "T4M12.py",
    "C": "T4C12.py",
}

F3A_BASE_PARAMS = {
    "M": {
        "shouxufei": 0.0003,
        "yinhaushui": 0.001,
    },
    "C": {
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

    # 默认：不调用 T4 脚本，直接读取已有 F3F4.xlsx 画图
    # 需要重新计算时，手动加 --run-analysis
    parser.add_argument(
        "--run-analysis",
        dest="run_analysis",
        action="store_true",
        help="调用 T4M10/T4C10/T4M12/T4C12 重新计算 F3a/F3b"
    )
    parser.set_defaults(run_analysis=False)

    # 默认：不写回 F3F4.xlsx
    # 需要写回时，手动加 --write-excel
    parser.add_argument(
        "--write-excel",
        dest="write_excel",
        action="store_true",
        help="把 F3a/F3b 结果写回 F3F4.xlsx"
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
        help="保存 F3a.png 和 F3b.png 到 result/batchxxx/"
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


def reorder_f3_df(df):
    col_order = [fmt_num(x) for x in LEARNING_RATES]

    for row in F3_ROW_ORDER:
        if row not in df.index:
            df.loc[row] = np.nan

    for col in col_order:
        if col not in df.columns:
            df[col] = np.nan

    return df.loc[F3_ROW_ORDER, col_order]


# ============================================================
# F3a：调用 T4M10.py / T4C10.py
# ============================================================

def run_f3a_lambdarank_sensitivity(args):
    result = {}

    for row_name in F3_ROW_ORDER:
        script_path = check_script_exists(F3A_SCRIPTS[row_name])
        base_params = F3A_BASE_PARAMS[row_name]
        arr_values = []

        for lr_value in LEARNING_RATES:
            script_args = [
                "--train_or_test", "test",
                "--shouxufei", fmt_num(base_params["shouxufei"]),
                "--yinhaushui", fmt_num(base_params["yinhaushui"]),
                "--learning_rate", fmt_num(lr_value),
            ]

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

    return reorder_f3_df(df)


# ============================================================
# F3b：调用 T4M12.py / T4C12.py
# ============================================================

def run_f3b_dqn_lr_sensitivity(args):
    """
    前提：
    T4M12.py / T4C12.py 已支持命令行参数：
        --lr
        --train_year
        --test_batch

    注意：
    F3b 必须使用 DQN 动态 action 版本；
    若你的 T4C12/T4M12 中 lr==0.002 用固定 action，其它 lr 用动态 action，
    则 0.002 用于 T4/Fig5 复现，其它点用于敏感性分析。
    """
    result = {}

    for row_name in F3_ROW_ORDER:
        script_path = check_script_exists(F3B_SCRIPTS[row_name])
        arr_values = []

        for lr_value in LEARNING_RATES:
            script_args = [
                "--lr", fmt_num(lr_value),
                "--train_year", str(args.train_year),
                "--test_batch", str(args.test_batch),
            ]

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

    return reorder_f3_df(df)


# ============================================================
# 写入 F3F4.xlsx
# ============================================================

def get_result_dir(test_batch):
    return (CODE_DIR / f"../result/batch{test_batch}").resolve()


def get_f3f4_path(test_batch):
    return get_result_dir(test_batch) / "F3F4.xlsx"


def write_f3_results_to_excel(f3a_df, f3b_df, file_path):
    file_path.parent.mkdir(parents=True, exist_ok=True)

    f3a_df = reorder_f3_df(f3a_df)
    f3b_df = reorder_f3_df(f3b_df)

    if file_path.exists():
        with pd.ExcelWriter(
            file_path,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace"
        ) as writer:
            f3a_df.to_excel(writer, sheet_name="F3a")
            f3b_df.to_excel(writer, sheet_name="F3b")
    else:
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="w") as writer:
            f3a_df.to_excel(writer, sheet_name="F3a")
            f3b_df.to_excel(writer, sheet_name="F3b")


# ============================================================
# 画图
# ============================================================

def read_f3_sheet(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)

    # 只保留 M 和 C
    df = df.loc[["M", "C"]]

    # 横轴作为文本处理
    x = [str(i) for i in df.columns]

    y_main = df.loc["M"].values
    y_chinext = df.loc["C"].values

    return x, y_main, y_chinext


def plot_f3(file_path, sheet_name, save_path=None):
    x, y_main, y_chinext = read_f3_sheet(file_path, sheet_name)

    fig, ax = plt.subplots(figsize=(4.6, 2.8), dpi=120)

    # 白色背景
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Main board market：蓝色三角
    ax.plot(
        x,
        y_main,
        color="#4472C4",
        marker="^",
        markersize=4,
        linewidth=1.2,
        label="Main board market"
    )

    # ChiNext market：黄色方块
    ax.plot(
        x,
        y_chinext,
        color="#FFC000",
        marker="s",
        markersize=4,
        linewidth=1.2,
        label="ChiNext market"
    )

    ax.set_xlabel("Learning rate", fontsize=9)
    ax.set_ylabel("Annualized Return", fontsize=9)

    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)

    # 白底灰线
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.grid(axis="x", visible=False)

    # 边框
    for spine in ax.spines.values():
        spine.set_color("#A0A0A0")
        spine.set_linewidth(0.8)

    # 图例
    ax.legend(
        loc="upper right",
        fontsize=7,
        frameon=False,
        handlelength=2.0
    )

    # y 轴自动范围，避免截断，也避免刻度过密
    # y 轴范围设置
    if sheet_name == "F3a":
        y_all = np.concatenate([
            np.asarray(y_main, dtype=float),
            np.asarray(y_chinext, dtype=float)
        ])

        y_max = np.nanmax(y_all)
        step = 0.2
        upper = max(1.2, np.ceil((y_max * 1.10) / step) * step)

        ax.set_ylim(0, upper)
        ax.set_yticks(np.arange(0, upper + 0.001, step))

    elif sheet_name == "F3b":
        # F3b 纵轴固定
        ax.set_ylim(-0.1, 3.2)
        ax.set_yticks(np.arange(0, 3.21, 0.4))

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
        f3a_df = run_f3a_lambdarank_sensitivity(args)
        f3b_df = run_f3b_dqn_lr_sensitivity(args)

        if args.write_excel:
            write_f3_results_to_excel(f3a_df, f3b_df, file_path)

    if args.plot_figure:
        result_dir = get_result_dir(args.test_batch)

        if args.save_fig:
            result_dir.mkdir(parents=True, exist_ok=True)
            f3a_save_path = result_dir / "F3a.png"
            f3b_save_path = result_dir / "F3b.png"
        else:
            f3a_save_path = None
            f3b_save_path = None

        plot_f3(file_path, "F3a", save_path=f3a_save_path)
        plot_f3(file_path, "F3b", save_path=f3b_save_path)
