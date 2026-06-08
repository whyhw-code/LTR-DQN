import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# 改成你的 Excel 实际路径
xlsx_path = Path(r"../result/batch123/基准+模型结果对比.xlsx")

# 保存到桌面
desktop_dir = Path.home() / "Desktop"
desktop_dir.mkdir(parents=True, exist_ok=True)

main_out_path = desktop_dir / "Fig5_Main_board_market.png"
chinext_out_path = desktop_dir / "Fig5_ChiNext_market.png"

series_cols = [
    "CSI 300 Index", "ChiNext Index", "Baseline portfolio",
    "LR", "MLP_R", "SVM_R", "XGBoost_R",
    "SVM_C", "MLP_C", "XGBoost_C",
    "LambdaRank", "LambdaMART", "LTR-DQN"
]


def _parse_report_date(df: pd.DataFrame) -> pd.Series:
    candidates = [c for c in ["qid", "qid_date"] if c in df.columns]

    for col in candidates:
        s = df[col]

        if pd.api.types.is_datetime64_any_dtype(s):
            d = pd.to_datetime(s, errors="coerce")
        else:
            raw = s.astype("string").str.replace(r"\.0$", "", regex=True)

            if raw.str.match(r"^\d{8}$").mean() > 0.8:
                d = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
            else:
                d = pd.to_datetime(
                    s,
                    unit="D",
                    origin="1899-12-30",
                    errors="coerce"
                )

        valid = d.dropna()
        if not valid.empty and valid.dt.year.between(2017, 2025).mean() > 0.9:
            return d

    raise ValueError(f"找不到可用日期列。当前列名为：{df.columns.tolist()}")


def read_market(sheet_name: str, index_col: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    df["date"] = _parse_report_date(df)
    df = df[df["date"].notna()].copy()
    df = df.sort_values("date")

    cols = [index_col, "Baseline portfolio"] + [
        c for c in series_cols
        if c in df.columns and c not in [index_col, "Baseline portfolio"]
    ]

    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").ffill()

    return df[["date"] + cols]


def plot_one_market(df: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))

    cols = [c for c in df.columns if c != "date"]

    for col in cols:
        lw = 2.4 if col == "LTR-DQN" else 1.2
        ax.plot(
            df["date"],
            df[col] / 1_000_000,
            label=col,
            linewidth=lw
        )

    ax.set_title(title)
    ax.set_ylabel("Total Fund (million)")
    ax.set_xlabel("Trading Day")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8, frameon=False)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main = read_market("主板", "CSI 300 Index")
    chinext = read_market("创业板", "ChiNext Index")

    plot_one_market(
        df=main,
        title="Main board market",
        out_path=main_out_path
    )

    plot_one_market(
        df=chinext,
        title="ChiNext Market",
        out_path=chinext_out_path
    )

    print(f"主板图已保存到：{main_out_path}")
    print(f"创业板图已保存到：{chinext_out_path}")