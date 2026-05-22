import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

xlsx_path = Path(r"../result/batch123/基准+模型结果对比.xlsx")
out_path = Path("Fig5_reproduction.png")

series_cols = [
    "CSI 300 Index", "ChiNext Index", "Baseline portfolio",
    "LR", "MLP_R", "SVM_R", "XGBoost_R",
    "SVM_C", "MLP_C", "XGBoost_C",
    "LambdaRank", "LambdaMART", "LTR-DQN"
]

def read_market(sheet_name):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df = df.rename(columns={df.columns[1]: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()
    df = df.iloc[:-1] if df.iloc[-1].dropna().shape[0] < 8 else df
    return df

main = read_market("主板")
chinext = read_market("创业板")

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

plots = [
    (axes[0], main, "Main board market", "CSI 300 Index"),
    (axes[1], chinext, "ChiNext Market", "ChiNext Index"),
]

for ax, df, title, index_col in plots:
    cols = [index_col, "Baseline portfolio"] + [
        c for c in series_cols if c in df.columns and c not in [index_col, "Baseline portfolio"]
    ]

    for col in cols:
        lw = 2.4 if col == "LTR-DQN" else 1.2
        ax.plot(df["date"], df[col] / 1_000_000, label=col, linewidth=lw)

    ax.set_title(title)
    ax.set_ylabel("Return / million")
    ax.set_xlabel("Trading Day")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8, frameon=False)

fig.tight_layout()
# fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()