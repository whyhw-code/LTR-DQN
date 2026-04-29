# F3.py
import pandas as pd
import matplotlib.pyplot as plt

# 当前代码位于 LTR-DQN-main/code/
file_path = "../result/batch123/F3F4.xlsx"


def read_f3_sheet(sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)

    # 只保留 M 和 C
    df = df.loc[["M", "C"]]

    # 横轴作为文本处理
    x = [str(i) for i in df.columns]

    y_main = df.loc["M"].values
    y_chinext = df.loc["C"].values

    return x, y_main, y_chinext


def plot_f3(sheet_name, title):
    x, y_main, y_chinext = read_f3_sheet(sheet_name)

    plt.figure(figsize=(7, 5))

    plt.plot(x, y_main, marker="o", label="Main Board market")
    plt.plot(x, y_chinext, marker="s", label="ChiNext market")

    plt.xlabel("Learning rate")
    plt.ylabel("ARR")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_f3("F3a", "Fig. 3(a) Learning rate of LambdaRank")
    plot_f3("F3b", "Fig. 3(b) Learning rate of DQN in LTR-DQN")