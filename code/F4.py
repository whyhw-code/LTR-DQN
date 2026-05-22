# F4.py
import pandas as pd
import matplotlib.pyplot as plt

# 当前代码位于 LTR-DQN-main/code/
file_path = "../result/batch123/F3F4.xlsx"


def read_f4_sheet(sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)

    # 只保留 M 和 C，不读取 M1、C1
    df = df.loc[["M", "C"]]

    # 横轴作为文本处理
    x = [str(i) for i in df.columns]

    y_main = df.loc["M"].values
    y_chinext = df.loc["C"].values

    return x, y_main, y_chinext


def plot_f4(sheet_name, title, xlabel):
    x, y_main, y_chinext = read_f4_sheet(sheet_name)

    plt.figure(figsize=(7, 5))

    plt.plot(x, y_main, marker="o", label="Main Board market")
    plt.plot(x, y_chinext, marker="s", label="ChiNext market")

    plt.xlabel(xlabel)
    plt.ylabel("ARR")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_f4(
        sheet_name="F4a",
        title="Fig. 4(a) Learning rate of LambdaMART",
        xlabel="Learning rate"
    )

    plot_f4(
        sheet_name="F4b",
        title="Fig. 4(b) Number of weak learners",
        xlabel="Number of weak learners"
    )

    plot_f4(
        sheet_name="F4c",
        title="Fig. 4(c) Maximum depth of the tree",
        xlabel="Maximum depth"
    )