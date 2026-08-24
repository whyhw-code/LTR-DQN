# LTR-DQN 论文复现

本仓库从固定输入数据重新生成论文的实证结果。模型、中间文件、表格和图片均由 GitHub Actions 运行时生成，不提交到 Git。

## 一键复现

打开仓库的 **Actions** 页面并选择：

- **Reproduce Core**：生成 T3、T4、T5、T7，以及除 Figure C4 外的实证图。
- **Reproduce T6**：分片运行耗时较长的 T6，并生成 Figure C4。

点击 **Run workflow**。完成后从该次运行页面下载 Artifact，无需配置本地环境。

## 仓库内容

- `data/`：复现所需的固定输入数据，其中 `data/reproducibility/` 保存 T6 seed ledger。
- `train.py`、`model.py`：冻结的训练代码。
- `main.py`：Core 结果入口。
- `T6_main.py`：T6 汇总入口。
- `.github/workflows/`：两个一键复现流程。
- `requirements.txt`：锁定的 Python 3.9.13 CPU 环境。

生成的 `model/`、`temp/`、`results/`、`runs/` 和日志均由 `.gitignore` 排除。
