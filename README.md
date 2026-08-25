# LTR-DQN 论文复现

本仓库只保留复现实验所需的固定输入、五个运行脚本和 GitHub Actions。模型、排名、动作、表格、图片及日志均在运行时生成，不提交到 Git。

## 一键复现

进入仓库的 **Actions** 页面：

- **Reproduce Core**：生成 T3、T4、T5、T7 及除 Figure C4 外的实证图。
- **Reproduce T6**：并行完成耗时较长的 T6，并生成 Figure C4。

点击 **Run workflow**，完成后下载 Artifact；不需要本地 Python 环境。

## 五个脚本

- `train.py`：训练入口。
- `model.py`：模型、评估及表格公共实现。
- `dl_dqn2.py`：DQN 实现。
- `main.py`：T3、T4、T5、T7 和非 T6 图形入口。
- `T6_main.py`：T6 采样、并行分片、汇总及 Figure C4 入口。

`data/` 只包含这些流程直接读取的固定输入，其中 `data/reproducibility/` 保存 T6 seed ledger。运行环境锁定在 `requirements.txt` 中，生成目录由 `.gitignore` 排除。
