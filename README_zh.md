# LTR-DQN 论文复现包

本仓库用于从固定的实验输入数据重新生成论文 *Enhancing Predictive
Selection of Sell-Side Analyst Reports via Learning to Rank and Reinforcement
Learning* 的实证结果。排序结果、DQN 模型、表格和图片均在运行时重新
生成，不在 Git 中保存生成产物。

Figure 1 和 Figure 2 属于示意图，不在数值复现流程中。

## 在 GitHub 上一键复现

使用者不需要配置本地 Python 环境。

1. 打开仓库的 **Actions** 页面。
2. 选择 **Reproduce Core** 或 **Reproduce T6**。
3. 点击 **Run workflow**。
4. 运行完成后，在该次工作流页面下载 Artifact。

### Reproduce Core

该流程会重新训练模型并生成：

- T3、T4、T5、T7；
- 正文 Figure 3-7；
- 附录 Figure C1、C2、C3、C5；
- 记录运行环境和生成文件哈希值的 manifest。

下载文件名为 `ltr-dqn-core-<run-id>`。

### Reproduce T6

T6 的运行时间明显更长。GitHub 工作流会先训练所需的三年期模型，再将
固定 seed ledger 确定性拆分为 10 个并行任务，最后合并并检查每个
市场、抽样率和模型组合是否恰好包含 500 个结果。该流程生成：

- T6 及完整的抽样结果审计文件；
- 附录 Figure C4；
- 记录所有分片和合并文件哈希值的 manifest。

下载文件名为 `ltr-dqn-t6-<run-id>`。

## 上传内容

仓库只保留：

- 代码；
- 代码直接使用的固定输入数据；
- `data/reproducibility/` 下的两份 T6 seed ledger；
- Python 环境锁定文件；
- GitHub Actions 工作流和说明文档。

仓库中的 `data/` 是模型直接使用的整理后输入，不是原始研报 PDF 集合。
训练过程不会修改 `data/`。

以下内容不会上传：

- DQN checkpoint；
- 排序中间结果和每日 action；
- 已生成的 Excel、CSV 和图片；
- 日志、本地虚拟环境和 IDE 配置。

## 本地运行（可选）

GitHub Actions 是正式复现路径。本地运行需要 Windows 64 位 Python
3.9.13，并安装锁定的 CPU 环境：

```powershell
python -m pip install torch==2.0.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-lock.txt
```

Core：

```powershell
python train.py
python main.py --export_csvs
```

本地完整 T6：

```powershell
python train.py --models all --years 3 --t6 --ranker_tree_method hist
python T6_main.py
```

T6 需要第一条训练命令生成的三年期排名和 DQN 模型。单机完整运行可能
需要数小时；中断后会从 `temp/t6_runs/t6_raw.csv` 继续。

## 生成目录

所有生成文件都位于 `.gitignore` 排除的目录：

```text
model/                       # 新训练的 DQN 模型
temp/rankings/               # 新生成的排序结果
temp/actions/                # 新生成的 DQN action
temp/t6_runs/                # T6 每个 seed 的结果
results/combined/            # T3/T4/T5/T7
results/figures/             # 正文实证图片
results/appendix_figures/    # 附录实证图片
results/T6/                  # T6 表格、CSV 和 manifest
```

GitHub Actions 只会将这些内容作为临时 Artifact 提供下载，不会自动提交
回仓库。

## 锁定环境

- Python 3.9.13
- NumPy 1.21.5
- pandas 1.4.4
- PyTorch 2.0.0 CPU
- XGBoost 1.7.6

实验入口会在运行前检查环境版本。修改 seed、模型参数、依赖版本、操作
系统或 XGBoost tree method 都属于不同实验，可能改变拟合后的排序结果。

T6 云端分片只按照 seed 在固定 ledger 中的位置进行划分；合并后的 seed
集合和单机完整运行完全相同。生成的 manifest 会记录运行环境、输入来源
及 SHA-256 哈希值。
