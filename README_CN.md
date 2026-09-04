# LTR-DQN 复现说明（CPU 版）

[English](README.md) | 简体中文

本仓库用于从原始数据复现论文中的 T3、T4、T5、T6、T7 表格、正文图和附录图 C1-C5。每次运行都会重新训练排序模型和 DQN，不依赖已生成的结果、中间模型、每日选择文件或历史 `meiri_xuanze` 文件。

## 文件结构

### 入口脚本

- `train.py`：长时间训练入口。训练 LambdaRank、LambdaMART，并使用本次新生成的 LambdaMART 排序训练 LTR-DQN。加入 `--t6` 时同时执行 20 次采样实验。
- `main.py`：读取本次训练产物，训练快速基线模型，重新生成 DQN 动作，并输出 T3、T4、T5、T7 Results。
- `T6_main.py`：检查 T6 原始采样 CSV 并输出 T6 工作簿，同时包含 T6 的采样和回测实现。
- `Fig_main.py`：根据当前运行结果重新生成正文图 3-7 及其审计 CSV。
- `Appendix_Fig_main.py`：重新生成附录图 C1-C5 及其审计 CSV，C4 使用本次新生成的 T6 原始结果。

### 共享实现

- `experiment_core.py`：数据读取、LambdaRank/LambdaMART、基线模型、DQN 环境与智能体、回测、指标、表格格式化、运行清单和哈希。
- `runtime_config.py`：Python/依赖版本锁定值、各阶段种子和确定性的单 CPU 设置。默认 DQN 设备为 CPU。

### 数据和在线验证

- `data/0060merge_open_close_final.csv`、`data/3068merge_open_close_final.csv`：股票特征以及开收盘价。
- `data/0060merge_T4.csv`、`data/3068merge_T4.csv`：论文表格使用的指数/参考序列。
- `data/0060merge.csv`、`data/3068merge.csv`：基线、绘图和 T6 回测使用的市场数据。
- `data/dapan/`：基线和 DQN 回测使用的大盘数据。
- `data/ESG/`：T7 和附录 C5 使用的 ESG 排序输入。
- `data/reproducibility/`：T6 使用的两张 20-seed 配置表，只记录运行所需种子，不保存拟合结果或固定选择结果。
- `.github/workflows/reproduce-core.yml`：从干净原始数据生成 Results、全部正文图和附录 C1/C2/C3/C5。
- `.github/workflows/reproduce-t6.yml`：单独从干净原始数据生成 T6 和附录 C4。

### 环境文件

- `requirements-lock.txt`：CPU 复现使用的精确 pip 依赖锁定文件，包括 `torch==2.0.0+cpu`。
- `environment.yml`：Python 3.9.13 的 Conda 环境定义。
- `.gitignore`：排除运行生成的 `results/`、`temp/`、`model/`、`runs/` 和 Python 缓存。

运行后才会生成以下目录，均不提交到仓库：

```text
temp/       新生成的排序、DQN 动作和运行清单
model/      新生成的 DQN 检查点
runs/       可选的独立运行产物
results/    工作簿、CSV、图片和审计文件
```

## 系统和设备要求

- Python 3.9.13，64 位 x86（x64）。
- Windows 10/11 或 Linux x86_64；GitHub Actions 使用 Ubuntu 22.04 x64。
- 全流程使用 CPU，不要求 GPU，也不会选择 GPU 设备。
- BLAS、XGBoost 和 PyTorch 固定为单线程，以降低不同机器之间的差异。
- 完整运行建议至少 8 GB 内存和 10 GB 可用磁盘空间。

主要固定版本：NumPy 1.21.5、pandas 1.4.4、scikit-learn 1.2.0、PyTorch 2.0.0+cpu、XGBoost 1.7.6。入口脚本会在训练前检查环境版本，发现不一致会停止。

## 环境配置

### Windows PowerShell

```powershell
git clone https://github.com/whyhw-code/LTR-DQN.git
Set-Location LTR-DQN
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
$env:PYTHONHASHSEED = "0"
```

### Linux Bash

```bash
git clone https://github.com/whyhw-code/LTR-DQN.git
cd LTR-DQN
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
export PYTHONHASHSEED=0
```

也可以使用 Conda：

```bash
conda env create -f environment.yml
conda activate ltr-dqn
```

安装后先检查版本和设备：

```bash
python -c "import sys,torch,xgboost,numpy,pandas,sklearn; print(sys.version); print(torch.__version__, xgboost.__version__, numpy.__version__, pandas.__version__, sklearn.__version__, torch.get_num_threads()); print(torch.cuda.is_available())"
python -m compileall -q *.py
```

最后一项应显示 `False`，线程数应为 `1`。

## 本地完整复现命令

以下命令都在仓库根目录执行。

### 1. 训练主要模型

```bash
python train.py --models all --years 2,3,4 --ranker_tree_method approx
```

该命令从 `data/` 重新拟合 LambdaRank 和 LambdaMART，再使用本次新生成的 LambdaMART 排序训练 DQN。不会读取预计算排序或固定每日选择文件。

### 2. 输出 Results

```bash
python main.py --export_csvs
```

主工作簿为 `results/combined/results.xlsx`，包含 T3、T4、T5 和 T7；同目录还会生成论文格式 CSV 和运行清单。

### 3. 生成正文图

```bash
python Fig_main.py --ranker_tree_method approx --force
```

输出目录：`results/figures/`。

### 4. 生成附录图 C1、C2、C3、C5

```bash
python Appendix_Fig_main.py --figures C1,C2,C3,C5 --force
```

输出目录：`results/appendix_figures/`。

### 5. 单独生成 T6 和附录 C4

```bash
python train.py --models all --years 3 --t6 --ranker_tree_method approx
python T6_main.py
python Appendix_Fig_main.py --figures C4 --force
```

输出为 `results/T6/T6.xlsx` 和附录图 C4。T6 的 100% 列来自同一次运行中新评估的 T4 模型，不从外部中间结果复制。

## GitHub Actions 在线复现

1. 打开仓库的 **Actions** 页面。
2. 选择 **1 - Results and Figures (CPU)**，点击 **Run workflow**，生成 T3/T4/T5/T7、全部正文图和附录 C1/C2/C3/C5。
3. 选择 **2 - T6 and Figure C4 (CPU)**，点击 **Run workflow**，生成 T6 和附录 C4。
4. 运行结束后下载 artifact，其中包含工作簿、CSV、图片和运行清单。

两个 workflow 都固定使用 Ubuntu 22.04 x64 CPU、Python 3.9.13、固定 hash seed 和单线程设置。启动时不需要选择 runner，也不需要 GPU runner。

## 复现注意事项

- 默认种子映射在 `runtime_config.py`，T6 种子表在 `data/reproducibility/`。
- LambdaRank、LambdaMART 每次都从原始数据重新训练；DQN 使用同一次运行产生的 LambdaMART 输出。
- `PYTHONHASHSEED=0`、固定种子、PyTorch 确定性设置、稳定 CSV 排序和 `n_jobs=1` 用于减少平台差异。
- 运行清单记录依赖版本、输入哈希、动作哈希和检查点哈希。不同操作系统之间不保证字节级完全相同，差异会在清单中显示。
- 复现默认结果时不要修改 `--seed`、`--seed_config`、`--lr`、`--n_games` 或 `--ranker_tree_method`。
