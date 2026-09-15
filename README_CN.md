# LTR-DQN 复现说明（CPU 版）

[English](README.md) | 简体中文

**[新 GitHub 账号一键复现指南](GITHUB_ACTIONS_GUIDE_CN.md)**：从 Fork 本仓库、启用 Actions、运行 Results/T6，到下载表格和图片的完整操作步骤。

本仓库用于从原始数据复现论文中的 T3、T4、T5、T6、T7 表格、正文图和附录图 C1-C5。每次运行都会重新训练排序模型和 DQN，不依赖已生成的结果、中间模型、每日选择文件或历史 `meiri_xuanze` 文件。

**正式参考环境：**论文复现以 Windows 10/11 x64 CPU 和 `parameters_windows.txt` 为准，本地复现和 GitHub Actions 均使用这套 Windows 流程。

## 文件结构

### 入口脚本

- `train.py`：长时间训练入口。训练 LambdaRank、LambdaMART，并使用本次新生成的 LambdaMART 排序训练 LTR-DQN。加入 `--t6` 时同时执行 20 次采样实验。
- `main.py`：读取本次训练产物，训练快速基线模型，重新生成 DQN 动作，并输出 T3、T4、T5、T7 Results。
- `T6_main.py`：检查 T6 原始采样 CSV 并输出 T6 工作簿，同时包含 T6 的采样和回测实现。
- `Fig_main.py`：根据当前运行结果重新生成正文图 3-7 及其审计 CSV。
- `Appendix_Fig_main.py`：重新生成附录图 C1-C5 及其审计 CSV，C4 使用本次新生成的 T6 原始结果。

### 共享实现

- `experiment_core.py`：数据读取、LambdaRank/LambdaMART、基线模型、DQN 环境与智能体、回测、指标、表格格式化、运行清单和哈希。
- `runtime_config.py`：选择并校验 Windows 正式配置，同时设置确定性的单 CPU 环境。
- `parameters_windows.txt`：Windows 正式复现使用的种子和论文未报告的实现参数；本地 Windows 与 GitHub Actions 共用。

### 数据和在线验证

- `data/0060merge_open_close_final.csv`、`data/3068merge_open_close_final.csv`：股票特征以及开收盘价。
- `data/0060merge_T4.csv`、`data/3068merge_T4.csv`：论文表格使用的指数/参考序列。
- `data/0060merge.csv`、`data/3068merge.csv`：基线、绘图和 T6 回测使用的市场数据。
- `data/dapan/`：基线和 DQN 回测使用的大盘数据。
- `data/ESG/`：T7 和附录 C5 使用的原始 ESG 排序输入。
- `data/reproducibility/`：T6 使用的两张 20-seed 配置表，只记录运行所需种子，不保存拟合结果或固定选择结果。
- `.github/workflows/reproduce-core.yml`、`reproduce-t6.yml`：Windows CPU 参考复现工作流。

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

## 文件逐项说明

仓库按“训练、评估、采样、绘图、配置”拆分。每个脚本有独立的输入和输出，便于单独检查，也避免把上一次运行的中间结果误当作本次复现输入。

| 文件或目录 | 作用 | 主要读取 | 主要写入 |
| --- | --- | --- | --- |
| `train.py` | 长时间入口：训练 LambdaRank、LambdaMART，并用本次 LambdaMART 排名训练 DQN；加 `--t6` 时生成 T6 原始采样结果。 | `data/`、当前平台参数 | `temp/`、`model/`、`temp/train_manifest.json`，以及可选的 `temp/t6_runs/` |
| `main.py` | 短时间评估和汇总：训练 7 个基线模型、评估 3 个主模型、重新生成 DQN 动作并输出 T3/T4/T5/T7。 | `temp/` 或指定运行目录、`model/`、`data/` | `results/combined/` 及评估排名文件 |
| `experiment_core.py` | 唯一的共享实现：市场映射、日期、特征、模型、DQN 环境/智能体、回测、指标、Excel/CSV 格式、manifest 和哈希。 | 原始数据、运行配置 | 由各入口调用的共享产物 |
| `runtime_config.py` | 加载并校验 Windows 正式配置，并设置随机种子、线程数和 CPU 设备。 | `parameters_windows.txt` | 不直接写结果；元数据进入 manifest |
| `T6_main.py` | 校验新生成的 T6 逐种子 CSV，输出 T6 汇总表和工作簿；文件后半部分包含 `train.py --t6` 调用的采样与回测逻辑。 | `temp/t6_runs/t6_raw.csv`、T6 种子表、`data/` | `results/T6/` |
| `Fig_main.py` | 重新生成正文图 3-7；同时输出合并图、分市场/分面板图和可审计 CSV。 | 当前运行排名/模型、原始数据、活动参数 | `results/figures/` |
| `Appendix_Fig_main.py` | 重新生成附录图 C1-C5；C2 优先使用机构级 XLSX，C4 使用本次 T6 原始 CSV，C5 使用原始 ESG 数据。 | 原始数据和当前运行产物 | `results/appendix_figures/` |
| `parameters_windows.txt` | Windows 正式复现配置：各阶段种子及论文未报告的 LambdaRank/LambdaMART 实现参数。供本地 Windows 和 Windows Actions 使用。 | 无 | 无 |
| `requirements-lock.txt` | Windows CPU 正式复现的精确 pip 依赖锁定文件。 | 无 | 无 |
| `environment.yml` | Windows 正式复现的 Conda 环境定义。 | 无 | 无 |
| `.github/workflows/reproduce-core.yml` | 手动 Windows CPU Action：T3/T4/T5/T7、正文图、C1/C2/C3/C5。 | 干净 checkout | 上传表格、图片和 manifest 的 artifact |
| `.github/workflows/reproduce-t6.yml` | 手动 Windows CPU Action：T6 和 C4。 | 干净 checkout | 上传 T6/C4 产物和 manifest 的 artifact |
| `GITHUB_ACTIONS_GUIDE_CN.md` | GitHub Actions 的中文操作指南，说明如何启用工作流和下载 artifact。 | 无 | 无 |
| `data/` | 版本化原始输入，不是模型缓存或固定结果。 | 无 | 无 |
| `README.md` / `README_CN.md` | 英文/中文仓库说明和复现指南。 | 无 | 无 |

### 数据文件含义

`0060` 表示主板，`3068` 表示创业板。两个 `*_merge_open_close_final.csv` 包含训练和组合评估所需的股票特征、标签以及开收盘价；较小的 `*_merge.csv` 用于基线、绘图和 DQN 回测；`*_merge_T4.csv` 是生成论文格式 T4 表格时使用的指数/参考序列。`data/dapan/` 是 DQN 状态和大盘回测数据。`data/ESG/ESG.csv` 是股票级 ESG 横截面，两个带日期的 ESG 文件包含 T7 排名面板所需的预测值和价格。两份机构 XLSX 只用于附录 C2 的机构级分析。`data/reproducibility/` 只保存 T6 种子台账，不保存拟合模型或固定选股结果。

### 股票表字段

`experiment_core.py` 中的 `FEATURES` 共包含 25 个输入变量：报告文本特征（`page`、`advance_reaction`、`star_analyst`、`title_len`、`num_sentence`、`avg_sentence_len`、`sd_sentence_len`、`num_authors`、`analyst_coverage`）、Fama-French 风格因子（`rm_rf`、`smb`、`hml`、`rmw`、`cma`）、机构和上市属性（`broker_size`、`listed`、`broker_status`）、历史收益统计（`prior_performance_avg`、`prior_performance_sd`）以及 6 个行业指示变量（`ind_1`-`ind_6`）。`real_return` 是排序/回归目标，`up_down` 是二元分类目标；`qid_date` 将股票分组为一个排序日，`stock_code`、`close`、`pclose` 分别标识股票和卖出/买入价格。

### 核心函数对应关系

| 函数 | 作用 |
| --- | --- |
| `load_stock_data` | 从原始股票文件选择指定年限的训练区间和统一测试区间。 |
| `fit_ranker` | 训练一个 LambdaRank 或 LambdaMART，并返回训练/测试预测表。 |
| `fit_baseline` / `model_for_baseline` | 训练一个短时间基线模型并返回测试预测。 |
| `train_dqn` | 构建 DQN 环境、训练智能体并保存带元数据的检查点。 |
| `evaluate_dqn` | 加载训练好的检查点，为测试期生成策略动作。 |
| `backtest_predictions` | 将每日预测转换为前四只交易，并计算 ARR/MDR/CR/SR/WR。 |
| `write_results` | 写出长表、论文格式 CSV 和多工作表工作簿。 |

`runtime_config.py` 会在模型运行前自动加载并应用仓库中的 Windows 复现配置。

## 系统和设备要求

- Python 3.9.13，64 位 x86（x64）。
- Windows 10/11 x64 CPU 是论文结果的正式参考环境。
- 全流程使用 CPU，不要求 GPU，也不会选择 GPU 设备。
- BLAS、XGBoost 和 PyTorch 固定为单线程，以降低不同机器之间的差异。
- 完整运行建议至少 8 GB 内存和 10 GB 可用磁盘空间。

Windows 正式锁定版本为 pip 24.1.2、NumPy 1.21.5、pandas 1.4.4、scikit-learn 1.2.0、PyTorch 2.0.0+cpu、XGBoost 1.7.6。

## 环境配置（Windows 正式参考）

### 正式参考环境：Windows PowerShell

```powershell
git clone https://github.com/whyhw-code/LTR-DQN.git
Set-Location LTR-DQN
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade "pip==24.1.2"
python -m pip install -r requirements-lock.txt
$env:PYTHONHASHSEED = "0"
$env:PYTHONWARNINGS = "ignore"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
```

Windows 也可以使用 Conda：

```powershell
conda env create -f environment.yml
conda activate ltr-dqn
```

安装后先检查版本和设备：

```powershell
python -c "import sys,torch,xgboost,numpy,pandas,sklearn; print(sys.version); print(torch.__version__, xgboost.__version__, numpy.__version__, pandas.__version__, sklearn.__version__, torch.get_num_threads()); print(torch.cuda.is_available())"
python -m compileall -q *.py
```

最后一项应显示 `False`，线程数应为 `1`。

Windows 入口脚本和工作流会统一屏蔽第三方 Python 警告、XGBoost 警告日志以及 pip 版本提示，运行输出只保留进度、结果和真正的错误。

## 本地完整复现命令（Windows PowerShell）

以下命令都在仓库根目录执行。

### 1. 训练主要模型

```powershell
python train.py --models all --years 2,3,4 --ranker_tree_method approx
```

该命令从 `data/` 重新拟合 LambdaRank 和 LambdaMART，再使用本次新生成的 LambdaMART 排序训练 DQN。不会读取预计算排序或固定每日选择文件。

### 2. 输出 Results

```powershell
python main.py --export_csvs
```

主工作簿为 `results/combined/results.xlsx`，包含 T3、T4、T5 和 T7；同目录还会生成论文格式 CSV 和运行清单。

### 3. 生成正文图

```powershell
python Fig_main.py --ranker_tree_method approx --force
```

输出目录：`results/figures/`。

### 4. 生成附录图 C1、C2、C3、C5

```powershell
python Appendix_Fig_main.py --figures C1,C2,C3,C5 --force
```

输出目录：`results/appendix_figures/`。

### 5. 单独生成 T6 和附录 C4

```powershell
python train.py --models all --years 3 --t6 --ranker_tree_method approx
python T6_main.py
python Appendix_Fig_main.py --figures C4 --force
```

输出为 `results/T6/T6.xlsx` 和附录图 C4。T6 的 100% 列来自同一次运行中新评估的 T4 模型，不从外部中间结果复制。

## GitHub Actions 在线复现

在线复现不要求本地安装 Python，也不需要 GPU。两个工作流都会在标准 Windows Server 2022 x64 CPU 环境中，从干净的仓库和原始数据开始运行。

### 1. Fork 仓库

1. 登录 GitHub，打开 <https://github.com/whyhw-code/LTR-DQN>。
2. 点击页面右上角的 **Fork**。
3. 在 **Owner** 中选择自己的账号，仓库建议保持公开，名称可以继续使用 `LTR-DQN`。
4. 点击 **Create fork**，等待页面跳转到自己账号下的仓库。
5. 确认当前分支为 `main`。Fork 只复制代码和原始数据，不会带入以前运行生成的结果。

### 2. 启用工作流

1. 在自己 Fork 后的仓库中进入顶部的 **Actions** 页面，不要停留在原仓库。
2. 如果出现 **I understand my workflows, go ahead and enable them**，点击一次启用。
3. 确认左侧出现两个工作流：

```text
1 - One-click Results and Figures (Windows CPU)
2 - One-click T6 and Figure C4 (Windows CPU)
```

### 3. 运行主结果和正文图

1. 点击 **1 - One-click Results and Figures (Windows CPU)**。
2. 点击右侧的 **Run workflow**，分支保持为 `main`；该页面没有需要填写的实验参数。
3. 再点击绿色的 **Run workflow** 确认启动。
4. 页面出现新的运行记录后，点击该记录即可查看各步骤状态。

该工作流会检查仓库中没有历史输出目录，安装锁定的 Windows 环境，确认读取 `parameters_windows.txt`，运行测试，然后从原始数据训练两年、三年和四年模型并生成：

```text
results/combined/results.xlsx       T3、T4、T5、T7 总工作簿
results/combined/                   论文格式 CSV 和运行清单
results/figures/                    全部正文图及审计 CSV
results/appendix_figures/           附录 C1、C2、C3、C5 及审计 CSV
```

### 4. 单独运行 T6 和附录 C4

1. 返回 **Actions** 页面，选择 **2 - One-click T6 and Figure C4 (Windows CPU)**。
2. 点击 **Run workflow**，保持 `main`，再点击绿色按钮确认。

这个任务独立训练三年期模型，并重新执行 T6 的 20 次采样，生成：

```text
results/T6/T6.xlsx                  T6 工作簿
results/appendix_figures/           附录图 C4 及审计 CSV
temp/t6_runs/t6_raw.csv             T6 逐次运行审计数据
temp/t6_runs/t6_manifest.json       T6 运行清单
```

两个工作流彼此独立，可以分别运行。关闭浏览器不会终止后台任务；黄色圆点表示排队或运行中，绿色对勾表示成功，红色叉号表示某个步骤失败。

### 5. 下载并核对结果

1. 打开带绿色对勾的运行记录。
2. 在运行概要页底部找到 **Artifacts**。
3. 根据工作流下载对应的压缩包：

```text
ltr-dqn-results-and-figures-<运行编号>
ltr-dqn-t6-and-c4-<运行编号>
```

4. 解压后按上面的目录查找工作簿、CSV 和图片。压缩包同时包含运行清单以及该次实际采用的 `parameters_windows.txt`。

Artifacts 保留 14 天。运行结果不会出现在仓库源文件列表中，必须从成功运行的详情页下载。

### 6. 处理失败或旧版本运行

- 进入失败记录，打开对应 job，展开第一个带红叉的步骤。依赖问题通常位于 **Install locked environment**，训练或绘图问题会显示在对应名称的步骤中。
- 如果原仓库已经更新，在自己 Fork 的首页点击 **Sync fork**，再点击 **Update branch**；同步完成后应新建一次运行。旧记录中的 **Re-run jobs** 仍会使用旧提交。
- 如果没有 **Run workflow** 按钮，确认 Actions 已启用、工作流文件位于默认 `main` 分支，并且当前账号对这个 Fork 有写入权限。

需要更细的页面操作说明和最终核对清单时，可继续查看 [GITHUB_ACTIONS_GUIDE_CN.md](GITHUB_ACTIONS_GUIDE_CN.md)。

## 复现注意事项

- Windows 正式复现配置位于 `parameters_windows.txt`；T6 种子表在 `data/reproducibility/`。
- LambdaRank、LambdaMART 每次都从原始数据重新训练；DQN 使用同一次运行产生的 LambdaMART 输出。
- `PYTHONHASHSEED=0`、固定种子、PyTorch 确定性设置、稳定 CSV 排序和 `n_jobs=1` 用于减少平台差异。
- 运行清单记录依赖版本、输入哈希、动作哈希和检查点哈希，便于核对不同 Windows 运行。
- 复现论文结果时保持仓库和工作流的默认设置，不要自行修改训练配置。
