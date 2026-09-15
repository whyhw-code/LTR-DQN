# LTR-DQN 复现说明（CPU 版）

[English](README.md) | 简体中文

**[新 GitHub 账号一键复现指南](GITHUB_ACTIONS_GUIDE_CN.md)**：从 Fork 本仓库、启用 Actions、运行 Results/T6，到下载表格和图片的完整操作步骤。

本仓库用于从原始数据复现论文中的 T3、T4、T5、T6、T7 表格、正文图和附录图 C1-C5。每次运行都会重新训练排序模型和 DQN，不依赖已生成的结果、中间模型、每日选择文件或历史 `meiri_xuanze` 文件。

**正式参考环境：**论文复现以 Windows 10/11 x64 CPU 和 `parameters_windows.txt` 为准。Linux 文件和 shell 脚本只是 Windows 环境不可用时的本地/租用服务器兼容路径；Linux 只要求整体结果接近，不是第二套论文参考实现。

本 README 的正式流程、参数和在线验证均以 Windows 为核心。Linux 只在安装命令、shell 写法或系统自动选择配置有差异时作兼容说明。

## 文件结构

### 入口脚本

- `train.py`：长时间训练入口。训练 LambdaRank、LambdaMART，并使用本次新生成的 LambdaMART 排序训练 LTR-DQN。加入 `--t6` 时同时执行 20 次采样实验。
- `main.py`：读取本次训练产物，训练快速基线模型，重新生成 DQN 动作，并输出 T3、T4、T5、T7 Results。
- `T6_main.py`：检查 T6 原始采样 CSV 并输出 T6 工作簿，同时包含 T6 的采样和回测实现。
- `Fig_main.py`：根据当前运行结果重新生成正文图 3-7 及其审计 CSV。
- `Appendix_Fig_main.py`：重新生成附录图 C1-C5 及其审计 CSV，C4 使用本次新生成的 T6 原始结果。

### 共享实现

- `experiment_core.py`：数据读取、LambdaRank/LambdaMART、基线模型、DQN 环境与智能体、回测、指标、表格格式化、运行清单和哈希。
- `runtime_config.py`：优先选择并校验 Windows 正式配置；在 Linux 上才选择兼容配置，同时设置确定性的单 CPU 环境。
- `parameters_windows.txt`：Windows 正式复现使用的种子和论文未报告的实现参数；本地 Windows 与 GitHub Actions 共用。
- `parameters_linux.txt`：仅供 Linux 兼容运行使用的参数；Windows 和 GitHub Actions 不使用该文件。

### 数据和在线验证

- `data/0060merge_open_close_final.csv`、`data/3068merge_open_close_final.csv`：股票特征以及开收盘价。
- `data/0060merge_T4.csv`、`data/3068merge_T4.csv`：论文表格使用的指数/参考序列。
- `data/0060merge.csv`、`data/3068merge.csv`：基线、绘图和 T6 回测使用的市场数据。
- `data/dapan/`：基线和 DQN 回测使用的大盘数据。
- `data/ESG/`：T7 和附录 C5 使用的原始 ESG 排序输入。
- T7 阈值在运行时由原始横截面文件 `data/ESG/ESG.csv` 计算。当前数据得到
  q25/q50 为 `5.52`/`6.02`；两个市场及 NS 与 PI 共用同一组 cutoff。

### T7 的两种 ESG 策略

论文规定的是剔除 ESG 得分最低的比例，分数 cutoff 由原始横截面 ESG 数据计算：

- **NS（Negative Screening，负面筛选）**：先按 DQN 的预测排名选出推荐股票，
  再剔除低于统一 q25 或 q50 cutoff 的持仓。因此组合股票数可以少于原始
  DQN 推荐数量。
- **PI（Positive Investing，积极投资）**：使用统一 q25 或 q50 cutoff，
  再按模型预测排名从合格推荐中选股，替换被排除股票，尽量补足 DQN 推荐数量；
  若合格股票不足，则使用全部合格股票。

两种策略在两个市场、同一档位共用一个 ESG 分数 cutoff；区别是 NS 不递补，
PI 对被剔除的股票进行递补。
- `data/reproducibility/`：T6 使用的两张 20-seed 配置表，只记录运行所需种子，不保存拟合结果或固定选择结果。
- `.github/workflows/reproduce-core.yml`、`reproduce-t6.yml`：Windows CPU 参考复现工作流。

### 环境文件

- `requirements-lock.txt`：CPU 复现使用的精确 pip 依赖锁定文件，包括 `torch==2.0.0+cpu`。
- `environment.yml`：Python 3.9.13 的 Conda 环境定义。
- `environment-linux.yml`、`requirements-linux.txt`、`run_linux.sh`、`run_t6_linux.sh`：仅供 Linux 命令和依赖有差异时使用的兼容路径，不属于 GitHub 在线验证。
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
| `runtime_config.py` | 优先加载并校验 Windows 正式配置；仅在 Linux 兼容运行时选择 Linux 配置，并设置随机种子/线程数和 CPU 设备。 | `parameters_windows.txt` 或 `parameters_linux.txt` | 不直接写结果；元数据进入 manifest |
| `T6_main.py` | 校验新生成的 T6 逐种子 CSV，输出 T6 汇总表和工作簿；文件后半部分包含 `train.py --t6` 调用的采样与回测逻辑。 | `temp/t6_runs/t6_raw.csv`、T6 种子表、`data/` | `results/T6/` |
| `Fig_main.py` | 重新生成正文图 3-7；同时输出合并图、分市场/分面板图和可审计 CSV。 | 当前运行排名/模型、原始数据、活动参数 | `results/figures/` |
| `Appendix_Fig_main.py` | 重新生成附录图 C1-C5；C2 优先使用机构级 XLSX，C4 使用本次 T6 原始 CSV，C5 使用原始 ESG 数据。 | 原始数据和当前运行产物 | `results/appendix_figures/` |
| `parameters_windows.txt` | Windows 正式复现配置：各阶段种子及论文未报告的 LambdaRank/LambdaMART 实现参数。供本地 Windows 和 Windows Actions 使用。 | 无 | 无 |
| `parameters_linux.txt` | 仅在脱离 Windows 正式环境运行同一代码时使用的 Linux CPU 兼容配置。 | 无 | 无 |
| `requirements-lock.txt` | Windows CPU 正式复现的精确 pip 依赖锁定文件。 | 无 | 无 |
| `environment.yml` | Windows 正式复现的 Conda 环境定义。 | 无 | 无 |
| `requirements-linux.txt` / `environment-linux.yml` | 仅供兼容说明使用的 Linux 依赖定义。 | 无 | 无 |
| `run_linux.sh` / `run_t6_linux.sh` | Linux 兼容路径的脚本；Windows 用户应使用下方 PowerShell 命令。 | 源代码和数据 | 与主流程相同的 `temp/`、`model/`、`results/` |
| `tests/test_linux_compat.py` | 快速结构和配置测试，不进行模型训练。 | 源文件、原始 ESG 数据 | 无 |
| `.github/workflows/reproduce-core.yml` | 手动 Windows CPU Action：T3/T4/T5/T7、正文图、C1/C2/C3/C5。 | 干净 checkout | 上传表格、图片和 manifest 的 artifact |
| `.github/workflows/reproduce-t6.yml` | 手动 Windows CPU Action：T6 和 C4。 | 干净 checkout | 上传 T6/C4 产物和 manifest 的 artifact |
| `GITHUB_ACTIONS_GUIDE_CN.md` | GitHub Actions 的中文操作指南，说明如何启用工作流和下载 artifact。 | 无 | 无 |
| `LINUX_REPRODUCIBILITY.md` | Linux 兼容路径、锁定/放宽环境和预期数值差异说明。 | 无 | 无 |
| `data/` | 版本化原始输入，不是模型缓存或固定结果。 | 无 | 无 |
| `README.md` / `README_CN.md` | 英文/中文使用、流程和参数说明。 | 无 | 无 |

### 数据文件含义

`0060` 表示主板，`3068` 表示创业板。两个 `*_merge_open_close_final.csv` 包含训练和组合评估所需的股票特征、标签以及开收盘价；较小的 `*_merge.csv` 用于基线、绘图和 DQN 回测；`*_merge_T4.csv` 是生成论文格式 T4 表格时使用的指数/参考序列。`data/dapan/` 是 DQN 状态和大盘回测数据。`data/ESG/ESG.csv` 是每只股票一条记录的 ESG 横截面，用于计算 q25/q50；两个带日期的 ESG 文件包含 T7 排名面板所需的预测值和价格。两份机构 XLSX 只用于附录 C2 的机构级分析。`data/reproducibility/` 只保存 T6 种子台账，不保存拟合模型或固定选股结果。

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
| `esg_thresholds_common` / `esg_metrics` | 从原始数据计算统一 ESG 阈值并评估 NS/PI 组合。 |
| `write_results` | 写出长表、论文格式 CSV 和多工作表工作簿。 |

`runtime_config.py` 中的对应配置函数为：`load_platform_parameters` 校验平台文件，`stage_seed` 解析阶段种子，`load_rank_config`/`load_mart_config` 合并仅限实现层面的覆盖，`set_global_determinism` 设置 Python/NumPy/PyTorch 随机种子。

## 参数说明

参数分为三层：论文明确报告的值由 Python 代码锁定；Windows 配置是正式参考配置，Linux 配置是面向不同操作系统和依赖构建的应急兼容配置；两者只保存论文没有完全说明的种子和实现选项；命令行参数是运行时控制，并会记录进 manifest。

### 论文锁定的模型参数

| 组件 | 主板（`0060`） | 创业板（`3068`） | 说明 |
| --- | --- | --- | --- |
| LambdaRank 目标 | `rank:pairwise` | `rank:pairwise` | 独立排序基线 |
| LambdaRank `learning_rate` | `0.01` | `0.1` | 锁定，平台文件不能覆盖 |
| LambdaRank 未报告值 | `max_depth=6`、`n_estimators=100`、`subsample=1.0`、`colsample_bytree=1.0` | 若活动平台配置提供未锁定字段，则按该配置 | `eval_metric=ndcg`、`n_jobs=1` |
| LambdaMART 目标 | `rank:map` | `rank:ndcg` | 两个市场目标函数不同 |
| LambdaMART `learning_rate` | `0.001` | `0.1` | 锁定 |
| LambdaMART `max_depth` | `5` | `6` | 锁定 |
| LambdaMART `n_estimators` | `1000` | `1000` | 锁定 |
| LTR-DQN `learning_rate` | `0.002` | `0.002` | 锁定，使用 Adam |

`--ranker_tree_method` 直接控制绘图敏感性拟合使用的 XGBoost CPU 建树方式，默认是 `approx`，也可选 `hist` 或 `exact`。`train.py` 主训练实际使用活动平台配置中的 `tree_method`；该选项值会记录在 manifest，正式 Windows 配置每一行都使用 `approx`。它不会改变论文锁定的学习率和树数量。DQN 的状态排名输入固定来自本次新生成的 LambdaMART（`DQN_RANKER = "LambdaMART"`）。

### DQN 默认参数

以下值是 `train.py` 的默认值，并写入 DQN 检查点元数据。其中 `--lr` 会校验为论文规定的 `0.002`。

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--n_games` | `31` | 训练回合数 |
| `--lr` | `0.002` | Adam 学习率，论文锁定 |
| `--gamma` | `0.9` | 未来奖励折扣因子 |
| `--epsilon` | `1.0` | 初始 epsilon-greedy 探索概率 |
| `--eps_end` | `0.03` | 最低探索概率 |
| `--eps_dec` | `0.00015` | 每次学习后的探索概率递减量 |
| `--batch_size` | `32` | 每次更新抽取的经验数量 |
| `--max_mem_size` | `100` | 经验回放容量 |
| `--replace_target_iter` | `8` | 每隔多少次学习复制一次目标网络 |
| 网络结构 | `13 -> 256 -> 128 -> 5` | 状态宽度、两个隐藏层和 5 个动作（选择 0-4 只股票） |
| 优化器/损失 | Adam / MSE | Adam betas=`(0.9, 0.999)`，`eps=1e-8` |

DQN 环境初始资金为 `500,000,000`，买入佣金为 `0.0003`，卖出佣金加印花税为 `0.0013`，按当前 LambdaMART 预测排名选择股票。排序模型/基线组合指标使用 `5,000,000` 初始资金；适用的全股票/指数比较使用 `1,000,000`。手续费和印花税在回测中保持一致。

### 基线模型参数

T3-T5 中的基线模型由 `main.py` 训练，不由 `train.py` 训练：

| 名称 | 估计器和固定设置 |
| --- | --- |
| `LR` | Lasso 回归，`alpha=0.0001` |
| `MLP_R` / `MLP_C` | 一个隐藏层 `(24,)`，`max_iter=100`，`random_state=42`；分别为回归/分类版本 |
| `SVM_R` / `SVM_C` | RBF 核 SVR/SVC，`C=1.0` |
| `XGB_R` | XGBoost 回归，`objective=reg:squarederror`、`max_depth=4`、`learning_rate=0.1`、`subsample=1.0`、`colsample_bytree=0.8`、`tree_method=approx`、`n_jobs=1`；树数量和 `max_bin` 按代码中的 `BASELINE_MAX_BIN` 年限/市场映射执行 |
| `XGB_C` | 使用相同建树参数的 XGBoost 分类器；两年/四年为 `200` 棵树，三年为 `150` 棵，并使用对应的 `max_bin` 映射 |

名称中的 `_R` 表示以 `real_return` 为目标的回归，`_C` 表示以二元 `up_down` 标签为目标的分类。`fit_baseline` 在训练前进行特征缩放，并保持论文代码的训练集/测试集归一化流程。即使某些估计器有自己的固定实现种子，基线阶段仍会从活动平台配置解析种子。

### 平台参数文件结构

每个 `.txt` 实际是 JSON，键为 `schema_version`、`profile`、`system`、`purpose`、`stage_seeds`、`rank_config` 和 `mart_config`。

- `stage_seeds[市场代码][训练年限][阶段]` 解析 `rank`、`mart`、`dqn`、`baseline`、`evaluation` 五类种子。命令行 `--seed` 会覆盖所有阶段；`--seed_config` 会在活动配置上合并 JSON 覆盖。
- `rank_config` 只允许实现层面的 LambdaRank 参数：`max_depth`、`n_estimators`、`subsample`、`colsample_bytree`、`tree_method`。
- `mart_config` 只允许实现层面的 LambdaMART 参数：`max_bin`、`min_child_weight`、`subsample`、`colsample_bytree`、`tree_method`。
- 抽样比例必须在 `(0, 1]`；`max_depth`、`n_estimators`、`max_bin` 必须是正整数。未知字段以及论文锁定的学习率会被拒绝，不会静默改变实验。

参数优先级是有意收窄的：`experiment_core.py` 中论文锁定的值不能覆盖；其余参数先取选定平台配置，再由可选的 `--seed_config`/`--rank_config`/`--mart_config` 提供经过校验的 JSON 覆盖；`--seed` 可在诊断运行中统一替换所有阶段种子。正式 Windows 复现时应不设置这些覆盖项。

程序根据 `platform.system()` 自动选择配置，并在每个 manifest 中记录配置名、文件名和 SHA-256；Windows 正式运行应始终记录 `windows-reference`。

正式 Windows 配置的每一行都使用 `subsample=1.0`、`colsample_bytree=1.0`、`tree_method=approx`；按市场/年限的具体值如下：

| 市场/年限 | 阶段种子（`rank/mart/dqn/baseline/evaluation`） | LambdaRank（`max_depth`、`n_estimators`） | LambdaMART（`max_bin`、`min_child_weight`） |
| --- | --- | --- | --- |
| 主板两年 | `40/41/40/43/19` | `5`、`100` | `32`、`1` |
| 主板三年 | `50/51/10/53/36` | `6`、`100` | `256`、`1` |
| 主板四年 | `60/61/40/63/59` | `6`、`150` | `32`、`1` |
| 创业板两年 | `50/51/50/53/67` | `8`、`25` | `3`、`0.43` |
| 创业板三年 | `60/61/50/63/31` | `6`、`100` | `256`、`1` |
| 创业板四年 | `70/71/50/73/49` | `3`、`50` | `9`、`1` |

这些是 Windows 实现/配置值，不会替代前文列出的论文锁定学习率、LambdaMART 深度或树数量。Linux 参数只为兼容运行单独维护。

### `train.py` 命令行参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--run_dir` | 仓库根目录 | `temp/`、`model/` 和 manifest 的根目录 |
| `--models` | `all` | `rankers`、`dqn` 或二者 `all` |
| `--markets` | 两个市场 | `Main`、`ChiNext` 或逗号分隔值 |
| `--years` | `2,3,4` | 训练窗口，只允许 2、3、4 |
| `--seed` | 配置文件中的种子 | 统一覆盖所有阶段 |
| `--seed_config` | 活动平台配置 | JSON 种子覆盖 |
| `--rank_config` | 活动平台配置 | 未报告的 LambdaRank 实现参数覆盖 |
| `--mart_config` | 活动平台配置 | 未报告的 LambdaMART 实现参数覆盖 |
| `--ranker_tree_method` | `approx` | XGBoost 建树方式 |
| `--t6` | 关闭 | 附加 T6 采样流程 |
| `--t6_markets` | `all` | T6 市场：`Main`、`ChiNext` 或两者 |
| `--t6_max_seeds` | `20` | 每个 T6 比例/模型/市场单元的新运行次数 |
| `--t6_seed_summary` | `data/reproducibility/t6_cpu20_seed_summary.csv` | Ranker/MART 的 T6 种子表 |
| `--t6_dqn_seed_summary` | `data/reproducibility/t6_cpu20_dqn_seed_summary.csv` | DQN 的 T6 种子表 |

`--lr`、`--n_games`、`--gamma`、`--epsilon`、`--eps_end`、`--eps_dec`、`--batch_size`、`--max_mem_size`、`--replace_target_iter` 是上表 DQN 参数。正式复现应保持默认值；特别是 `--lr` 改为非 `0.002` 会被拒绝。

### 评估和绘图命令行参数

`main.py` 支持 `--tables T3,T4,T5,T7`、`--markets all|Main,ChiNext`、`--seed`、`--seed_config`、`--output_dir`、`--export_csvs` 和 `--no_baselines`。`--export_csvs` 会额外生成 `results_long.csv` 和每个表格的论文格式 CSV；`--no_baselines` 只适合诊断时跳过快速基线拟合。DQN 评估固定使用训练出的策略，`--dqn_eval_mode dqn` 不是切换到缓存动作文件。

`Fig_main.py` 支持 `--figures 3,4,5,6,7`、`--run_dir`、`--output_dir`、`--seed`、`--seed_config`、`--ranker_tree_method`、`--n_games` 和 `--force`；`--force` 会在源数据或参数变化后重新生成图表缓存。`Appendix_Fig_main.py` 支持 `--figures C1,C2,C3,C4,C5`、`--run_dir`、`--output_dir`、`--t6_csv`、`--broker_file`、`--broker_column`、`--min_broker_reports` 和 `--force`。

### 固定日期和指标常量

| 常量 | 值 | 用途 |
| --- | --- | --- |
| `TRAIN_END` | `20211206` | 所有训练窗口的结束日 |
| `TEST_START` / `TEST_END` | `20211207` / `20230303` | 统一测试区间 |
| 训练开始日 | 两年：`20191206`；三年：`20181206`；四年：`20171206` | 由 `--years` 选择 |
| 年化 | 每年 `242` 个交易日 | ARR 和 Sharpe 计算 |
| 佣金 | `0.0003` | 买入和卖出佣金 |
| 印花税 | `0.001` | 卖出税费 |
| 排名组合规模 | `4` | 非分类器回测默认选前四只 |

### T6 和 T7 参数

T6 固定对两个市场、三个模型在 `50%`、`60%`、`70%`、`80%`、`90%` 五个采样率下各运行 `20` 次；`100%` 参考行来自同一次运行的 T4 评估。两个种子台账放在 `data/reproducibility/`，用于披露随机性但不提交拟合结果。

T7 从原始横截面 `data/ESG/ESG.csv` 的数值 `ESG` 列计算统一阈值，当前数据为 q25=`5.52`、q50=`6.02`。**NS** 先按 DQN 预测选前四只，再删除低于阈值的持仓；**PI** 先筛出达到阈值的股票，再按预测排名递补，尽量恢复前四只。两个市场在同一档位共用同一阈值。

### 仅用于绘图的敏感性网格

敏感性图使用公开的参数网格，不会悄悄复用主流程模型。图 3 和图 4 的学习率网格为 `0.0001`、`0.001`、`0.002`、`0.01`、`0.1`、`0.2`；图 4 还测试 `n_estimators=800/900/1000/1100/1200` 和 `max_depth=4/5/6/7/8`。图 3(b) 的六个 DQN 学习率单元固定使用主板种子 `36`、创业板种子 `66`。这些只影响敏感性图，并记录在图表 manifest 中，不会替换 T4/T5 的论文锁定配置。

## 系统和设备要求

- Python 3.9.13，64 位 x86（x64）。
- Windows 10/11 x64 CPU 是论文结果的正式参考环境；Linux x86-64 只作为兼容路径。Linux 不要求与 Windows 逐位一致，目标是保持结果整体接近。
- 全流程使用 CPU，不要求 GPU，也不会选择 GPU 设备。
- BLAS、XGBoost 和 PyTorch 固定为单线程，以降低不同机器之间的差异。
- 完整运行建议至少 8 GB 内存和 10 GB 可用磁盘空间。

Windows 正式锁定版本为 pip 24.1.2、NumPy 1.21.5、pandas 1.4.4、scikit-learn 1.2.0、PyTorch 2.0.0+cpu、XGBoost 1.7.6。Linux 兼容文件在可用时请求同一版本；若 Linux 设备无法安装，必须显式设置 `LTR_DQN_RELAXED_RUNTIME=1`，并接受数值差异。

## Windows 正式参数选择

Windows 命令不需要输入参数文件路径。`runtime_config.py` 启动时检测
`platform.system()` 并读取 `parameters_windows.txt`，配置名为
`windows-reference`。这是本地 Windows 和 GitHub Actions 使用的正式配置。

如果同一代码必须在 Linux 上运行，程序才会读取 `parameters_linux.txt`，配置名为
`linux-emergency-compatibility`，用于适配命令和依赖差异；它不是论文参考配置。其他系统会直接报错。

两个文本文件采用 JSON 格式，只存储种子和论文没有明确报告的实现参数。论文正文或附录已经给出的学习率、树深、树数量等参数仍由代码统一锁定，平台文件不能覆盖。每次运行的 manifest 会记录配置名、文件名和 SHA-256 哈希。

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
```

### Linux 命令差异（仅供兼容）

```bash
git clone https://github.com/whyhw-code/LTR-DQN.git
cd LTR-DQN
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-linux.txt
export MPLBACKEND=Agg PYTHONHASHSEED=0 PYTHONUTF8=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE
```

Linux 也可以使用 Conda（兼容路径）：

```bash
conda env create -f environment-linux.yml
conda activate ltr-dqn-linux
```

Linux 不是正式复现环境。若必须在 Linux 上运行，可使用 `conda env create -f environment-linux.yml`、`conda activate ltr-dqn-linux`，再运行 `bash run_linux.sh` 或 `bash run_t6_linux.sh`。脚本默认要求锁定版本；仅在确认无法安装锁定版本时，才使用 `export LTR_DQN_RELAXED_RUNTIME=1`。
脚本固定 CPU 单线程设置，避免 GPU 和 BLAS 并行带来更大的结果偏差。

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

1. 打开仓库的 **Actions** 页面。
2. 选择 **1 - One-click Results and Figures (Windows CPU)**，直接点击 **Run workflow**，生成 T3/T4/T5/T7、全部正文图和附录 C1/C2/C3/C5。
3. 选择 **2 - One-click T6 and Figure C4 (Windows CPU)**，直接点击 **Run workflow**，生成 T6 和附录 C4。
4. 运行结束后下载 artifact，其中包含工作簿、CSV、图片、运行清单和本次使用的 `parameters_windows.txt`。

仓库只有 `main` 分支。两个在线 workflow 都固定使用 `windows-2022` x64 CPU runner，并在训练前断言已经读取 `parameters_windows.txt`。Linux 仅通过本地 shell 脚本运行，不提供 GitHub Actions 入口。

## 复现注意事项

- Windows/Linux 的种子和论文未报告实现参数分别位于 `parameters_windows.txt`、`parameters_linux.txt`；T6 种子表在 `data/reproducibility/`。
- LambdaRank、LambdaMART 每次都从原始数据重新训练；DQN 使用同一次运行产生的 LambdaMART 输出。
- `PYTHONHASHSEED=0`、固定种子、PyTorch 确定性设置、稳定 CSV 排序和 `n_jobs=1` 用于减少平台差异。
- 运行清单记录依赖版本、输入哈希、动作哈希和检查点哈希。Windows 结果是参考值；Linux 清单会记录 `runtime_mode=linux-compat`。即使依赖完全相同，Linux 与 Windows 的 XGBoost/PyTorch 编译器、CPU 指令集和底层数学库仍可能造成小幅数值差异，因此不能承诺逐位一致。
- 复现默认结果时不要修改 `--seed`、`--seed_config`、`--lr`、`--n_games` 或 `--ranker_tree_method`。
