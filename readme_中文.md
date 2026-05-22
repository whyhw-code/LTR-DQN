# 复现包 README（中文校对版）

## 1. 基本信息

- **论文题目**：Enhancing Predictive Selection of Sell-Side Analyst Reports via Learning to Rank and Reinforcement Learning
- **稿件编号**：IJF-D-24-00575R3
- **作者**：Jiaming Liu, Hongyang Wang, Yongli Li, Kaiwei Jia
- **通讯作者邮箱**：liyongli@hit.edu.cn
- **打包日期**：2026.04.13

------

## 2. 仓库结构

仓库结构如下：

```text
LTR-DQN-main/
├── code/
│   ├── data/                 # 复现脚本使用的数据文件
│   ├── model/                # 强化学习模型文件
│   ├── temp/                 # 复现脚本使用的中间文件
│       ├── os/               # 复现脚本使用的中间值
│       ├── seed_summary.csv  # 复现T6使用的种子
│   ├── dl_dqn2.py            # LTR-DQN 模型过程
│   ├── DQN_train.py          # DQN 训练脚本
│   ├── F3.py                 # 复现 Figure 3 的脚本
│   ├── F4.py                 # 复现 Figure 4 的脚本
│   ├── F5.py                 # 复现 Figure 5 的脚本
│   ├── F6.py                 # 复现 Figure 6 的脚本
│   ├── T3C1.py               # 表格复现脚本示例
│   ├── T3C2.py
│   ├── T3M1.py
│   └── ...
├── result/                   # 结果文件和汇总输出
├── readme.md                 # 本 README 文件
└── requirements.txt          # Python 依赖文件
```

大多数可执行脚本，包括 `T*.py`、`F3.py`、`F4.py`、`F5.py`、`F6.py` 和 `DQN_train.py`，均位于 `code/` 目录下。因此，表格生成、图像生成和模型训练脚本均应在进入 `code/` 目录后运行。

------

## 3. 计算环境要求

### 3.1 软件要求

本复现包已在以下软件版本下测试：

- Python: 3.9.16
- PyTorch: 2.0.0
- XGBoost: 1.7.6

模型训练推荐使用 GPU。不过，如果使用已提供的中间结果和预训练输出，表格生成脚本可以直接运行。

### 3.2 依赖安装

本复现包提供一个依赖文件：`requirements.txt`。

请在项目根目录下创建干净的 Python 3.9 环境，并按如下方式安装依赖：

```bash
cd LTR-DQN-main
conda create -n ltr-dqn python=3.9.16
conda activate ltr-dqn
pip install -r requirements.txt
```

所需 Python 包列于 `requirements.txt` 中。

------

## 4. 数据说明

### 4.1 数据来源

本研究使用的数据来源于卖方分析师报告。

### 4.2 数据可获得性

原始分析师报告可从东方财富公开下载。

### 4.3 数据文件和格式

原始数据为 PDF 格式的卖方分析师报告文件。复现脚本直接使用的数据文件已存放在仓库对应的数据目录中。

------

## 5. 仓库内容和中间文件说明

本仓库包含模型训练脚本、代码使用的数据、实验过程中产生的中间文件，以及用于复现论文中表格和图像的脚本。

### 5.1 `code/`

`code/` 目录包含主要可执行脚本及其使用的文件，包括：

- 模型训练代码；
- 表格复现脚本；
- 图像复现脚本；
- 代码直接使用的数据文件；
- 实验过程中生成的中间文件。

### 5.2 `code/temp/`

`code/temp/` 目录存放复现脚本使用的中间文件。

其中：

- `code/temp/seed_summary.csv` 记录了首次运行 Table 6 相关脚本时使用的随机种子。该文件是复现 Table 6 结果所必需的文件，因为当前 Table 6 相关脚本仍继续使用这些已记录的种子以保证复现性。
- `code/temp/oc/batch123/` 包含 LambdaMART 排序阶段生成的中间 CSV 文件，以及模型评估和图像生成过程中使用的中间文件。
- 文件名形如 `meiri_xuanze*.csv` 的文件记录了不同训练年份设置下每日选股数量。若文件名中未标注训练年份，则对应设置为 `train_year = 3`。

`code/temp/oc/batch123/` 中形如特定板块的 `temp_train` 或 `temp_test` CSV 文件，是由前序 LambdaMART 排序阶段生成的中间结果。具体而言，原始整理数据：

```text
code/data/0060merge_open_close_final.csv
code/data/3068merge_open_close_final.csv
```

首先分别由 `T4M10.py` 和 `T4C10.py` 处理。其中，`T4M10.py` 对应主板市场，即 `0060`；`T4C10.py` 对应创业板市场，即 `3068`。运行这两个脚本并设置不同参数后，可生成 `code/temp/oc/batch123/` 下的 `temp_train` 和 `temp_test` CSV 文件。

其中：

- `temp_train` 文件是训练期样本上的 LambdaMART 排序结果，用于后续 DQN 训练；
- `temp_test` 文件是样本外测试期样本上的 LambdaMART 排序结果，用于后续 DQN 测试、收益评价和表格复现。

这些 CSV 文件通常包含以下字段：

```text
qid_date, stock_code, real_return, prediction, close, pclose
```

其中，`prediction` 为 LambdaMART 模型对每日候选股票生成的排序得分，`real_return`、`close` 和 `pclose` 用于后续收益计算、DQN 状态构造和策略评价。

文件名记录了生成该文件时使用的参数设置。例如：

```text
0060temp_test_ndcg_train3_0.0003_0.001_0.001_5_1000.csv
```

表示该文件对应主板市场 `0060`、测试期样本、`ndcg` 排序指标、3 年训练窗口、手续费 `0.0003`、印花税 `0.001`、LambdaMART 学习率 `0.001`、最大树深度 `5` 和估计器数量 `1000`。

又如：

```text
3068temp_train_ndcg_train3_0.0003_0.001_0.1_6_1000.csv
```

表示该文件对应创业板市场 `3068`、训练期样本、`ndcg` 排序指标、3 年训练窗口、手续费 `0.0003`、印花税 `0.001`、LambdaMART 学习率 `0.1`、最大树深度 `6` 和估计器数量 `1000`。

随后，`DQN_train.py` 会读取 `code/temp/oc/batch123/` 下对应的 `temp_train` 文件，并基于 LambdaMART 排序结果训练 DQN 投资组合选择模型。训练完成后，DQN 模型文件保存至：

```text
code/model/batch123/
```

后续测试和结果复现脚本，例如 `T4M12.py` 和 `T4C12.py`，会同时使用 `code/model/batch123/` 下的 DQN 模型文件，以及 `code/temp/oc/batch123/` 下对应的 `temp_test` 文件，生成论文中的投资组合表现结果。其他后续表格和图像复现脚本也会基于这些中间结果和模型输出复现论文中的表格与图像。

在当前发布的复现包中，上述中间文件已经提供。`T4M10.py` 和 `T4C10.py` 中对应的保存语句默认处于注释状态，以避免用户在复现过程中重复运行脚本并覆盖已提供的标准中间文件。若用户希望从原始整理数据重新生成这些中间文件，可以取消相应 `to_csv` 保存语句的注释，并使用相同参数设置重新运行 `T4M10.py` 和 `T4C10.py`。


### 5.3 `result/`

`result/` 目录存放最终输出、汇总结果以及用于表格和图像复现的文件。例如，`result/batch123/基准+模型结果对比.xlsx` 用于复现收益曲线图。

------

## 6. 代码说明

### 6.1 主要脚本

- `code/DQN_train.py`：DQN 训练脚本，如需重新训练模型时使用
- `code/dl_dqn2.py`：LTR-DQN 模型过程
- `code/T*.py`：复现论文表格的脚本
- `code/F3.py`：复现 Figure 3 的脚本
- `code/F4.py`：复现 Figure 4 的脚本
- `code/F5.py`：复现 Figure 5 的脚本
- `code/F6.py`：复现 Figure 6 的脚本

------

## 7. 表格和图像复现

### 7.1 脚本命名规则

用于生成论文表格结果的脚本遵循统一命名规则。

#### （1）普通表格

命名格式：

```text
T[table number][market][row number].py
```

- `T`：表示 Table
- 表格编号：对应论文中的表格编号，例如 `T4` 表示 Table 4
- `C`：创业板市场
- `M`：主板市场
- 行号：对应表格中的具体行

示例：

- `T4C3.py` → Table 4，创业板市场，第 3 行
- `T5M12.py` → Table 5，主板市场，第 12 行

#### （2）Table 6 的特殊命名规则

命名格式：

```text
T6[market][ratio]_[model].py
```

- `C`：创业板市场
- `M`：主板市场

比例映射：

- `5` = 50%
- `6` = 60%
- `7` = 70%
- `8` = 80%
- `9` = 90%

模型标识：

- `_1`：LambdaRank
- `_2`：LambdaMART + LTR-DQN

示例：

- `T6C5_1.py` → Table 6，创业板，50%，LambdaRank
- `T6M6_2.py` → Table 6，主板，60%，LambdaMART + LTR-DQN

### 7.2 代码和结果对应关系

论文中的表格和脚本对应关系如下：

- Table 3：由 `T3*.py` 生成
- Table 4：由 `T4*.py` 生成
- Table 5：由 `T5*.py` 生成
- Table 6：由 `T6*_*.py` 生成
- Table 7：由 `T7*.py` 生成

其中：

- `C` 表示创业板市场
- `M` 表示主板市场

每个脚本生成对应表格中的一行或一列结果。

### 7.3 图像复现

论文中的图像根据用途采用不同方式生成。

#### Figure 1 和 Figure 2

Figure 1 和 Figure 2 是示意图，而不是由数值数据生成的图。

- Figure 1 展示 LTR-DQN 投资组合构建框架。
- Figure 2 展示本文采用的 DQN 机制。

这两个图由绘图软件手动绘制，不依赖 Python 脚本的数值输出。

#### Figure 3

Figure 3 可通过 `code/` 目录中的以下脚本复现：

```bash
cd LTR-DQN-main/code
python F3.py
```

`F3.py` 用于复现 Figure 3 中的超参数敏感性分析图。`F3.py` 所需输入数据已随脚本一并包含在仓库中。运行该脚本时，请保持原始文件夹结构不变。

#### Figure 4

Figure 4 可通过 `code/` 目录中的以下脚本复现：

```bash
cd LTR-DQN-main/code
python F4.py
```

`F4.py` 用于复现 Figure 4 中的超参数敏感性分析图。`F4.py` 所需输入数据已随脚本一并包含在仓库中。运行该脚本时，请保持原始文件夹结构不变。

#### Figure 5 和 Figure 6

Figure 5 和 Figure 6 可通过 `code/` 目录中的以下脚本复现：

```bash
cd LTR-DQN-main/code
python F5.py
python F6.py
```

`F5.py` 用于复现 Figure 5。它依赖以下汇总收益对比文件：

```text
result/batch123/基准+模型结果对比.xlsx
```

该 Excel 文件包含基准方法和本文提出的 LTR-DQN 模型的收益曲线结果。

`F6.py` 用于复现 Figure 6。除使用：

```text
result/batch123/基准+模型结果对比.xlsx
```

外，还依赖每日选股数量文件：

```text
code/temp/oc/batch123/meiri_xuanze.csv
```

Excel 文件提供收益曲线数据，`meiri_xuanze.csv` 提供 LTR-DQN 下每日选股数量。

上述路径均为相对于仓库根目录的路径。`code/` 与 `result/` 在 `LTR-DQN-main/` 下处于同一级目录。由于脚本从 `code/` 目录运行，代码中读取根目录 `result/` 下文件时，应使用类似 `../result/...` 的相对路径。

------

## 8. 结果复现说明

### 8.1 复现流程

本复现包可通过两种方式使用。

#### 路径 A：使用已提供的中间结果和预训练输出复现论文表格

这是推荐的复现检查路径。该路径**不需要重新训练 DQN 模型**。

1. 克隆或下载仓库。
2. 使用 `requirements.txt` 创建 Python 环境并安装依赖。
3. 进入 `code/` 目录。
4. 运行对应的 `T*.py` 脚本，复现论文中的表格。
5. 将打印输出与论文中的结果进行核对。

示例：

```bash
cd LTR-DQN-main
conda create -n ltr-dqn python=3.9.16
conda activate ltr-dqn
pip install -r requirements.txt

cd code
python T3M1.py
python T3M2.py
python T4M1.py
```

#### 路径 B：重新训练 DQN 模型后再复现结果

该路径为可选路径，仅当用户希望重新生成 DQN 模型，而不是使用已提供的预训练模型文件时需要。

1. 使用 `requirements.txt` 创建 Python 环境并安装依赖。
2. 进入 `code/` 目录。
3. 使用所需命令行参数运行 `DQN_train.py`。
4. 重新训练后，运行对应的 `T*.py` 脚本复现表格。

除非特别需要重新训练 DQN 模型，否则建议用户采用路径 A。

### 8.2 环境配置

建议使用 Python 3.9.16 创建干净的 conda 环境：

```bash
cd LTR-DQN-main
conda create -n ltr-dqn python=3.9.16
conda activate ltr-dqn
pip install -r requirements.txt
```

### 8.3 工作目录

所有表格生成、图像生成和训练脚本均位于 `code/` 目录下。因此，安装依赖后，请先进入 `code/` 目录再运行任何脚本：

```bash
cd LTR-DQN-main
conda activate ltr-dqn
pip install -r requirements.txt

cd code
python T3M1.py
```

请不要直接在项目根目录下运行表格或图像生成脚本。这些脚本假定当前工作目录为 `code/`。

### 8.4 数据准备

复现脚本所需数据文件存放在：

```text
code/data/
```

运行复现脚本时请保持原始文件夹结构不变。

数据来源和获取方式见第 4 节。

### 8.5 预训练模型

复现论文报告的表格结果**不需要重新训练模型**。

相关模型已完成训练。复现时可直接使用项目中提供的已训练结果、中间结果或模型文件。

因此：

- 使用已提供输出复现表格时，`DQN_train.py` 不是必要步骤。
- 除非需要重新训练，否则不需要运行训练脚本。
- 复现表格结果时，直接运行相应的 `T*.py` 脚本即可。

### 8.6 重新生成 LambdaMART 中间文件

`code/temp/oc/batch123/` 下的 `temp_train` 和 `temp_test` 文件由 `T4M10.py` 和 `T4C10.py` 生成。其中，`T4M10.py` 对应主板市场 `0060`，`T4C10.py` 对应创业板市场 `3068`。这两个脚本会读取 `code/data/` 下的原始整理数据，训练 LambdaMART 排序模型，并输出包含 `qid_date`、`stock_code`、`real_return`、`prediction`、`close` 和 `pclose` 的中间 CSV 文件。

在当前复现包中，这些中间文件已经提供。为避免用户在复现过程中重复运行脚本并覆盖已提供的标准中间文件，`T4M10.py` 和 `T4C10.py` 中的保存语句默认处于注释状态。若用户希望从原始整理数据重新生成这些文件，需要先取消如下语句的注释：

```python
temp.to_csv(f'temp/oc/batch{test_batch}/{dapan_code}temp_{train_or_test}_{m}_train{train_year}_{shouxufei}_{yinhaushui}_{learning_rate}.csv')
```

随后在 `code/` 目录下运行相应命令。`train_or_test`、`shouxufei`、`yinhaushui` 和 `learning_rate` 可通过命令行输入；其中 `train_or_test=train` 生成训练期文件，`train_or_test=test` 生成样本外测试期文件。

| 目标文件 | 生成命令 |
|---|---|
| `0060temp_train_ndcg_train3_0.0003_0.001_0.001_5_1000.csv` | `python T4M10.py --train_or_test train --shouxufei 0.0003 --yinhaushui 0.001 --learning_rate 0.001` |
| `0060temp_test_ndcg_train3_0.0003_0.001_0.001_5_1000.csv` | `python T4M10.py --train_or_test test --shouxufei 0.0003 --yinhaushui 0.001 --learning_rate 0.001` |
| `0060temp_test_ndcg_train3_0.0001_0.001_0.001_5_1000.csv` | `python T4M10.py --train_or_test test --shouxufei 0.0001 --yinhaushui 0.001 --learning_rate 0.001` |
| `0060temp_test_ndcg_train3_0.0003_0.001_0.0001_5_1000.csv` | `python T4M10.py --train_or_test test --shouxufei 0.0003 --yinhaushui 0.001 --learning_rate 0.0001` |
| `0060temp_test_ndcg_train3_0.0003_0.001_0.01_5_1000.csv` | `python T4M10.py --train_or_test test --shouxufei 0.0003 --yinhaushui 0.001 --learning_rate 0.01` |
| `0060temp_test_ndcg_train3_0.0003_0.001_0.1_5_1000.csv` | `python T4M10.py --train_or_test test --shouxufei 0.0003 --yinhaushui 0.001 --learning_rate 0.1` |
| `0060temp_test_ndcg_train3_0.0003_0.001_0.2_5_1000.csv` | `python T4M10.py --train_or_test test --shouxufei 0.0003 --yinhaushui 0.001 --learning_rate 0.2` |
| `3068temp_train_ndcg_train3_0.0003_0.001_0.1_6_1000.csv` | `python T4C10.py --train_or_test train --shouxufei 0.0003 --yinhaushui 0.001 --learning_rate 0.1` |
| `3068temp_test_ndcg_train3_0.0003_0.001_0.1_6_1000.csv` | `python T4C10.py --train_or_test test --shouxufei 0.0003 --yinhaushui 0.001 --learning_rate 0.1` |

对于 `train_year=2` 或 `train_year=4` 的文件，需要在脚本中将 `train_year` 设置为对应值后再运行同类命令。对于文件名中最大树深度或估计器数量不同的文件，例如 `_4_1000`、`_6_1000`、`_5_800`、`_5_900`、`_5_1100` 和 `_5_1200`，需要在脚本中调整对应的 LambdaMART 参数后生成。


### 8.7 重新训练 DQN 模型

DQN 训练脚本为：

```text
code/DQN_train.py
```

该脚本应在 `code/` 目录下运行。

两个必需命令行参数如下：

| 参数 | 含义 | 可选值 |
|---|---|---|
| `--bankuaicode` | 市场代码 | `0060` 表示主板；`3068` 表示创业板 |
| `--train_year` | 训练年份长度 | `2`、`3` 或 `4` |

脚本中使用的默认 DQN 超参数如下：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--lr` | `0.002` | 学习率 |
| `--dec` | `0.00015` | epsilon 衰减率 |
| `--n_games` | `31` | 训练轮次 |
| `--gamma` | `0.9` | 折扣因子 |
| `--epsilon` | `1.0` | 初始 epsilon |
| `--eps_end` | `0.03` | 最小 epsilon |
| `--batch_size` | `32` | replay batch 大小 |
| `--fc1_dims` | `256` | 第一隐藏层神经元数量 |
| `--fc2_dims` | `128` | 第二隐藏层神经元数量 |

当前实现中，实验使用的 LTR 排序目标固定为 `ndcg`，运行 `DQN_train.py` 时无需单独指定。

示例：

```bash
cd LTR-DQN-main
conda activate ltr-dqn
cd code

# 主板，3 年训练窗口
python DQN_train.py --bankuaicode 0060 --train_year 3

# 创业板，3 年训练窗口
python DQN_train.py --bankuaicode 3068 --train_year 3

# 主板，2 年训练窗口
python DQN_train.py --bankuaicode 0060 --train_year 2

# 创业板，4 年训练窗口
python DQN_train.py --bankuaicode 3068 --train_year 4
```

训练后的 DQN 模型文件将保存至：

```text
code/model/batch123/
```

模型文件命名规则如下：

```text
{bankuaicode}_ndcg_{train_year}year_top4_train{train_year}TESToc
{bankuaicode}_ndcg_{train_year}year_top4_train{train_year}TESToc_{lr}
```

其中，`bankuaicode` 表示市场代码，`0060` 为主板，`3068` 为创业板；`ndcg` 表示当前实验使用的 LTR 排序目标；`train_year` 表示训练窗口长度；`top4` 表示 DQN 阶段使用前 4 个候选；`lr` 表示学习率。当使用默认学习率 `0.002` 时，模型文件名通常不额外添加学习率后缀；当使用其他学习率进行超参数敏感性分析时，文件名会在末尾添加对应的学习率后缀。

`code/model/batch123/` 中保留的模型文件及其对应训练命令如下。所有命令均应在 `code/` 目录下运行：

```bash
cd LTR-DQN-main/code
```

| 模型文件 | 对应训练命令 |
|---|---|
| `0060_ndcg_2year_top4_train2TESToc` | `python DQN_train.py --bankuaicode 0060 --train_year 2 --lr 0.002` |
| `0060_ndcg_3year_top4_train3TESToc` | `python DQN_train.py --bankuaicode 0060 --train_year 3 --lr 0.002` |
| `0060_ndcg_3year_top4_train3TESToc_0.0001` | `python DQN_train.py --bankuaicode 0060 --train_year 3 --lr 0.0001` |
| `0060_ndcg_3year_top4_train3TESToc_0.001` | `python DQN_train.py --bankuaicode 0060 --train_year 3 --lr 0.001` |
| `0060_ndcg_3year_top4_train3TESToc_0.01` | `python DQN_train.py --bankuaicode 0060 --train_year 3 --lr 0.01` |
| `0060_ndcg_3year_top4_train3TESToc_0.1` | `python DQN_train.py --bankuaicode 0060 --train_year 3 --lr 0.1` |
| `0060_ndcg_3year_top4_train3TESToc_0.2` | `python DQN_train.py --bankuaicode 0060 --train_year 3 --lr 0.2` |
| `0060_ndcg_4year_top4_train4TESToc` | `python DQN_train.py --bankuaicode 0060 --train_year 4 --lr 0.002` |
| `3068_ndcg_2year_top4_train2TESToc` | `python DQN_train.py --bankuaicode 3068 --train_year 2 --lr 0.002` |
| `3068_ndcg_3year_top4_train3TESToc` | `python DQN_train.py --bankuaicode 3068 --train_year 3 --lr 0.002` |
| `3068_ndcg_3year_top4_train3TESToc_0.0001` | `python DQN_train.py --bankuaicode 3068 --train_year 3 --lr 0.0001` |
| `3068_ndcg_3year_top4_train3TESToc_0.001` | `python DQN_train.py --bankuaicode 3068 --train_year 3 --lr 0.001` |
| `3068_ndcg_3year_top4_train3TESToc_0.01` | `python DQN_train.py --bankuaicode 3068 --train_year 3 --lr 0.01` |
| `3068_ndcg_3year_top4_train3TESToc_0.1` | `python DQN_train.py --bankuaicode 3068 --train_year 3 --lr 0.1` |
| `3068_ndcg_3year_top4_train3TESToc_0.2` | `python DQN_train.py --bankuaicode 3068 --train_year 3 --lr 0.2` |
| `3068_ndcg_4year_top4_train4TESToc` | `python DQN_train.py --bankuaicode 3068 --train_year 4 --lr 0.002` |

上述模型文件用于支持论文报告结果及相关超参数敏感性分析的复现。对于论文主要表格结果，通常使用默认学习率 `0.002` 下的模型；带有学习率后缀的模型主要用于学习率敏感性分析。

重新训练后，用户可以运行相应的 `T*.py` 脚本来复现表格结果.

### 8.8 复现表格结果

首先进入 `code/` 目录，然后按需运行对应脚本。例如：

```bash
cd LTR-DQN-main/code
python T4M1.py
python T4M2.py
python T4C1.py
python T4C2.py
```

或：

```bash
cd LTR-DQN-main/code
python T6M5_1.py
python T6M5_2.py
```

运行对应脚本后，将生成相应表格中的结果。

关于 Table 7：Table 7 的前三行，即市场指数、基准投资组合和不含 ESG 的 LTR-DQN，与 Table 4 中报告的基准/参考结果相同。额外的 Table 7 脚本用于生成 ESG 相关行，包括 Negative Screening 和 Positive Investing 结果。

### 8.9 复现图像结果

Figure 3 至 Figure 6 可在 `code/` 目录下复现：

```bash
cd LTR-DQN-main/code
python F3.py
python F4.py
python F5.py
python F6.py
```

运行这些脚本前，请确保第 7.3 节中说明的依赖文件均可用。

### 8.10 随机性与复现说明

DQN 训练过程包含随机成分，例如随机初始化、epsilon-greedy 探索和采样。

对于 Table 6 相关实验，原始运行时使用的随机种子记录在：

```text
code/temp/seed_summary.csv
```

为保证复现性，当前 Table 6 相关脚本继续使用这些已记录的种子。因此，复现 Table 6 时必须保留 `code/temp/seed_summary.csv`。

对于论文表格结果的直接复现，建议用户使用仓库中提供的已训练模型文件、已记录随机种子和中间输出。如果用户选择重新训练 DQN 模型，而不是使用已提供文件，则可能由于随机训练过程以及硬件或底层库实现差异而产生数值差异。

### 8.11 注意事项

- 表格生成、图像生成和模型训练脚本均应在 `code/` 目录下运行。
- 请保持原始仓库结构不变，尤其是 `code/data/`、`code/model/`、`code/temp/` 和 `result/`。
- 请确保中间文件和预训练结果可用。
- 如有必要，可以重新训练模型，但复现论文报告的表格结果不要求重新训练。

------

## 9. 预计运行时间

实际运行时间取决于硬件配置、操作系统、CPU/GPU 可用性、磁盘 I/O，以及用户是使用已提供的中间输出还是重新训练 DQN 模型。因此，此处不报告固定总运行时间。

一般而言：

- 使用已提供中间结果时，大多数普通单个表格生成脚本通常可在数秒至数分钟内完成。
- Table 6 脚本由于涉及重复采样实验，单个脚本可能需要更长时间，具体取决于硬件配置。
- 当所需汇总结果文件可用时，Figure 3 至 Figure 6 可通过对应脚本生成。
- 模型重新训练比直接复现表格需要更长时间，并且对 GPU 可用性更敏感。
- 所有表格的完整复现时间会因机器配置差异而变化。如果用户选择重新训练模型或重新生成中间结果，运行时间会更长。

------

## 10. 硬件要求

推荐硬件配置如下：

- CPU：多核处理器，推荐 4 核及以上
- 内存：至少 16GB，推荐 32GB
- GPU：推荐支持 CUDA 11.7 的 NVIDIA GPU
- Python：3.9.16

说明：

- GPU 环境下项目运行效率更高。
- CPU 上运行可能较慢，尤其是模型重新训练和重复采样实验。

------

## 11. 其他说明

请按照上述说明运行，并保持仓库结构不变。如果出现路径错误，请检查当前工作目录是否为 `code/`。
