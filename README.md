# Reproducibility Package README

## 1. General Information

- **Paper Title**: Enhancing Predictive Selection of Sell-Side Analyst Reports via Learning to Rank and Reinforcement Learning
- **Manuscript ID**: IJF-D-24-00575R3
- **Authors**: Jiaming Liu, Hongyang Wang, Yongli Li, Kaiwei Jia
- **Corresponding Author Email**: liyongli@hit.edu.cn
- **Date of Packaging**: 2026.04.29

------

## 2. Repository Structure

The repository is organized as follows:

```text
LTR-DQN-main/
├── code/
│   ├── data/                   # data files used by the reproduction scripts
│   ├── model/                  # reinforcement learning model files
│   ├── temp/                   # intermediate files used by the reproduction scripts
│       ├── os/                 # Intermediate values used by the reproduction script
│       ├── seed_summary.csv    # Reproduce the seed used by T6
│   ├── dl_dqn2.py              # LTR-DQN model procedures
│   ├── DQN_train.py            # DQN training script
│   ├── F3.py                   # script for reproducing Figure 3
│   ├── F4.py                   # script for reproducing Figure 4
│   ├── F5.py                   # script for reproducing Figure 5
│   ├── F6.py                   # script for reproducing Figure 6
│   ├── F7a.py                  # script for reproducing Figure 7(a)
│   ├── F7b.py                  # script for reproducing Figure 7(b)
│   ├── T3C1.py                 # example script for table reproduction
│   ├── T3C2.py
│   ├── T3M1.py
│   └── ...
├── result/          # result files and summarized outputs
├── readme.md        # this README file
└── requirements.txt  # Python dependency file
```

Most executable scripts, including `T*.py`, `F3.py`, `F4.py`, `F5.py`, `F6.py`, `F7a.py`, `F7b.py`, and `DQN_train.py`, are located in the `code/` directory. Therefore, table-generation, figure-generation, and model-training scripts should be executed after entering the `code/` directory.

------

## 3. Computational Requirements

### 3.1 Software Requirements

The reproducibility package has been tested with the following software versions:

- Python: 3.9.16
- PyTorch: 2.0.0
- XGBoost: 1.7.6

A GPU is recommended for model training. However, the table-generation scripts can be run using the provided intermediate results and pre-trained outputs.

### 3.2 Dependency Installation

This reproducibility package provides a single dependency file, `requirements.txt`.

Please create a clean Python 3.9 environment from the project root directory and install the dependencies as follows:

```bash
cd LTR-DQN-main
conda create -n ltr-dqn python=3.9.16
conda activate ltr-dqn
pip install -r requirements.txt
```

The required Python packages are listed in `requirements.txt`.

------

## 4. Data Description

### 4.1 Data Sources

The data used in this study are derived from sell-side analyst reports.

### 4.2 Data Availability

The original analyst reports are publicly available and can be downloaded from Eastmoney.

### 4.3 Data Files and Format

The original data consist of sell-side analyst report files in PDF format. The files used directly by the reproduction scripts are stored in the repository under the corresponding data folders.

------

## 5. Repository Content and Intermediate Files

The repository contains training scripts, data used by the scripts, intermediate files generated during the experiments, and scripts for reproducing the tables and figures reported in the paper.

### 5.1 `code/`

The `code/` directory contains the main executable scripts and the files used by these scripts, including:

- model-training code;
- table-reproduction scripts;
- figure-reproduction scripts;
- data files used directly by the code;
- intermediate files generated during the experimental process.

### 5.2 `code/temp/`

The `code/temp/` directory stores intermediate files used by the reproduction scripts.

In particular:

- `code/temp/seed_summary.csv` records the random seeds used during the initial execution of Table 6-related scripts. This file is required for reproducing Table 6 results, because the current Table 6 scripts continue to use these recorded seeds to ensure reproducibility.
- `code/temp/oc/batch123/` contains intermediate files used in the model evaluation and figure-generation process.
- Files named in the form `meiri_xuanze*.csv` record the daily number of selected stocks under different training-year settings. When no training-year suffix is shown, the corresponding setting is `train_year = 3`.
- Files named in the form of market-specific `temp_train` or `temp_test` CSV files store ranking scores produced by the LambdaMART model. These files are generated from the corresponding Table 4-related scripts and are used in subsequent reinforcement-learning training and testing.

These intermediate files are provided to support direct reproduction of the reported tables and figures without requiring users to regenerate every intermediate output from the original PDF files. The current reproduction package therefore focuses on reproducing the reported experimental results from the prepared data and intermediate files included in the repository.

### 5.3 `result/`

The `result/` directory stores final outputs, summarized results, and files used for table and figure reproduction. For example, `result/batch123/基准+模型结果对比.xlsx` is used for reproducing the return-curve figures.

------

## 6. Code Description

### 6.1 Main Scripts

- `code/DQN_train.py`: DQN training script, used if model retraining is required
- `code/dl_dqn2.py`: LTR-DQN model procedures
- `code/T*.py`: scripts for reproducing tables in the paper
- `code/F3.py`: script for reproducing Figure 3
- `code/F4.py`: script for reproducing Figure 4
- `code/F5.py`: script for reproducing Figure 5
- `code/F6.py`: script for reproducing Figure 6
- `code/F7a.py`: script for reproducing Figure 7(a)
- `code/F7b.py`: script for reproducing Figure 7(b)

------

## 7. Table and Figure Reproduction

### 7.1 Script Naming Rules

The scripts used to generate table results in the paper follow a unified naming convention.

#### (1) General Tables

Naming format:

```text
T[table number][market][row number].py
```

- `T`: indicates Table
- Table number: corresponds to the table number in the paper, e.g., `T4` denotes Table 4
- `C`: ChiNext market
- `M`: Main Board market
- Row number: corresponds to the specific row in the table

Examples:

- `T4C3.py` → Table 4, ChiNext market, Row 3
- `T5M12.py` → Table 5, Main Board market, Row 12

#### (2) Special Rules for Table 6

Naming format:

```text
T6[market][ratio]_[model].py
```

- `C`: ChiNext market
- `M`: Main Board market

Ratio mapping:

- `5` = 50%
- `6` = 60%
- `7` = 70%
- `8` = 80%
- `9` = 90%

Model identifiers:

- `_1`: LambdaRank
- `_2`: LambdaMART + LTR-DQN

Examples:

- `T6C5_1.py` → Table 6, ChiNext, 50%, LambdaRank
- `T6M6_2.py` → Table 6, Main Board, 60%, LambdaMART + LTR-DQN

### 7.2 Mapping Between Code and Results

The correspondence between tables in the paper and scripts is as follows:

- Table 3: generated by `T3*.py`
- Table 4: generated by `T4*.py`
- Table 5: generated by `T5*.py`
- Table 6: generated by `T6*_*.py`
- Table 7: generated by `T7*.py`

Where:

- `C` denotes ChiNext market
- `M` denotes Main Board market

Each script generates one row or one column of results in the corresponding table.

### 7.3 Figure Reproduction

The figures in the paper were generated in different ways depending on their purpose.

#### Figure 1 and Figure 2

Figure 1 and Figure 2 are schematic diagrams rather than data-generated plots.

- Figure 1 illustrates the overall LTR-DQN portfolio construction framework.
- Figure 2 illustrates the DQN mechanism adopted in this study.

These two figures were manually drawn using diagram/plotting software and do not rely on numerical output from the Python scripts.

#### Figure 3

Figure 3 can be reproduced using the following script located in the `code/` directory:

```bash
cd LTR-DQN-main/code
python F3.py
```

`F3.py` reproduces the hyperparameter sensitivity analysis reported in Figure 3. The input data required by `F3.py` are included in the repository together with the script. Please keep the original folder structure unchanged when running the script.

#### Figure 4

Figure 4 can be reproduced using the following script located in the `code/` directory:

```bash
cd LTR-DQN-main/code
python F4.py
```

`F4.py` reproduces the hyperparameter sensitivity analysis reported in Figure 4. The input data required by `F4.py` are included in the repository together with the script. Please keep the original folder structure unchanged when running the script.

#### Figure 5 and Figure 6

Figure 5 and Figure 6 can be reproduced using the following scripts located in the `code/` directory:

```bash
cd LTR-DQN-main/code
python F5.py
python F6.py
python F7a.py
python F7b.py
```

`F5.py` reproduces Figure 5. It depends on the summarized return-comparison file:

```text
result/batch123/基准+模型结果对比.xlsx
```

This Excel file contains the return-curve results of the benchmark methods and the proposed LTR-DQN model.

`F6.py` reproduces Figure 6. In addition to using:

```text
result/batch123/基准+模型结果对比.xlsx
```

it also depends on the daily stock-selection file:

```text
code/temp/oc/batch123/meiri_xuanze.csv
```

The Excel file provides the return-curve data, while `meiri_xuanze.csv` provides the daily number of selected stocks under LTR-DQN.

#### Figure 7

Figure 7 consists of two subplots and can be reproduced using the following scripts located in the `code/` directory:

```bash
cd LTR-DQN-main/code
python F7a.py
python F7b.py
```

- `F7a.py` reproduces Figure 7(a).
- `F7b.py` reproduces Figure 7(b).

The two scripts generate the subplots of Figure 7 separately. Please ensure that the required input data files are available and that the original folder structure is preserved when running the scripts.

The paths above are described relative to the repository root. The `code/` and `result/` directories are at the same level under `LTR-DQN-main/`. Since the scripts are executed from the `code/` directory, files stored in the root-level `result/` directory should be accessed in code using relative paths such as `../result/...`.

------

## 8. Instructions to Reproduce Results

### 8.1 Reproducibility Workflow

The reproducibility package can be used in two ways.

#### Path A: Reproduce the reported tables using provided intermediate results and pre-trained outputs

This is the recommended path for reproducibility checking. It does **not** require retraining the DQN models.

1. Clone or download the repository.
2. Create the Python environment and install dependencies using `requirements.txt`.
3. Enter the `code/` directory.
4. Run the corresponding `T*.py` scripts to reproduce the tables reported in the paper.
5. Check the printed outputs against the values reported in the manuscript.

Example:

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

#### Path B: Re-train the DQN models and then reproduce the results

This path is optional. It is only required if users want to regenerate the DQN models instead of using the provided pre-trained model files.

1. Create the Python environment and install dependencies using `requirements.txt`.
2. Enter the `code/` directory.
3. Run `DQN_train.py` with the required command-line arguments.
4. After retraining, run the corresponding `T*.py` scripts to reproduce the tables.

Unless DQN model retraining is specifically required, users are advised to follow Path A.

### 8.2 Environment Setup

It is recommended to use a clean conda environment with Python 3.9.16:

```bash
cd LTR-DQN-main
conda create -n ltr-dqn python=3.9.16
conda activate ltr-dqn
pip install -r requirements.txt
```

### 8.3 Working Directory

All table-generation, figure-generation, and training scripts are located in the `code/` directory. Therefore, after installing the dependencies, please enter the `code/` directory before running any script:

```bash
cd LTR-DQN-main
conda activate ltr-dqn
pip install -r requirements.txt

cd code
python T3M1.py
```

Please do not run the table-generation or figure-generation scripts directly from the project root directory. The scripts assume that the current working directory is `code/`.

### 8.4 Data Preparation

The data files required by the reproduction scripts are stored under:

```text
code/data/
```

Please keep the provided folder structure unchanged when running the reproduction scripts.

For details on data sources and acquisition methods, see Section 4.

### 8.5 Pre-trained Models

Reproducing the reported table results does **not require retraining the model**.

The relevant models have already been trained. For reproduction, directly use the existing trained results, intermediate results, or model files provided in the project.

Therefore:

- `DQN_train.py` is not a required step for reproducing the table results using the provided outputs.
- Unless necessary, the training scripts do **not** need to be executed.
- To reproduce table results, directly run the corresponding `T*.py` scripts.

### 8.6 Re-training DQN Models

The DQN training script is:

```text
code/DQN_train.py
```

It should be run from the `code/` directory.

The two required command-line arguments are:

| Argument | Meaning | Available values |
|---|---|---|
| `--bankuaicode` | Market code | `0060` for Main Board; `3068` for ChiNext |
| `--train_year` | Number of training years | `2`, `3`, or `4` |

The default DQN hyperparameters used in the script are:

| Argument | Default value | Meaning |
|---|---:|---|
| `--lr` | `0.002` | Learning rate |
| `--dec` | `0.00015` | Epsilon decay rate |
| `--n_games` | `31` | Number of training episodes |
| `--gamma` | `0.9` | Discount factor |
| `--epsilon` | `1.0` | Initial epsilon |
| `--eps_end` | `0.03` | Minimum epsilon |
| `--batch_size` | `32` | Replay batch size |
| `--fc1_dims` | `256` | Number of neurons in the first hidden layer |
| `--fc2_dims` | `128` | Number of neurons in the second hidden layer |

The LTR ranking objective used in the experiments is fixed as `ndcg` in the current implementation and does not need to be specified separately when running `DQN_train.py`.

Examples:

```bash
cd LTR-DQN-main
conda activate ltr-dqn
cd code

# Main Board, 3-year training window
python DQN_train.py --bankuaicode 0060 --train_year 3

# ChiNext, 3-year training window
python DQN_train.py --bankuaicode 3068 --train_year 3

# Main Board, 2-year training window
python DQN_train.py --bankuaicode 0060 --train_year 2

# ChiNext, 4-year training window
python DQN_train.py --bankuaicode 3068 --train_year 4
```

The trained DQN model files will be saved under:

```text
code/model/batch123/
```

After retraining, users can run the corresponding `T*.py` scripts to reproduce the table results.

### 8.7 Reproducing Table Results

First enter the `code/` directory, then run the corresponding scripts as needed. For example:

```bash
cd LTR-DQN-main/code
python T4M1.py
python T4M2.py
python T4C1.py
python T4C2.py
```

or:

```bash
cd LTR-DQN-main/code
python T6M5_1.py
python T6M5_2.py
```

Executing the corresponding scripts will generate the results for the respective tables.

Note for Table 7: the first three rows of Table 7, namely market indices, baseline portfolios, and LTR-DQN without ESG, are the same baseline/reference results reported in Table 4. The additional Table 7 scripts are used to generate the ESG-related rows, including Negative Screening and Positive Investing results.

### 8.8 Reproducing Figure Results

Figure 3 to Figure 7 can be reproduced from the `code/` directory:

```bash
cd LTR-DQN-main/code
python F3.py
python F4.py
python F5.py
python F6.py
python F7a.py
python F7b.py
```

Please make sure that the required files described in Section 7.3 are available before running these scripts.

### 8.9 Notes on Randomness and Reproducibility

The DQN training process involves stochastic components such as random initialization, epsilon-greedy exploration, and sampling.

For Table 6-related experiments, the random seeds used during the original execution were recorded in:

```text
code/temp/seed_summary.csv
```

To ensure reproducibility, the current Table 6-related scripts continue to use these recorded seeds. Therefore, `code/temp/seed_summary.csv` must be kept in the repository when reproducing Table 6.

For direct reproduction of the reported tables, users are advised to use the provided trained model files, recorded seeds, and intermediate outputs. If users retrain the DQN models rather than using the provided files, numerical differences may occur due to stochastic training procedures and hardware- or library-specific implementations.

### 8.10 Notes

- The table-generation, figure-generation, and model-training scripts should be run from the `code/` directory.
- Keep the original repository structure unchanged, especially `code/data/`, `code/model/`, `code/temp/`, and `result/`.
- Ensure that intermediate files and pre-trained results are available.
- Model retraining can be performed if necessary, but it is not required for reproducing the reported table results.

------

## 9. Expected Runtime

The actual runtime depends on hardware configuration, operating system, CPU/GPU availability, disk I/O, and whether users reproduce the results using the provided intermediate outputs or retrain the DQN models. Therefore, a fixed total runtime is not reported.

As a general reference:

- Most ordinary single table-generation scripts can usually be completed within seconds to a few minutes when the provided intermediate results are used.
- Table 6 scripts are more time-consuming because they involve repeated sampling experiments; a single script may take substantially longer, depending on hardware configuration.
- Figure 3 to Figure 7 can be generated after the required summarized result files are available.
- Model retraining requires more time than direct table reproduction and is sensitive to GPU availability.
- Full reproduction of all tables may vary considerably across machines. Users should expect longer runtimes if they choose to retrain models or regenerate intermediate results.

------

## 10. Hardware Requirements

The recommended hardware configuration is as follows:

- CPU: multi-core processor, 4 cores or above recommended
- Memory: at least 16GB, 32GB recommended
- GPU: NVIDIA GPU supporting CUDA 11.7 recommended
- Python: 3.9.16

Notes:

- The project runs more efficiently on a GPU environment.
- Execution on CPU may be slower, especially for model retraining and repeated sampling experiments.

------

## 11. Additional Notes

Please follow the instructions above and keep the repository structure unchanged. If path errors occur, please check whether the current working directory is `code/`.
