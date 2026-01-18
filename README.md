LTR-DQN Reproducible Code Package

---

Project Overview

This repository contains complete data and code to reproduce all experimental results, charts, and tables from the paper "Enhancing Portfolio Selection Ability through Reinforcement Learning in Learning to Rank for Analyst Report Prediction." The paper proposes LTR-DQN, a novel hybrid model that combines Learning to Rank (LTR) with Deep Q-Network (DQN) reinforcement learning to identify high-value analyst reports and dynamically optimize investment portfolio decisions, specifically tailored for the Chinese A-share market environment.

Core Features

- Data Scale: 145,000 analyst reports (Dec 2017 - Mar 2023) with 27 engineered features
- Methodology: LambdaRank, LambdaMART, DQN, and 7 baseline models (Lasso, SVM, MLP, XGBoost)
- Market Coverage: Main Board (主板) and ChiNext (创业板) markets
- ESG Integration: Negative screening (NS) and Positive Impact (PI) strategies
- Robustness Tests: Training duration, transaction costs, sampling bias, and data completeness

---

📁 Repository Structure & I/O Specification

Data Directory (data/)

```python
data/
├── dapan/           # Market index data (CSI 300, ChiNext Index)
│   ├── 0060merge.csv    # Main board: 9 index features
│   └── 3068merge.csv    # ChiNext: 9 index features
├── 0060merge_open_close_final.csv  # Main board: merged report & stock OHLC data (106,255 records)
├── 3068merge_open_close_final.csv  # ChiNext: merged report & stock OHLC data (38,359 records)
├── 3068report_broker_merged.xlsx   # Reports categorized by brokerage
└── ESG.csv             # ESG ratings from Wind database
```

Learning to Rank Module (LTR/)

```python
LTR/
├── allreport_return.py      # Baseline: No screening, buy all target reports
│   - Input: data/0060merge_open_close_final.csv
│   - Output: end/oc/all_report0060return17_23.csv (daily returns + 5 metrics)

├── butong_quanshang.py      # Brokerage heterogeneity analysis
│   - Input: data/3068report_broker_merged.xlsx
│   - Output: end/institution/3068broker_diff_analysis.xlsx (5 brokerage metrics)

├── main_lambdamart.py       # Main board LambdaMART experiment
└── chinext_lambdamart.py    # ChiNext LambdaMART experiment
│     - Input: data/{dapan_code}merge_open_close_final.csv
│     - Intermediate output (for RL training/testing): temp/oc/batch123/{dapan_code}temp_test_{m}_train{train_year}_{shouxufei}_{yinhuashui}_{learning_rate}_{max_depth}_{n_estimators}.csv
│     - Final output: end/oc/batch123/{dapan_code}return_test_{m}_train{train_year}_{shouxufei}_{yinhuashui}_{learning_rate}_{max_depth}_{n_estimators}.csv

├── main_lambdarank.py       # Main board LambdaRank experiment
└── chinext_lambdarank.py    # ChiNext LambdaRank experiment

├── huigui.py                # Baseline models (Lasso, SVM, MLP, XGBoost)
│   - Input: data/{dapan_code}merge_open_close_final.csv
│   - temp output: temp/oc/batch{test_batch}/{dapan_code}temp_test_{Reg_or_Class}_train{train_year}.csv
│   - Final output: end/oc/batch{test_batch}/{dapan_code}return_test_{train_or_test}_{Reg_or_Class}_train{train_year}.csv

├── parameter.py             # Hyperparameter optimization
│   - Input: data/{dapan_code}merge_open_close_final.csv

├── esg_xuanze.py            # ESG strategy implementation
│   - Input: temp/oc/ESG/{dapan_code}temp_test_{m}_train3_esg.csv
│   - Secondary input: temp/meiri_xuanze.csv (DQN daily stock selection count)
│   - Output: end/oc/batch123/{dapan_code}return_dqn{esg}PI.csv (ESG strategy results)

├── esg_merge_temp.py        # Merging ESG data with LTR results
│   - Input: temp/oc/ESG/{dapan_code}temp_test_pairwise11_train3.csv
│   - Output: temp/oc/ESG/{dapan_code}temp_test_pairwise11_train3_esg.csv

└── run_experiments.py       # Sampling stability tests
    - Input: data/{dapan_code}merge_open_close_final.csv
    - Secondary input: temp/meiri_xuanze.csv
    - Output: end/oc/batch{test_batch}/{dapan_code}return_test_{train_or_test}_{m}_train{train_year}_{chouyang_rate}.csv
```

Deep Reinforcement Learning Module (DQN/)

```python
DQN/
├── dl_dqn2.py               # DQN class implementation (PyTorch)
├── DQN_train.py             # DQN training script
│   - Input: data/dapan/{bankuaicode}merge.csv (market index data)
│   - Input: temp/oc/batch{test_batch}/{bankuaicode}temp_train_{LTR}_train{train_year}_0.0003_0.001_0.1_6_1000.csv (LTR intermediate output)
│   - Output: model/batch{test_batch}/{bankuaicode}_{LTR}_{train_year}year_top4_train{train_year}TESToc (trained model)

├── DQN_test_0060.py         # Main board DQN testing
└── DQN_test_3068.py         # ChiNext DQN testing
    - Input: data/dapan/{bankuaicode}merge.csv
    - Input: {bankuaicode}temp_test_{LTR}_train{train_year}_0.0003_0.001_0.1_6_1000.csv (LTR scores)
    - Output: result/batch{test_batch}/{bankuaicode}_{LTR}_{train_year}year_train{train_year}_top4TESToc_{lr}xinxin.xlsx
```

---

Computational Environment Requirements

System Specifications

- OS: Windows 10 (64-bit)
- CPU: AMD Ryzen 7 6800H with Radeon Graphics (3.20 GHz)
- RAM: 16GB
- GPU: NVIDIA RTX 3060 8GB
- Storage: 50GB+ available space

Software & Package Versions

- Python: 3.9.10
- PyTorch: 2.0.0 (for DQN implementation)
- XGBoost: 2.0.0
- Scikit-learn: 1.1.1
- Pandas: 1.4.3
- NumPy: 1.23.0
- Matplotlib: 3.7.2
- OpenPyXL: 3.1.2

Environment Setup

```bash
# Clone repository
git clone https://github.com/whyhw-code/LTR-DQN/

# Create conda environment
conda env create -f environment.yml
conda activate ltr-dqn

# Or install via pip
pip install -r requirements.txt
```

---

Usage Guide

Step 1: Data Preparation

```bash
# Generate brokerage-level data (optional)
python LTR/quanshang_merge.py
# Output: data/3068report_broker_merged.xlsx
```

Step 2: Hyperparameter Optimization (Optional)

```bash
# Parameter tuning for comparison models
python LTR/parameter.py
```

Step 3: Train LTR Models

```bash
# Main board LambdaMART (3-year training set)
python LTR/main_lambdamart.py
# Output: temp/oc/batch123/0060temp_test_ndcg_train3_0.0003_0.001_0.001_5_1000.csv
# Output: end/oc/batch123/0060return_test_ndcg_train3_0.0003_0.001_0.001_5_1000.csv

# ChiNext LambdaMART
python LTR/chinext_lambdamart.py
# Output: temp/oc/batch123/3068temp_test_ndcg_train3_0.0003_0.001_0.1_6_1000.csv
# Output: end/oc/batch123/3068return_test_ndcg_train3_0.0003_0.001_0.1_6_1000.csv
```

Step 4: Train DQN Models

```bash
# Train DQN using LTR outputs
python DQN/DQN_train.py
# Input: temp/oc/batch123/3068temp_train_ndcg_train3_0.0003_0.001_0.1_6_1000.csv
# Output: model/batch123/{0060,3068}_ndcg_3year_top4_train3TESToc.pth
```

Step 5: Generate Final Test Results

```bash
# Evaluate DQN on test set
python DQN/DQN_test_0060.py  # Main board
# Output: result/batch123/0060_ndcg_3year_train3_top4TESToc_0.002xinxin.xlsx

python DQN/DQN_test_3068.py  # ChiNext 
# Output: result/batch123/3068_ndcg_3year_train3_top4TESToc_0.002xinxin.xlsx
```

Step 6: Generate Baseline Results

```bash
# Generate buy-all-reports baseline
python LTR/allreport_return.py
# Output: end/oc/all_report0060return17_23.csv
# Output: end/oc/all_report3068return17_23.csv
```

Step 7: Run Robustness Tests

```bash
# Sampling stability
python LTR/run_experiments.py
# Output: end/oc/batch123/0060return_test_ndcg_train3_0.7.csv

# ESG strategy
python LTR/esg_merge_temp.py
python LTR/esg_xuanze.py
# Output: end/oc/batch123/3068return_dqn6.02PI.csv
```

---
