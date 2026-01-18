
可复现性代码包：


---

项目概述

本仓库包含完整的数据和代码，用于复现论文《通过排序学习与强化学习提升卖方分析师研报预测选择能力》中的所有实证结果、图表和表格。该论文提出了 LTR-DQN，一种新颖的混合模型，将排序学习（LTR）与深度Q网络（DQN）强化学习相结合，用于识别高价值分析师研报并动态优化投资组合决策，专门针对中国非卖空股票市场环境。

核心特色：
- 数据规模：近145,000份分析师研报（2017年12月–2023年3月），包含36+个工程特征
- 方法体系：LambdaRank、LambdaMART、DQN及7个基准模型（Lasso、SVM、MLP、XGBoost）
- 市场覆盖：中国主板（主板）和创业板（创业板）市场
- ESG整合：负面筛选（NS）和正面投资（PI）策略
- 稳健性测试：训练时长、交易成本、采样率和数据完整性

---

📁 仓库结构与输入输出说明

数据目录 (data/)

```
data/
├── dapan/                          # 市场指数数据（沪深300、创业板指）
│   ├── 0060merge.csv               # 主板市场指数9个特征
│   └── 3068merge.csv               # 创业板指数9个特征
├── 0060merge_open_close_final.csv  # 主板：研报与股票开收盘价合并数据（106,255条记录）
├── 3068merge_open_close_final.csv  # 创业板：研报与股票开收盘价合并数据（38,359条记录）
├── 3068report_broker_merged.xlsx   # 按券商分类的研报股票数据
└── ESG/                            # 来自Wind数据库的ESG评分数据
```

排序学习模块 (LTR/)

```
LTR/
├── allreport_return.py              # 基线：不筛选，买入所有推荐研报
│   - 输入: data/0060merge_open_close_final.csv
│   - 输出: end/oc/all_report0060return17_23.csv (每日收益率+5项指标)
│
├── butong_quanshang.py              # 券商异质性分析
│   - 输入: data/3068report_broker_merged.xlsx
│   - 输出: end/institution/3068不同券商分析.xlsx (不同券商5指标结果)
│
├── main_lambdamart.py               # 主板LambdaMART实验
├── chinext_lambdamart.py            # 创业板LambdaMART实验
│   - 输入: data/{dapan_code}merge_open_close_final.csv
│   - 中间输出: temp/oc/batch123/{dapan_code}temp_test_{m}_train{train_year}_{shouxufei}_{yinhuashui}_{learning_rate}_{max_depth}_{n_estimators}.csv
│   - 最终输出: end/oc/batch123/{dapan_code}return_test_{m}_train{train_year}_{shouxufei}_{yinhuashui}_{learning_rate}_{max_depth}_{n_estimators}.csv
│
├── main_lambdarank.py               # 主板LambdaRank实验
├── chinext_lambdarank.py            # 创业板LambdaRank实验
│
├── huigui.py                        # 基准模型（Lasso, SVM, MLP, XGBoost）
│   - 输入: data/{dapan_code}merge_open_close_final.csv
│   - 输出: temp/oc/batch{test_batch}/{dapan_code}temp_test_{Reg_or_Class}_train{train_year}.csv
│   - 输出: end/oc/batch{test_batch}/{dapan_code}return_test_{train_or_test}_{Reg_or_Class}_train{train_year}.csv
│
├── parameter.py                     # 超参数调优
│   - 输出: temp/oc/batch123/{dapan_code}temp_test_ndcg_train3_*.csv
│
├── esg_xuanze.py                    # ESG策略实现
│   - 输入: temp/oc/ESG/{dapan_code}temp_test_{m}_train3_esg.csv
│   - 再输入: temp/meiri_xuanze.csv (DQN推荐的每日选股数量)
│   - 输出: end/oc/batch123/{dapan_code}return_dqn{esg}PI.csv (ESG策略收益结果)
│
├── esg_merge_temp.py                # ESG数据与排序结果合并
│   - 输入: temp/oc/ESG/{dapan_code}temp_test_pairwise11_train3.csv
│   - 输出: temp/oc/ESG/{dapan_code}temp_test_pairwise11_train3_esg.csv
│
├── run_experiments.py               # 采样稳健性测试
│   - 输入: 全样本数据 + temp/meiri_xuanze.csv
│   - 输出: end/oc/batch{test_batch}/{dapan_code}return_test_{train_or_test}_{m}_train{train_year}_{chouyang_rate}.csv
│       例: 0060return_test_ndcg_train3_0.7.csv (70%采样率结果)
│
└── open_close.py                    # 数据预处理流程
    - 输入: 原始研报数据 + 股票开收盘数据
    - 输出: data/{dapan_code}merge_open_close_final.csv
```

强化学习模块 (DQN/)

```
DQN/
├── dl_dqn2.py                      # DQN类实现（PyTorch）
├── DQN_train.py                    # DQN训练脚本
│   - 输入: data/dapan/{bankuaicode}merge.csv (大盘数据)
│   - 输入: temp/oc/batch{test_batch}/{bankuaicode}temp_train_{LTR}_train{train_year}_0.0003_0.001_0.1_6_1000.csv
│   - 输出: model/batch{test_batch}/{bankuaicode}_{LTR}_{train_year}year_top4_train{train_year}TESToc (训练好的模型)
│
├── DQN_test_0060.py                # 主板DQN测试
├── DQN_test_3068.py                # 创业板DQN测试
    - 输入: data/dapan/{bankuaicode}merge.csv (大盘数据)
    - 输入: {bankuaicode}temp_test_{LTR}_train{train_year}_0.0003_0.001_0.1_6_1000.csv (排序得分)
    - 输出: result/batch{test_batch}/{bankuaicode}_{LTR}_{train_year}year_train{train_year}_top4TESToc_{lr}xinxin.xlsx
        例: 0060_ndcg_3year_train3_top4TESToc_0.002xinxin.xlsx (含每日收益和选股情况)
```

---

计算环境要求

系统要求
- 操作系统：Linux（推荐Ubuntu 20.04+）或macOS
- CPU：8核及以上推荐
- 内存：32GB+（推荐64GB以处理完整数据集）
- GPU：可选但推荐用于DQN训练（NVIDIA GPU，8GB+显存）
- 存储：50GB+可用空间

软件及包版本
- Python：3.8.12
- PyTorch：1.11.0（用于DQN实现）
- XGBoost：1.6.2
- Scikit-learn：1.1.1
- Pandas：1.4.3
- NumPy：1.23.0
- Matplotlib：3.5.2
- OpenPyXL：3.0.10

环境配置

```bash
# 克隆仓库
git clone https://github.com/yourusername/LTR-DQN-Analyst-Reports.git
cd LTR-DQN-Analyst-Reports

# 创建conda环境
conda env create -f environment.yml
conda activate ltr-dqn

# 或通过pip安装
pip install -r requirements.txt
```

---

使用指南

步骤1：准备数据

```bash
# 运行数据预处理流程
python LTR/open_close.py
# 输出: data/{0060,3068}merge_open_close_final.csv

# 生成券商层面数据（可选）
python LTR/quanshang_merge.py
# 输出: data/3068report_broker_merged.xlsx
```

步骤2：超参数调优（可选）

```bash
# 调优LambdaRank/LambdaMART
python LTR/parameter.py

# 结果保存至: temp/oc/batch123/0060temp_test_ndcg_train3_*.csv
```

步骤3：训练LTR模型

```bash
# 主板LambdaMART（3年训练集）
python LTR/main_lambdamart.py
# 输出: temp/oc/batch123/0060temp_test_ndcg_train3_*.csv
# 输出: end/oc/batch123/0060return_test_ndcg_train3_*.csv

# 创业板LambdaMART
python LTR/chinext_lambdamart.py
# 输出: temp/oc/batch123/3068temp_test_ndcg_train3_*.csv
# 输出: end/oc/batch123/3068return_test_ndcg_train3_*.csv
```

步骤4：训练DQN模型

```bash
# 使用LTR排序结果训练DQN
python DQN/DQN_train.py
# 输入: temp/oc/batch123/{0060,3068}temp_train_ndcg_train3_*.csv
# 输出: model/batch123/{0060,3068}_ndcg_3year_top4_train3TESToc.pth
```

步骤5：生成最终测试结果

```bash
# 在测试集上评估DQN
python DQN/DQN_test_0060.py  # 主板
# 输出: result/batch123/0060_ndcg_3year_train3_top4TESToc_0.002xinxin.xlsx

python DQN/DQN_test_3068.py  # 创业板
# 输出: result/batch123/3068_ndcg_3year_train3_top4TESToc_0.002xinxin.xlsx
```

步骤6：生成基线结果

```bash
# 生成全报告买入基线
python LTR/allreport_return.py
# 输出: end/oc/all_report0060return17_23.csv
# 输出: end/oc/all_report3068return17_23.csv
```

步骤7：运行稳健性测试

```bash
# 训练时长稳健性
python LTR/huigui.py --train_years 2
python LTR/huigui.py --train_years 4
# 输出: end/oc/batch{2,4}/*return*.csv

# 交易成本敏感性
python LTR/main_lambdamart.py --transaction_fee 0.0001  # 1个基点
python LTR/main_lambdamart.py --transaction_fee 0.0005  # 5个基点

# 采样稳健性
python LTR/run_experiments.py --sampling_rate 0.7
# 输出: end/oc/batch123/*return_*_0.7.csv

# ESG策略
python LTR/esg_merge_temp.py
python LTR/esg_xuanze.py --strategy NS --threshold 0.25
# 输出: end/oc/batch123/*return_dqnNS25PI.csv
```

---


