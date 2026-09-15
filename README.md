# LTR-DQN Reproduction (CPU)

English | [简体中文](README_CN.md)

This repository reproduces the paper tables T3, T4, T5, T6 and T7, the main-text
figures, and Appendix Figures C1-C5. Every run starts from the tracked source
data and trains fresh rankers and DQN models. No fitted model, result workbook,
daily action file, or historical `meiri_xuanze` selection file is required.

**Reference environment:** the formal reproduction target is a Windows 10/11
x64 CPU runner using `parameters_windows.txt`. The Linux files and shell
wrappers exist only for local or rented-server compatibility when the Windows
environment cannot be used; Linux is expected to be close in aggregate, not a
second paper reference implementation.

This README treats Windows as the authoritative workflow. Linux is mentioned
only where shell commands, package installation, or platform selection differ.

## Repository layout

### Entry points

- `train.py`: long-running training. Fits LambdaRank and LambdaMART, then trains
  LTR-DQN from the fresh LambdaMART ranking. With `--t6`, it also runs the
  20-replication sampling experiment.
- `main.py`: evaluates the trained artifacts, fits the short baseline models,
  regenerates DQN actions, and writes Results tables T3/T4/T5/T7.
- `T6_main.py`: validates the fresh T6 raw CSV and writes the T6 workbook. It
  also contains the sampling and backtest implementation used by `train.py`.
- `Fig_main.py`: recomputes and writes main-text Figures 3-7 and their audit
  CSVs from the current run.
- `Appendix_Fig_main.py`: recomputes and writes Appendix Figures C1-C5 and
  their audit CSVs. Figure C4 consumes the fresh T6 raw CSV.

### Shared implementation

- `experiment_core.py`: shared data loading, LambdaRank/LambdaMART fitting,
  baseline fitting, DQN environment and agent, backtesting, metrics, table
  formatting, manifests, and hashes.
- `runtime_config.py`: selects and validates the Windows reference profile (or
  the optional Linux compatibility profile) and applies deterministic CPU
  settings.
- `parameters_windows.txt`: formal Windows reference seeds and unreported
  implementation parameters used locally and by GitHub Actions.
- `parameters_linux.txt`: optional Linux CPU compatibility parameters; it is
  not part of the formal Windows workflow or GitHub Actions.

### Data and automation

- `data/0060merge_open_close_final.csv` and `data/3068merge_open_close_final.csv`:
  stock features and open/close prices used for training and evaluation.
- `data/0060merge_T4.csv` and `data/3068merge_T4.csv`: index/reference series
  used by the paper tables.
- `data/0060merge.csv` and `data/3068merge.csv`: market data used by baselines,
  figures, and T6 backtests.
- `data/0060report_broker_merged.xlsx` and `data/3068report_broker_merged.xlsx`:
  institution-level brokerage data used by Appendix Figure C2.
- `data/dapan/`: broad-market data used by the baseline and DQN backtest paths.
- `data/ESG/`: supplied raw ESG ranking inputs used by T7 and Appendix C5.
- T7 thresholds are computed from the numeric `ESG` column in the raw
  cross-sectional `data/ESG/ESG.csv` at runtime. The current data produce a
  shared q25/q50 of `5.52`/`6.02` for both markets and both strategies.

### T7 ESG strategies

The paper defines the screening levels as the lowest ESG proportions to remove
from the DQN portfolio; the corresponding score cutoffs are calculated from
the raw cross-sectional ESG observations in `data/ESG/ESG.csv`:

- **NS (Negative Screening)**: select the DQN-recommended stocks by predicted
  ranking, then remove holdings below the shared q25 or q50 cutoff. The
  portfolio may therefore contain fewer stocks than the original DQN
  recommendation.
- **PI (Positive Investing)**: apply the shared q25 or q50 cutoff, then
  select the highest-ranked eligible recommendations by model prediction to
  replace excluded holdings until the DQN-recommended count is reached (or
  until eligible candidates are exhausted).

Both markets use the same score cutoff at each level; NS does not replenish
excluded holdings, whereas PI does.
- `data/reproducibility/`: the two 20-seed T6 configuration ledgers. They store
  only the seeds used by the run, not fitted outputs or selection manifests.
- `.github/workflows/reproduce-core.yml` and `reproduce-t6.yml`: Windows CPU
  reference workflows.

### Environment and housekeeping

- `requirements-lock.txt`: exact pip lock for the CPU reproduction, including
  `torch==2.0.0+cpu`.
- `environment.yml`: Conda environment definition for Python 3.9.13 and the
  CPU package set.
- `environment-linux.yml`, `requirements-linux.txt`, `run_linux.sh`, and
  `run_t6_linux.sh`: Linux-only compatibility files for the command and
  dependency differences described later. There is no Linux Actions workflow.
- `.gitignore`: excludes generated `results/`, `temp/`, `model/`, `runs/`, and
  Python caches from commits.
- `README.md`: this guide.
- `README_CN.md`: Chinese version of this guide.

Generated directories are created only after a run:

```text
temp/       fresh rankings, DQN actions, and manifests
model/      fresh DQN checkpoints
runs/       optional self-contained run artifacts
results/    workbooks, paper CSVs, figure SVGs, and audit CSVs
```

## File-by-file guide

The repository is intentionally organized as a small source-only pipeline. The
scripts are separate because training, table evaluation, T6 sampling, and
figure generation have different inputs and runtimes.

| File or directory | Role | Reads | Writes |
| --- | --- | --- | --- |
| `train.py` | Main long-running entry point. Fits LambdaRank/LambdaMART and trains DQN from the fresh LambdaMART ranking. `--t6` additionally creates the T6 raw sampling ledger. | `data/`, active parameter profile | `temp/`, `model/`, `temp/train_manifest.json`, and optionally `temp/t6_runs/` |
| `main.py` | Short evaluation/export entry point. Fits the seven baseline models, evaluates the three primary rankers, regenerates DQN actions, and assembles T3/T4/T5/T7. | `temp/` or a named run directory, `model/`, `data/` | `results/combined/` and evaluation rankings |
| `experiment_core.py` | Single source of truth for market mapping, dates, features, model fitting, DQN environment/agent, backtests, metrics, Excel/CSV formatting, manifests, and hashes. | Raw data and runtime configuration | Shared artifacts called by the entry points |
| `runtime_config.py` | Loads and validates the Windows reference profile first; selects the Linux profile only for compatibility runs, then sets seeds/thread limits and the CPU device. | `parameters_windows.txt` or `parameters_linux.txt` | No result files; metadata is included in manifests |
| `T6_main.py` | Validates the fresh T6 per-seed CSV and writes T6 summary tables/workbook. Its lower section contains the ranker sampling and backtest implementation used by `train.py --t6`. | `temp/t6_runs/t6_raw.csv`, T6 seed ledgers, `data/` | `results/T6/` |
| `Fig_main.py` | Recomputes main-text Figures 3-7. It generates one combined SVG and separate market/panel SVGs plus auditable CSVs. | Fresh rankings/models, raw data, active parameters | `results/figures/` |
| `Appendix_Fig_main.py` | Recomputes Appendix Figures C1-C5. C2 uses institution-level XLSX files when available; C4 uses the current T6 raw CSV; C5 uses raw ESG observations. | Raw data and current-run artifacts | `results/appendix_figures/` |
| `parameters_windows.txt` | JSON-formatted Windows reference profile: stage seeds plus unreported LambdaRank/LambdaMART implementation settings. Used by local Windows and Windows Actions. | None | None |
| `parameters_linux.txt` | Optional Linux CPU compatibility profile, used only when the same source tree is run outside the Windows reference environment. | None | None |
| `requirements-lock.txt` | Exact pip package set for the formal Windows CPU reference runtime. | None | None |
| `environment.yml` | Conda definition for the formal Windows reference environment. | None | None |
| `requirements-linux.txt` / `environment-linux.yml` | Linux-only dependency definitions for the compatibility note below. | None | None |
| `run_linux.sh` / `run_t6_linux.sh` | Linux-only wrappers for the compatibility path; Windows users should follow the PowerShell commands below. | Scripts and source data | Same `temp/`, `model/`, `results/` layout |
| `tests/test_linux_compat.py` | Fast structural/configuration tests. It does not perform model training. | Source files and raw ESG data | None |
| `.github/workflows/reproduce-core.yml` | Manual Windows CPU Action for T3/T4/T5/T7, main figures, and C1/C2/C3/C5. | Clean checkout | Artifact containing tables, figures, and manifest |
| `.github/workflows/reproduce-t6.yml` | Manual Windows CPU Action for T6 and C4. | Clean checkout | Artifact containing T6/C4 outputs and manifest |
| `GITHUB_ACTIONS_GUIDE_CN.md` | Chinese click-by-click guide for enabling and running the two Actions and downloading artifacts. | N/A | N/A |
| `LINUX_REPRODUCIBILITY.md` | Notes on the local Linux compatibility path, locked/relaxed runtime, and expected numerical differences. | N/A | N/A |
| `data/` | Versioned source inputs. The files are inputs, not cached model outputs. | N/A | N/A |
| `README.md` / `README_CN.md` | English/Chinese usage and parameter documentation. | N/A | N/A |

### Data file meanings

`0060` denotes the Main Board and `3068` denotes ChiNext. The two
`*_merge_open_close_final.csv` files contain stock-level features, labels and
open/close prices used to fit rankers and evaluate portfolios. The smaller
`*_merge.csv` files are the market/backtest inputs used by baselines, figures,
and DQN. `*_merge_T4.csv` contains the reference/index series used when
building the paper-format T4 sheets. `data/dapan/` contains the broad-market
series used as DQN state input. `data/ESG/ESG.csv` is the one-observation-per-
stock ESG cross-section used to compute q25/q50; the two dated ESG files carry
the ranking-panel predictions and prices used by T7. The two brokerage XLSX
files are only needed for institution-level C2. The files under
`data/reproducibility/` are disclosed T6 seed ledgers, not fitted results.

### Stock table columns

`FEATURES` in `experiment_core.py` contains 25 model inputs grouped as report
text (`page`, `advance_reaction`, `star_analyst`, `title_len`, `num_sentence`,
`avg_sentence_len`, `sd_sentence_len`, `num_authors`, `analyst_coverage`),
Fama-French-style factors (`rm_rf`, `smb`, `hml`, `rmw`, `cma`), brokerage and
listing attributes (`broker_size`, `listed`, `broker_status`), prior-return
statistics (`prior_performance_avg`, `prior_performance_sd`), and six industry
indicators (`ind_1`-`ind_6`). `real_return` is the ranking/regression target;
`up_down` is the binary classification target. `qid_date` groups stocks into
one ranking day, while `stock_code`, `close`, and `pclose` identify the stock
and its sell/buy prices.

### Core function map

| Function | Purpose |
| --- | --- |
| `load_stock_data` | Selects the year-specific training window and common test window from the raw stock file. |
| `fit_ranker` | Fits one LambdaRank or LambdaMART model and returns train/test prediction frames. |
| `fit_baseline` / `model_for_baseline` | Fits one of the seven short baseline estimators and returns test predictions. |
| `train_dqn` | Builds the DQN environment, trains the agent, and saves a checkpoint with metadata. |
| `evaluate_dqn` | Loads a trained checkpoint and generates policy actions for the test period. |
| `backtest_predictions` | Converts daily predictions into top-four trades and ARR/MDR/CR/SR/WR metrics. |
| `esg_thresholds_common` / `esg_metrics` | Computes shared raw-data ESG cutoffs and evaluates NS/PI portfolios. |
| `write_results` | Writes the long-form records, paper-format CSVs, and multi-sheet workbook. |

`runtime_config.py` provides the corresponding configuration functions:
`load_platform_parameters` validates a profile, `stage_seed` resolves one
stage seed, `load_rank_config`/`load_mart_config` merge implementation-only
overrides, and `set_global_determinism` applies Python/NumPy/PyTorch seeds.

## Parameter reference

There are three parameter layers. Values explicitly reported by the paper are
locked in Python code. The Windows profile is the formal reference profile;
the Linux profile is an emergency compatibility profile for a different OS and
dependency build. Both profiles hold only seeds and implementation choices
that were not fully specified in the manuscript. Command-line options are
runtime controls and are recorded in the manifest.

### Paper-locked model parameters

| Component | Main Board (`0060`) | ChiNext (`3068`) | Notes |
| --- | --- | --- | --- |
| LambdaRank objective | `rank:pairwise` | `rank:pairwise` | Independent ranking baseline |
| LambdaRank `learning_rate` | `0.01` | `0.1` | Locked; not a profile override |
| LambdaRank unreported defaults | `max_depth=6`, `n_estimators=100`, `subsample=1.0`, `colsample_bytree=1.0` | Same unless the active profile overrides an unreported field | `eval_metric=ndcg`, `n_jobs=1` |
| LambdaMART objective | `rank:map` | `rank:ndcg` | The objective differs by market |
| LambdaMART `learning_rate` | `0.001` | `0.1` | Locked |
| LambdaMART `max_depth` | `5` | `6` | Locked |
| LambdaMART `n_estimators` | `1000` | `1000` | Locked |
| LTR-DQN `learning_rate` | `0.002` | `0.002` | Locked; implemented with Adam |

The `--ranker_tree_method` option selects XGBoost's CPU tree builder for the
figure sensitivity fits (`approx` by default; `hist` and `exact` are
accepted). For the main `train.py` fit, the actual tree builder is the
`tree_method` in the active profile, while the option value is retained in the
manifest; the formal Windows profile uses `approx` in every row. The option
does not change paper learning rates or estimator counts. `main.py` consumes
the fresh LambdaMART ranking as the DQN input (`DQN_RANKER = "LambdaMART"`).

### DQN training defaults

These values are the defaults in `train.py` and are written into the DQN
checkpoint metadata. `--lr` is checked against the paper-locked `0.002`.

| Option | Default | Meaning |
| --- | ---: | --- |
| `--n_games` | `31` | Training episodes |
| `--lr` | `0.002` | Adam learning rate; paper-locked |
| `--gamma` | `0.9` | Discount factor for future rewards |
| `--epsilon` | `1.0` | Initial epsilon-greedy exploration probability |
| `--eps_end` | `0.03` | Minimum exploration probability |
| `--eps_dec` | `0.00015` | Exploration decrement after each learning step |
| `--batch_size` | `32` | Replay samples per update |
| `--max_mem_size` | `100` | Replay-memory capacity |
| `--replace_target_iter` | `8` | Learning steps between target-network copies |
| network shape | `13 -> 256 -> 128 -> 5` | State width, hidden layers, and five actions (`0`-`4` stocks) |
| optimizer/loss | Adam / MSE | Adam betas `(0.9, 0.999)`, `eps=1e-8` |

The DQN environment starts with `500,000,000` units of capital, applies a
`0.0003` buy fee and `0.0013` sell-side fee/tax, and ranks selected stocks by
the current LambdaMART prediction. Backtest portfolio evaluation uses the
same fee rates, with an initial capital of `5,000,000` for ranker/baseline
metrics and `1,000,000` for all-stock/index comparisons where applicable.

### Baseline model parameters

The baseline names in T3-T5 are the following scikit-learn/XGBoost estimators.
They are fitted by `main.py`, not by `train.py`.

| Name | Estimator and fixed settings |
| --- | --- |
| `LR` | Lasso regression, `alpha=0.0001` |
| `MLP_R` / `MLP_C` | One hidden layer `(24,)`, `max_iter=100`, `random_state=42`; regression/classification variant respectively |
| `SVM_R` / `SVM_C` | RBF-kernel SVR/SVC, `C=1.0` |
| `XGB_R` | XGBoost regressor, `objective=reg:squarederror`, `max_depth=4`, `learning_rate=0.1`, `subsample=1.0`, `colsample_bytree=0.8`, `tree_method=approx`, `n_jobs=1`; estimator count and `max_bin` follow the year/market map in `BASELINE_MAX_BIN` |
| `XGB_C` | XGBoost classifier with the same tree settings; it uses `200` estimators for years 2/4 and `150` for year 3, plus the corresponding `max_bin` map |

`_R` means regression on `real_return`; `_C` means classification on the
binary `up_down` label. Before fitting, `fit_baseline` scales the features and
keeps the paper-compatible train/test normalization protocol. The baseline
seed is resolved from the active profile even when an estimator has its own
fixed implementation seed.

### Platform profile schema

Each `.txt` profile is JSON with the keys `schema_version`, `profile`,
`system`, `purpose`, `stage_seeds`, `rank_config`, and `mart_config`.

- `stage_seeds[market_code][train_year][stage]` resolves one seed for
  `rank`, `mart`, `dqn`, `baseline`, and `evaluation`. A command-line
  `--seed` overrides every stage; `--seed_config` merges a JSON override over
  the active profile.
- `rank_config` contains implementation-only LambdaRank fields:
  `max_depth`, `n_estimators`, `subsample`, `colsample_bytree`, and
  `tree_method`.
- `mart_config` contains implementation-only LambdaMART fields:
  `max_bin`, `min_child_weight`, `subsample`, `colsample_bytree`, and
  `tree_method`.
- Sampling values are fractions in `(0, 1]`; `max_depth`, `n_estimators`, and
  `max_bin` are positive integers. Unknown fields and paper-locked learning
  rates are rejected instead of silently changing the experiment.

Parameter precedence is deliberately narrow: paper-locked values in
`experiment_core.py` cannot be overridden; otherwise the selected platform
profile supplies defaults, an optional `--seed_config`/`--rank_config`/
`--mart_config` supplies validated JSON overrides, and `--seed` can replace all
stage seeds for a diagnostic run. Keep all overrides unset for the reported
Windows reproduction.

The active file is selected from `platform.system()` and recorded with its
profile name and SHA-256 in every manifest. Windows runs should always report
`windows-reference`.

For the formal Windows profile, every row uses `subsample=1.0`,
`colsample_bytree=1.0`, and `tree_method=approx`; the market/year-specific
values are:

| Market/year | Stage seeds (`rank/mart/dqn/baseline/evaluation`) | LambdaRank (`max_depth`, `n_estimators`) | LambdaMART (`max_bin`, `min_child_weight`) |
| --- | --- | --- | --- |
| Main 2-year | `40/41/40/43/19` | `5`, `100` | `32`, `1` |
| Main 3-year | `50/51/10/53/36` | `6`, `100` | `256`, `1` |
| Main 4-year | `60/61/40/63/59` | `6`, `150` | `32`, `1` |
| ChiNext 2-year | `50/51/50/53/67` | `8`, `25` | `3`, `0.43` |
| ChiNext 3-year | `60/61/50/63/31` | `6`, `100` | `256`, `1` |
| ChiNext 4-year | `70/71/50/73/49` | `3`, `50` | `9`, `1` |

These are Windows implementation/profile values, not replacements for the
paper's learning rates, LambdaMART depth, or estimator counts listed above.
Linux values are maintained separately only for the compatibility fallback.

### Training CLI

| Option | Default | Meaning |
| --- | --- | --- |
| `--run_dir` | repository root | Root for `temp/`, `model/`, and manifests |
| `--models` | `all` | `rankers`, `dqn`, or both (`all`) |
| `--markets` | both | `Main`, `ChiNext`, or comma-separated values |
| `--years` | `2,3,4` | Training-window lengths; only `2`, `3`, `4` are valid |
| `--seed` | profile values | One global seed override |
| `--seed_config` | active profile | JSON seed overrides |
| `--rank_config` | active profile | JSON overrides for unreported LambdaRank fields |
| `--mart_config` | active profile | JSON overrides for unreported LambdaMART fields |
| `--ranker_tree_method` | `approx` | XGBoost tree builder |
| `--t6` | off | Add the T6 sampling run |
| `--t6_markets` | `all` | T6 markets: `Main`, `ChiNext`, or both |
| `--t6_max_seeds` | `20` | Fresh replications per T6 rate/model/market cell |
| `--t6_seed_summary` | `data/reproducibility/t6_cpu20_seed_summary.csv` | Ranker/MART T6 seed ledger |
| `--t6_dqn_seed_summary` | `data/reproducibility/t6_cpu20_dqn_seed_summary.csv` | DQN T6 seed ledger |

`--lr`, `--n_games`, `--gamma`, `--epsilon`, `--eps_end`, `--eps_dec`,
`--batch_size`, `--max_mem_size`, and `--replace_target_iter` are the DQN
controls shown above. For the formal reproduction, leave them at their
defaults; in particular, changing `--lr` away from `0.002` is rejected.

### Evaluation and figure CLI

`main.py` accepts `--tables T3,T4,T5,T7`, `--markets all|Main,ChiNext`,
`--seed`, `--seed_config`, `--output_dir`, `--export_csvs`, and
`--no_baselines`. `--export_csvs` adds `results_long.csv` and one paper-format
CSV per table; `--no_baselines` omits the short baseline fit and is useful only
for diagnostics. DQN evaluation is intentionally fixed to the trained policy;
the `--dqn_eval_mode dqn` option is not a switch to a cached action file.

`Fig_main.py` accepts `--figures 3,4,5,6,7`, `--run_dir`, `--output_dir`,
`--seed`, `--seed_config`, `--ranker_tree_method`, `--n_games`, and `--force`.
`--force` invalidates figure CSV caches after a source or parameter change.
`Appendix_Fig_main.py` accepts `--figures C1,C2,C3,C4,C5`, `--run_dir`,
`--output_dir`, `--t6_csv`, `--broker_file`, `--broker_column`,
`--min_broker_reports`, and `--force`.

### Fixed dates and metric constants

| Constant | Value | Use |
| --- | --- | --- |
| `TRAIN_END` | `20211206` | End of every training window |
| `TEST_START` / `TEST_END` | `20211207` / `20230303` | Common test period |
| training starts | year 2: `20191206`; year 3: `20181206`; year 4: `20171206` | Window length selected by `--years` |
| annualization | `242` trading days/year | ARR and Sharpe calculations |
| commission | `0.0003` | Buy and sell commission |
| stamp tax | `0.001` | Sell-side tax |
| top-portfolio size | `4` | Non-classifier ranking backtest |

### T6 and T7 controls

T6 uses exactly `20` fresh replications at sampling rates `50%`, `60%`,
`70%`, `80%`, and `90%`, for both markets and three models. The 100% reference
row comes from the same run's T4 evaluation. The two seed ledgers are kept in
`data/reproducibility/` so the sampling randomness is disclosed without
committing fitted outputs.

T7 computes one shared ESG cutoff per level from the numeric `ESG` column in
the raw cross-sectional `data/ESG/ESG.csv`: q25=`5.52`, q50=`6.02` for the
current data. **NS** first selects the DQN top-four and removes holdings below
the cutoff; **PI** filters eligible stocks first and replenishes excluded
holdings by prediction rank until the top-four count is restored. The two
markets use the same cutoff at each level.

### Figure-only sensitivity grids

The sensitivity panels intentionally use a disclosed grid rather than silently
reusing the main-run model. Figures 3 and 4 evaluate learning rates
`0.0001, 0.001, 0.002, 0.01, 0.1, 0.2`; Figure 4 additionally evaluates
`n_estimators=800, 900, 1000, 1100, 1200` and
`max_depth=4, 5, 6, 7, 8`. Figure 3(b) uses fixed evaluation seeds Main=`36`
and ChiNext=`66` across its six DQN learning-rate cells. These settings affect
only the sensitivity figures and are recorded in their figure manifest; they
do not replace the paper-locked T4/T5 training configuration.

## Requirements

- Python 3.9.13, 64-bit x86 (`x64`).
- Windows 10/11 x64 CPU is the formal reference environment for the paper
  results. Linux x86-64 is supported only as a compatibility path; Linux
  output is intended to be close overall, not bit-for-bit identical.
- CPU only. GPU is not required and is not selected by the default code path.
- One compute thread is enforced for BLAS, XGBoost, and PyTorch to reduce
  cross-machine variation. More CPU cores may improve operating-system
  scheduling overhead only; they do not change the configured algorithm.
- At least 8 GB RAM and 10 GB free disk space are recommended for the full
  all-years run, because the training and figure steps create temporary files.

The Windows reference lock uses pip 24.1.2, NumPy 1.21.5, pandas 1.4.4,
scikit-learn 1.2.0, PyTorch 2.0.0+cpu, and XGBoost 1.7.6. The Linux
compatibility file requests the same versions where wheels are available; if a
Linux device cannot install them, `LTR_DQN_RELAXED_RUNTIME=1` must be set
explicitly and numerical differences should be expected.

## Windows parameter selection

No parameter path is required on the Windows command line. At startup,
`runtime_config.py` checks `platform.system()` and, on Windows, loads the
tracked JSON-formatted `parameters_windows.txt` profile:

`windows-reference` is the profile used by local Windows and GitHub Actions.
When the same source tree is run on Linux, the code instead selects
`parameters_linux.txt` (`linux-emergency-compatibility`) so that the shell
commands and dependency differences are explicit; this is only a compatibility
fallback. Any other operating system stops with an error.

The files contain only seeds and implementation settings not fixed by the
paper. Paper-reported learning rates and other reported hyperparameters remain
defined and validated in code, so the platform profiles cannot override them.
Every manifest records the profile name, file name, and SHA-256 hash.

## Installation (Windows reference)

### Formal reference: Windows PowerShell

```powershell
git clone https://github.com/whyhw-code/LTR-DQN.git
Set-Location LTR-DQN
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade "pip==24.1.2"
python -m pip install -r requirements-lock.txt
$env:PYTHONHASHSEED = "0"
```

### Linux command differences (compatibility only)

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

Linux is not the reference workflow. If reproduction must be moved to Linux,
the equivalent Conda command is `conda env create -f environment-linux.yml`
followed by `conda activate ltr-dqn-linux`; use `bash run_linux.sh` or
`bash run_t6_linux.sh`. The scripts require the locked versions by default;
use `export LTR_DQN_RELAXED_RUNTIME=1` only when the locked wheels are
unavailable.

The formal Windows Conda alternative is:

```powershell
conda env create -f environment.yml
conda activate ltr-dqn
```

Verify the installation before a long run:

```powershell
python -c "import sys,torch,xgboost,numpy,pandas,sklearn; print(sys.version); print(torch.__version__, xgboost.__version__, numpy.__version__, pandas.__version__, sklearn.__version__, torch.get_num_threads()); print(torch.cuda.is_available())"
python -m compileall -q *.py
```

The last line should print `False` for CUDA availability and the thread count
should be `1`.

## Complete local reproduction (Windows PowerShell)

Run these commands from the repository root, in order.

### 1. Train all primary models

```powershell
python train.py --models all --years 2,3,4 --ranker_tree_method approx
```

This refits LambdaRank and LambdaMART from `data/`, then trains DQN from the
new LambdaMART rankings. It does not read a precomputed ranking or a fixed
daily selection file.

### 2. Export Results tables

```powershell
python main.py --export_csvs
```

Main output: `results/combined/results.xlsx`, containing T3, T4, T5 and T7.
Paper-format CSVs and JSON manifests are written beside the workbook.

### 3. Generate main-text figures

```powershell
python Fig_main.py --ranker_tree_method approx --force
```

Output: `results/figures/`.

### 4. Generate Appendix Figures C1, C2, C3 and C5

```powershell
python Appendix_Fig_main.py --figures C1,C2,C3,C5 --force
```

Output: `results/appendix_figures/`.

When the two institution-level brokerage workbooks are present in `data/`, C2
automatically uses the `institution` column and the original daily-mean return
aggregation. A custom report file or directory can still be supplied with
`--broker_file`; use `--min_broker_reports` to change the minimum group size.

### 5. Run T6 separately

T6 is a separate 20-seed sampling experiment. Run:

```powershell
python train.py --models all --years 3 --t6 --ranker_tree_method approx
python T6_main.py
python Appendix_Fig_main.py --figures C4 --force
```

Outputs are `results/T6/T6.xlsx` and Appendix Figure C4. The 100% T6 column
comes from the same run's freshly evaluated T4 models; no result is copied from
an external intermediate file.

## GitHub Actions reproduction

1. Open the repository's **Actions** tab.
2. Select **1 - One-click Results and Figures (Windows CPU)** and choose **Run workflow** for
   T3/T4/T5/T7, all main figures, and Appendix C1/C2/C3/C5.
3. Select **2 - One-click T6 and Figure C4 (Windows CPU)** and choose **Run workflow** for T6 and
   Appendix C4.
4. Open the completed run and download its artifact. The artifact contains the
   generated workbook, CSVs, figures, manifests, and the selected
   `parameters_windows.txt`.

The repository has only the `main` branch. Both online workflows require the
standard `windows-2022` x64 CPU runner and assert that
`parameters_windows.txt` was selected before training. Linux compatibility is
available only through the local shell scripts, not through GitHub Actions.

## Reproducibility notes

- Platform-specific seeds and unreported implementation settings are in
  `parameters_windows.txt` and `parameters_linux.txt`; the T6 seed ledgers are
  under `data/reproducibility/`.
- `LambdaRank` and `LambdaMART` are retrained from the raw tracked data on each
  run. DQN consumes the LambdaMART output created by that same run.
- `PYTHONHASHSEED=0`, fixed seeds, deterministic PyTorch settings, stable CSV
  ordering, and `n_jobs=1` are enabled to reduce platform variation.
- Manifests record runtime versions, input hashes, action hashes, and checkpoint
  hashes. Linux manifests identify `runtime_mode=linux-compat`. Even with
  identical dependencies, compiler, CPU-instruction-set, and math-library
  differences between Linux and Windows can cause small numerical changes, so
  bit-for-bit identity is not promised.
- Do not change `--seed`, `--seed_config`, `--lr`, `--n_games`, or
  `--ranker_tree_method` when reproducing the reported default run.
