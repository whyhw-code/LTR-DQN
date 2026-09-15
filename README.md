# LTR-DQN Reproduction (CPU)

English | [简体中文](README_CN.md)

This repository reproduces the paper tables T3, T4, T5, T6 and T7, the main-text
figures, and Appendix Figures C1-C5. Every run starts from the tracked source
data and trains fresh rankers and DQN models. No fitted model, result workbook,
daily action file, or historical `meiri_xuanze` selection file is required.

**Reference environment:** Windows 10/11 x64 CPU using
`parameters_windows.txt`. Local reproduction and GitHub Actions both follow
this Windows reference workflow.

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
- `runtime_config.py`: selects and validates the Windows reference profile and
  applies deterministic CPU settings.
- `parameters_windows.txt`: formal Windows reference seeds and unreported
  implementation parameters used locally and by GitHub Actions.

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
- `data/reproducibility/`: the two 20-seed T6 configuration ledgers. They store
  only the seeds used by the run, not fitted outputs or selection manifests.
- `.github/workflows/reproduce-core.yml` and `reproduce-t6.yml`: Windows CPU
  reference workflows.

### Environment and housekeeping

- `requirements-lock.txt`: exact pip lock for the CPU reproduction, including
  `torch==2.0.0+cpu`.
- `environment.yml`: Conda environment definition for Python 3.9.13 and the
  CPU package set.
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
| `runtime_config.py` | Loads and validates the Windows reference profile, then sets seeds/thread limits and the CPU device. | `parameters_windows.txt` | No result files; metadata is included in manifests |
| `T6_main.py` | Validates the fresh T6 per-seed CSV and writes T6 summary tables/workbook. Its lower section contains the ranker sampling and backtest implementation used by `train.py --t6`. | `temp/t6_runs/t6_raw.csv`, T6 seed ledgers, `data/` | `results/T6/` |
| `Fig_main.py` | Recomputes main-text Figures 3-7. It generates one combined SVG and separate market/panel SVGs plus auditable CSVs. | Fresh rankings/models, raw data, active parameters | `results/figures/` |
| `Appendix_Fig_main.py` | Recomputes Appendix Figures C1-C5. C2 uses institution-level XLSX files when available; C4 uses the current T6 raw CSV; C5 uses raw ESG observations. | Raw data and current-run artifacts | `results/appendix_figures/` |
| `parameters_windows.txt` | JSON-formatted Windows reference profile: stage seeds plus unreported LambdaRank/LambdaMART implementation settings. Used by local Windows and Windows Actions. | None | None |
| `requirements-lock.txt` | Exact pip package set for the formal Windows CPU reference runtime. | None | None |
| `environment.yml` | Conda definition for the formal Windows reference environment. | None | None |
| `.github/workflows/reproduce-core.yml` | Manual Windows CPU Action for T3/T4/T5/T7, main figures, and C1/C2/C3/C5. | Clean checkout | Artifact containing tables, figures, and manifest |
| `.github/workflows/reproduce-t6.yml` | Manual Windows CPU Action for T6 and C4. | Clean checkout | Artifact containing T6/C4 outputs and manifest |
| `GITHUB_ACTIONS_GUIDE_CN.md` | Chinese click-by-click guide for enabling and running the two Actions and downloading artifacts. | N/A | N/A |
| `data/` | Versioned source inputs. The files are inputs, not cached model outputs. | N/A | N/A |
| `README.md` / `README_CN.md` | English/Chinese repository and reproduction guides. | N/A | N/A |

### Data file meanings

`0060` denotes the Main Board and `3068` denotes ChiNext. The two
`*_merge_open_close_final.csv` files contain stock-level features, labels and
open/close prices used to fit rankers and evaluate portfolios. The smaller
`*_merge.csv` files are the market/backtest inputs used by baselines, figures,
and DQN. `*_merge_T4.csv` contains the reference/index series used when
building the paper-format T4 sheets. `data/dapan/` contains the broad-market
series used as DQN state input. `data/ESG/ESG.csv` is the stock-level ESG
cross-section; the two dated ESG files carry the ranking-panel predictions and
prices used by T7. The two brokerage XLSX
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
| `write_results` | Writes the long-form records, paper-format CSVs, and multi-sheet workbook. |

`runtime_config.py` loads the tracked Windows reproducibility configuration and
applies it automatically before model code runs.

## Requirements

- Python 3.9.13, 64-bit x86 (`x64`).
- Windows 10/11 x64 CPU is the formal reference environment for the paper
  results.
- CPU only. GPU is not required and is not selected by the default code path.
- One compute thread is enforced for BLAS, XGBoost, and PyTorch to reduce
  cross-machine variation. More CPU cores may improve operating-system
  scheduling overhead only; they do not change the configured algorithm.
- At least 8 GB RAM and 10 GB free disk space are recommended for the full
  all-years run, because the training and figure steps create temporary files.

The Windows reference lock uses pip 24.1.2, NumPy 1.21.5, pandas 1.4.4,
scikit-learn 1.2.0, PyTorch 2.0.0+cpu, and XGBoost 1.7.6.

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
$env:PYTHONWARNINGS = "ignore"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
```

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

The Windows entry points and workflows suppress third-party Python warnings,
XGBoost warning logs, and pip version notices so the run output contains only
progress, results, and actual errors.

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

No local Python installation or GPU is needed. The two workflows run on a
standard Windows Server 2022 x64 CPU runner and start from the raw files in a
clean checkout.

### 1. Fork the repository

1. Sign in to GitHub and open <https://github.com/whyhw-code/LTR-DQN>.
2. Select **Fork** in the upper-right corner.
3. Choose your account as **Owner**, keep the repository public, and select
   **Create fork**. The repository name may remain `LTR-DQN`.
4. Confirm that the fork is on the `main` branch. Generated results from older
   runs are not copied into the fork.

### 2. Enable the workflows

1. Open the **Actions** tab in your fork, not in the original repository.
2. If GitHub displays **I understand my workflows, go ahead and enable them**,
   select it once.
3. Confirm that the left side lists both workflows:

```text
1 - One-click Results and Figures (Windows CPU)
2 - One-click T6 and Figure C4 (Windows CPU)
```

### 3. Run the primary results workflow

1. Select **1 - One-click Results and Figures (Windows CPU)**.
2. Select **Run workflow**. Keep the branch as `main`; there are no experiment
   fields to fill in.
3. Select the green **Run workflow** button to start the run, then open the new
   run record to follow its steps.

The workflow verifies that the checkout contains no previous output folders,
installs the locked Windows environment, validates `parameters_windows.txt`,
runs the tests, trains all 2/3/4-year models, and generates:

```text
results/combined/results.xlsx       T3, T4, T5, and T7 workbook
results/combined/                   paper-format CSVs and manifests
results/figures/                    all main-text figures and audit CSVs
results/appendix_figures/           C1, C2, C3, and C5 plus audit CSVs
```

### 4. Run T6 separately

1. Return to **Actions** and select
   **2 - One-click T6 and Figure C4 (Windows CPU)**.
2. Select **Run workflow**, keep `main`, and confirm with the green button.

This independent workflow trains the three-year models and performs the fresh
T6 sampling run. Its outputs include:

```text
results/T6/T6.xlsx                  T6 workbook
results/appendix_figures/           Appendix Figure C4 and audit CSV
temp/t6_runs/t6_raw.csv             per-run T6 audit data
temp/t6_runs/t6_manifest.json       T6 run manifest
```

The two workflows can be run separately. Closing the browser does not stop an
active run. A yellow dot means queued or running, a green check means success,
and a red cross means that a step failed.

### 5. Download and check the results

1. Open a run marked with a green check.
2. Scroll to **Artifacts** at the bottom of the summary page.
3. Download the artifact matching the workflow:

```text
ltr-dqn-results-and-figures-<run-id>
ltr-dqn-t6-and-c4-<run-id>
```

4. Extract the ZIP and locate the workbooks and figures under the paths shown
   above. Each package also contains the run manifests and the exact
   `parameters_windows.txt` used for that run.

Artifacts are retained for 14 days. Results are not added to the repository's
source-file list, so they must be downloaded from the completed run page.

### 6. Handle a failed or outdated run

- Open the failed run, select the job, and expand the first step marked with a
  red cross. Installation failures appear under **Install locked environment**;
  training and plotting failures appear under their named steps.
- If the original repository has changed since the fork was created, open the
  fork's main page, select **Sync fork**, then **Update branch**. Start a new
  workflow run after synchronization; rerunning an old job still uses its old
  commit.
- If **Run workflow** is missing, confirm that Actions is enabled, the workflow
  file is on the default `main` branch, and the signed-in account has write
  permission to the fork.

For a screenshot-style walkthrough and a final checklist, see
[GITHUB_ACTIONS_GUIDE_CN.md](GITHUB_ACTIONS_GUIDE_CN.md).

## Reproducibility notes

- Windows reference seeds and unreported implementation settings are in
  `parameters_windows.txt`; the T6 seed ledgers are under
  `data/reproducibility/`.
- `LambdaRank` and `LambdaMART` are retrained from the raw tracked data on each
  run. DQN consumes the LambdaMART output created by that same run.
- `PYTHONHASHSEED=0`, fixed seeds, deterministic PyTorch settings, stable CSV
  ordering, and `n_jobs=1` are enabled to reduce platform variation.
- Manifests record runtime versions, input hashes, action hashes, and checkpoint
  hashes so separate Windows runs can be audited and compared.
- Keep the repository and workflow defaults unchanged when reproducing the
  reported results.
