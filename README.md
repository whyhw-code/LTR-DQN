# LTR-DQN Reproduction (CPU)

English | [简体中文](README_CN.md)

This repository reproduces the paper tables T3, T4, T5, T6 and T7, the main-text
figures, and Appendix Figures C1-C5. Every run starts from the tracked source
data and trains fresh rankers and DQN models. No fitted model, result workbook,
daily action file, or historical `meiri_xuanze` selection file is required.

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
- `runtime_config.py`: detects Windows/Linux, validates the selected platform
  parameter file, and applies deterministic single-CPU settings.
- `parameters_windows.txt`: formal Windows reference seeds and unreported
  implementation parameters used locally and by GitHub Actions.
- `parameters_linux.txt`: emergency Linux CPU compatibility parameters. It is
  not used by GitHub Actions.

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
- T7 thresholds are computed from the combined Main and ChiNext raw ESG files
  at runtime. The current data produce a shared q25/q50 of `5.52`/`6.02` for
  both markets and both strategies.

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
  `run_t6_linux.sh`: local/rented-server Linux emergency path. There is no
  Linux GitHub Actions workflow.
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

## Requirements

- Python 3.9.13, 64-bit x86 (`x64`).
- Windows 10/11 CPU is the reference environment for the paper results. A
  64-bit Linux x86 CPU compatibility path is also supported; Linux output is
  intended to be close overall, not bit-for-bit identical.
- CPU only. GPU is not required and is not selected by the default code path.
- One compute thread is enforced for BLAS, XGBoost, and PyTorch to reduce
  cross-machine variation. More CPU cores may improve operating-system
  scheduling overhead only; they do not change the configured algorithm.
- At least 8 GB RAM and 10 GB free disk space are recommended for the full
  all-years run, because the training and figure steps create temporary files.

Reference versions include pip 24.1.2, NumPy 1.21.5, pandas 1.4.4,
scikit-learn 1.2.0, PyTorch 2.0.0+cpu, and XGBoost 1.7.6. The Linux
compatibility file installs these same exact versions. If a device cannot
install them, set `LTR_DQN_RELAXED_RUNTIME=1` explicitly before running and
accept that the output may differ.

## Automatic platform parameter selection

No platform parameter path is required on the command line. At startup,
`runtime_config.py` checks `platform.system()` and loads exactly one tracked
JSON-formatted text file:

- Windows loads `parameters_windows.txt` and profile `windows-reference`.
- Linux loads `parameters_linux.txt` and profile
  `linux-emergency-compatibility`.
- Any other operating system stops with an error instead of silently choosing
  a profile.

The files contain only seeds and implementation settings not fixed by the
paper. Paper-reported learning rates and other reported hyperparameters remain
defined and validated in code, so the platform profiles cannot override them.
Every manifest records the profile name, file name, and SHA-256 hash.

## Installation

### Windows PowerShell

```powershell
git clone https://github.com/whyhw-code/LTR-DQN.git
Set-Location LTR-DQN
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade "pip==24.1.2"
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
python -m pip install -r requirements-linux.txt
export MPLBACKEND=Agg PYTHONHASHSEED=0 PYTHONUTF8=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE
```

The Conda alternative is `conda env create -f environment-linux.yml`, followed
by `conda activate ltr-dqn-linux`. Run `bash run_linux.sh` for the core tables
and figures, or `bash run_t6_linux.sh` for T6 and Figure C4. The scripts require
the locked versions by default; use `export LTR_DQN_RELAXED_RUNTIME=1` only
when the locked wheels are unavailable.

The Conda alternative is:

```bash
conda env create -f environment.yml
conda activate ltr-dqn
```

Verify the installation before a long run:

```bash
python -c "import sys,torch,xgboost,numpy,pandas,sklearn; print(sys.version); print(torch.__version__, xgboost.__version__, numpy.__version__, pandas.__version__, sklearn.__version__, torch.get_num_threads()); print(torch.cuda.is_available())"
python -m compileall -q *.py
```

The last line should print `False` for CUDA availability and the thread count
should be `1`.

## Complete local reproduction

Run these commands from the repository root, in order.

### 1. Train all primary models

```bash
python train.py --models all --years 2,3,4 --ranker_tree_method approx
```

This refits LambdaRank and LambdaMART from `data/`, then trains DQN from the
new LambdaMART rankings. It does not read a precomputed ranking or a fixed
daily selection file.

### 2. Export Results tables

```bash
python main.py --export_csvs
```

Main output: `results/combined/results.xlsx`, containing T3, T4, T5 and T7.
Paper-format CSVs and JSON manifests are written beside the workbook.

### 3. Generate main-text figures

```bash
python Fig_main.py --ranker_tree_method approx --force
```

Output: `results/figures/`.

### 4. Generate Appendix Figures C1, C2, C3 and C5

```bash
python Appendix_Fig_main.py --figures C1,C2,C3,C5 --force
```

Output: `results/appendix_figures/`.

When the two institution-level brokerage workbooks are present in `data/`, C2
automatically uses the `institution` column and the original daily-mean return
aggregation. A custom report file or directory can still be supplied with
`--broker_file`; use `--min_broker_reports` to change the minimum group size.

### 5. Run T6 separately

T6 is a separate 20-seed sampling experiment. Run:

```bash
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
