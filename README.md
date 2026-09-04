# LTR-DQN Reproduction (CPU)

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
- `runtime_config.py`: Python/package lock values, independent stage seeds, and
  deterministic single-CPU settings. The default DQN device is CPU.

### Data and automation

- `data/0060merge_open_close_final.csv` and `data/3068merge_open_close_final.csv`:
  stock features and open/close prices used for training and evaluation.
- `data/0060merge_T4.csv` and `data/3068merge_T4.csv`: index/reference series
  used by the paper tables.
- `data/0060merge.csv` and `data/3068merge.csv`: market data used by baselines,
  figures, and T6 backtests.
- `data/dapan/`: broad-market data used by the baseline and DQN backtest paths.
- `data/ESG/`: supplied ESG ranking inputs used by T7 and Appendix C5.
- `data/reproducibility/`: the two 20-seed T6 configuration ledgers. They store
  only the seeds used by the run, not fitted outputs or selection manifests.
- `.github/workflows/reproduce-core.yml`: GitHub Actions workflow for training,
  Results, and main figures.
- `.github/workflows/reproduce-t6.yml`: GitHub Actions workflow for T6 and
  Appendix Figure C4.

### Environment and housekeeping

- `requirements-lock.txt`: exact pip lock for the CPU reproduction, including
  `torch==2.0.0+cpu`.
- `requirements.txt`: standard pip dependency list used by the GitHub workflow.
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
results/    workbooks, paper CSVs, figure PNGs, and audit CSVs
```

## Requirements

- Python 3.9.13, 64-bit x86 (`x64`).
- Windows 10/11 or Linux x86_64. GitHub Actions uses Ubuntu 22.04 x64.
- CPU only. GPU is not required and is not selected by the default code path.
- One compute thread is enforced for BLAS, XGBoost, and PyTorch to reduce
  cross-machine variation. More CPU cores may improve operating-system
  scheduling overhead only; they do not change the configured algorithm.
- At least 8 GB RAM and 10 GB free disk space are recommended for the full
  all-years run, because the training and figure steps create temporary files.

Important pinned versions include NumPy 1.21.5, pandas 1.4.4, scikit-learn
1.2.0, PyTorch 2.0.0+cpu, and XGBoost 1.7.6. The entry points check the locked
runtime before fitting and stop on a mismatch.

## Installation

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
2. Select **Reproduce Core (CPU)** and choose **Run workflow** for Results and
   main figures.
3. Select **Reproduce T6** and choose **Run workflow** for T6 and Appendix C4.
4. Open the completed run and download its artifact. The artifact contains the
   generated workbook, CSVs, figures, and manifests.

Both workflows use an x64 Ubuntu 22.04 CPU runner, Python 3.9.13, a fixed hash
seed, and single-thread settings. They do not require or request a GPU runner.

## Reproducibility notes

- The default seed map is in `runtime_config.py`; the T6 seed ledgers are under
  `data/reproducibility/`.
- `LambdaRank` and `LambdaMART` are retrained from the raw tracked data on each
  run. DQN consumes the LambdaMART output created by that same run.
- `PYTHONHASHSEED=0`, fixed seeds, deterministic PyTorch settings, stable CSV
  ordering, and `n_jobs=1` are enabled to reduce platform variation.
- Manifests record runtime versions, input hashes, action hashes, and checkpoint
  hashes. Exact byte-for-byte equality across unrelated operating systems is
  not guaranteed by XGBoost/PyTorch; a mismatch is visible in the manifests.
- Do not change `--seed`, `--seed_config`, `--lr`, `--n_games`, or
  `--ranker_tree_method` when reproducing the reported default run.
