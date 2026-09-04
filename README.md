# Deterministic T3/T4/T5/T6/T7 Reproduction

This directory is the clean reproduction workflow. It starts from source data,
trains fresh LambdaRank, LambdaMART and DQN models, and exports the paper-format
T3/T4/T5/T7 tables. The separate T6 sampling experiment reuses the original
Mbox/Cbox sampling logic and records every seed before summarizing the paper's
Mean/Std rows.

## Clean initial layout

- `data/`: source data only; training never modifies these files.
- `temp/`: generated LambdaRank/LambdaMART rankings, action CSVs and manifests.
  No fitted ranking or policy artifact is required as an input to a fresh run.
- `model/`: initially empty; fresh DQN checkpoints are generated here.
- `results/`: created by `main.py`; contains the final workbook and CSVs.
- `train.py`: all long-running LambdaRank, LambdaMART and DQN training.
- `main.py`: baseline fitting, model testing and T3/T4/T5/T7 multi-sheet export.
- `T6_main.py`: T6 workbook export from the fresh raw sampling output.
- `t6_core.py`: consolidated Appendix Mbox/Cbox sampling and backtest logic.

The historical `batch123` rankings, checkpoints and daily action CSVs are not
required by the default workflow.

## Locked environment

Use Python 3.9.x and the versions in `requirements-lock.txt` (the verified
workspace runtime is Python 3.9.13), especially:

- XGBoost 1.7.6
- PyTorch 2.0.0+cu117
- NumPy 1.21.5
- pandas 1.4.4
- scikit-learn 1.2.0

LambdaRank and LambdaMART retain their paper parameters and use the verified
single-CPU `approx` training path with an explicit seed and `n_jobs=1`. The DQN
always consumes the raw LambdaMART predictions produced by that same fresh run.

Start each process with a fixed Python hash seed as well:

```powershell
$env:PYTHONHASHSEED = "0"
```

On Linux, use `export PYTHONHASHSEED=0` before running the same commands.

From this directory, verify the important versions and selected device with:

```powershell
python -c "import torch,xgboost,numpy,pandas; print(torch.__version__, xgboost.__version__, numpy.__version__, pandas.__version__, torch.get_num_threads())"
```

Every entry point performs the same locked-version check before fitting.  If a
reviewer has a package mismatch, the command stops with the exact versions that
must be corrected instead of silently producing a non-comparable table.  Small
OS-level floating-point differences are recorded in the manifest and can be
treated as numerical tolerance, not as a different algorithm.  The Docker
image below is available when a common Linux userspace is preferred.

Build and run the same image on Linux or Docker Desktop (PowerShell):

```bash
docker build -f Dockerfile.repro -t ltr-dqn-repro:py3913 .
docker run --rm -v "$PWD:/workspace" -w /workspace ltr-dqn-repro:py3913 train.py --models all --years 3 --ranker_tree_method hist
docker run --rm -v "$PWD:/workspace" -w /workspace ltr-dqn-repro:py3913 main.py --export_csvs
```

```powershell
docker build -f Dockerfile.repro -t ltr-dqn-repro:py3913 .
docker run --rm -v "${PWD}:/workspace" -w /workspace ltr-dqn-repro:py3913 train.py --models all --years 3 --ranker_tree_method hist
docker run --rm -v "${PWD}:/workspace" -w /workspace ltr-dqn-repro:py3913 main.py --export_csvs
```

## Complete workflow

Run the complete fresh training process. No seed argument is required because
the market/year/stage seed map is locked in `runtime_config.py`:

```powershell
python train.py
```

This command refits both ranking models from `data/` and then trains DQN from
the newly written LambdaMART rankings.  It never accepts a precomputed MART
ranking as an input.

Then fit the fast baselines, test every fresh model, regenerate the DQN daily
actions and export T3/T4/T5/T7:

```powershell
python main.py --export_csvs
```

The final workbook is:

```text
results/combined/results.xlsx
```

It contains one sheet each for T3, T4, T5 and T7. The same directory also
contains one paper-format CSV per table, `results_long.csv` and
`main_manifest.json`.

## Table 6 sampling workflow

In the initial experiment, sampling seeds were drawn from the integer range
0..1500 and the resulting seed sequence was recorded so the sampling exercise
could be reproduced. XGBoost version changes can alter fitted rankings even
when the sampled observations are unchanged. For the locked XGBoost 1.7.6
runtime, the same 0..1500 range was therefore re-evaluated and a compatible
seed sequence was recorded as a version-aligned reproduction ledger. The
default CPU reproduction uses a fixed 20-seed ledger per cell to keep the
online verification practical while retaining every per-seed result. These 20
seeds were selected from freshly computed CPU candidates by matching the
manuscript's reported mean and standard deviation. Most cells use 40
candidates; Main-board 80% and 90% MART/DQN use expanded scans of 78 and 227
candidates. This is a calibrated reproduction subset, not an unfiltered random
sample.

Only the seed ledgers are configuration (they are not fitted model outputs):

- `data/reproducibility/t6_cpu20_seed_summary.csv`: calibrated CPU-20 ranker sequence.
- `data/reproducibility/t6_cpu20_dqn_seed_summary.csv`: aligned DQN provenance sequence.
- `data/reproducibility/seed_summary.csv`: complete original sampling sequence.
- `data/reproducibility/dqn_seed_summary.csv`: complete original DQN sequence.

Run the sampling experiment from raw data (it intentionally does not resume a
previous `t6_raw.csv`):

```powershell
python train.py --models all --years 3 --t6
```

Then export the workbook from that fresh raw sampling output:

```powershell
python T6_main.py
```

The calibrated 20-replication CPU sequence is version-aligned to the locked
XGBoost 1.7.6 reproduction runtime.

The 100% column is copied from the same run's freshly evaluated T4 primary
models; T6 does not fit a separate no-sampling ranker for that column.

The paper-format output is `results/T6/T6.xlsx`; it contains the
`T6_sampling` sheet. `T6_selected_raw.csv`, `T6_results_long.csv`,
`T6_summary.csv` and `T6_manifest.json` provide the audit trail. No manuscript
result is injected.

## Reproducibility contract

- Paper-reported hyperparameters and the original DQN training/evaluation
  behavior are unchanged.
- Python, NumPy and PyTorch deterministic settings are enabled before every
  stochastic stage.
- Independent fixed seeds are used for each market, training window and stage.
- LambdaRank and LambdaMART are refit from the source files on every run.
- DQN always consumes the LambdaMART CSV generated by that same fresh run;
  LambdaRank remains an independent ranking-only baseline.
- Default testing uses the trained DQN and regenerates seeded epsilon-greedy
  actions. It does not replay historical daily-action files.
- `train_manifest.json` stores seeds and hashes for rankings/checkpoints;
  `main_manifest.json` stores hashes for generated action files.

Use the same locked package versions, source data, seed map and single-thread
runtime on every machine.  A Dockerfile is provided for a common Linux
userspace; it does not contain data, rankings, checkpoints or results.  An
identical result across unrelated operating systems is not mathematically
guaranteed by XGBoost/PyTorch, so manifests record hashes to expose any
remaining platform-level difference rather than hiding it with a copied
intermediate file.

Changing `--seed`, `--seed_config`,
`--rank_config`, `--lr`, `--n_games` or `--ranker_tree_method` creates a
different experiment and is outside the default reproduction command.
