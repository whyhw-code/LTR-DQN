# LTR-DQN Reproduction Package

This repository reproduces the empirical results for *Enhancing Predictive
Selection of Sell-Side Analyst Reports via Learning to Rank and Reinforcement
Learning* from fixed prepared input data. Fresh rankings, DQN checkpoints,
tables, and empirical figures are generated during each run; generated
artifacts are not stored in Git.

Figures 1 and 2 are schematic diagrams and are outside the numerical
reproduction workflow.

## One-click reproduction on GitHub

No local Python environment is required.

1. Open the repository's **Actions** tab.
2. Select **Reproduce Core** or **Reproduce T6**.
3. Click **Run workflow**.
4. When the workflow finishes, download its artifact from the workflow summary.

### Reproduce Core

The Core workflow trains fresh models and produces:

- Tables 3, 4, 5, and 7;
- main-text Figures 3-7;
- Appendix Figures C1, C2, C3, and C5;
- manifests containing runtime versions and hashes of generated inputs and
  outputs.

The downloadable artifact is named `ltr-dqn-core-<run-id>`.

### Reproduce T6

Table 6 is substantially more expensive than the Core workflow. The T6
workflow trains the required three-year models, partitions the fixed seed
ledger into ten deterministic cloud jobs, merges their outputs, checks that
every market/rate/model cell contains exactly 500 results, and produces:

- Table 6 and its complete selected-result audit trail;
- Appendix Figure C4;
- a manifest containing hashes for every shard and merged output.

The downloadable artifact is named `ltr-dqn-t6-<run-id>`.

## Repository layout

```text
.
|-- .github/workflows/
|   |-- reproduce-core.yml
|   `-- reproduce-t6.yml
|-- data/
|   |-- dapan/                 # prepared market-index inputs
|   |-- ESG/                   # prepared ESG evaluation inputs
|   `-- reproducibility/       # fixed T6 seed ledgers
|-- train.py                   # fresh LambdaRank/LambdaMART/DQN training
|-- main.py                    # T3/T4/T5/T7 and non-T6 figures
|-- T6_main.py                 # T6 shard merge, table, and Figure C4
|-- t6_shard_runner.py         # cloud-only deterministic T6 partition runner
|-- model.py                   # original model entry point
|-- workflow.py                # table evaluation implementation
|-- Fig_main.py                # main-text empirical figures
|-- Appendix_Fig_main.py       # appendix empirical figures
|-- requirements-lock.txt      # fully pinned Python environment
`-- .gitignore                 # excludes every generated artifact
```

The repository contains prepared model inputs, not the original analyst-report
PDF collection. Training never modifies files under `data/`.

The two files under `data/reproducibility/` are fixed experimental
configuration:

- `seed_summary.csv`: LambdaRank and LambdaMART sampling seeds;
- `dqn_seed_summary.csv`: independent LTR-DQN sampling seeds.

They are included because the paper does not report the full sampled seed
sequence and Table 6 cannot be reproduced without that ledger.

## Local reproduction (optional)

GitHub Actions is the reference execution path. For a local run, use 64-bit
Python 3.9.13 on Windows and install the CPU build of PyTorch before the locked
requirements:

```powershell
python -m pip install torch==2.0.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-lock.txt
```

Core workflow:

```powershell
python train.py
python main.py --export_csvs
```

T6 workflow without cloud sharding:

```powershell
python train.py --models all --years 3 --t6 --ranker_tree_method hist
python T6_main.py
```

The T6 command requires the three-year rankings and DQN checkpoints produced by
the Core training command. A complete single-machine T6 run can take many hours
and resumes from `temp/t6_runs/t6_raw.csv` if interrupted.

## Generated outputs

Local runs write only to ignored directories:

```text
model/                       # fresh DQN checkpoints
temp/rankings/               # fresh ranker outputs
temp/actions/                # fresh DQN evaluation actions
temp/t6_runs/                # per-seed T6 results
results/combined/            # T3/T4/T5/T7 workbook and CSVs
results/figures/             # main-text empirical figures
results/appendix_figures/    # Appendix Figures C1-C5
results/T6/                  # T6 workbook, CSVs, and manifest
```

These directories are uploaded as temporary GitHub Actions artifacts when
needed; they are never committed by the workflows.

## Locked environment

The validated runtime is:

- Python 3.9.13
- NumPy 1.21.5
- pandas 1.4.4
- PyTorch 2.0.0 CPU
- XGBoost 1.7.6

Every executable entry point checks this runtime before starting an experiment.
Changing seeds, model parameters, package versions, operating system, or the
tree method defines a different experiment and may change fitted rankings.

## Reproducibility contract

- All long-running model artifacts are generated from the checked-in prepared
  data.
- The DQN stage consumes LambdaRank output from the same run.
- Independent fixed seeds are used for every market, training window, and
  stochastic stage.
- T6 cloud shards partition the recorded seed sequence by position; merging the
  shards reconstructs the same 500-seed cells as a single-machine run.
- Manifests record runtime versions, source paths, and SHA-256 hashes.
- Workflows have read-only repository permissions and never commit results.
