# LTR-DQN reproducibility package

This package contains the minimum Python code and locked configuration used
to train LambdaRank/LambdaMART/DQN, export Tables T3--T7, and generate the
main-text Figures 3--7 plus Appendix Figures C1--C5.

## Reproduction environment

- Python 3.9.13
- NumPy 1.21.5
- pandas 1.4.4
- PyTorch 2.0.0+cu117 package
- XGBoost 1.7.6

LambdaRank and LambdaMART retain their paper parameters and are refit from the
source data on every run. All rankers, XGBoost baselines and DQN stages use one
CPU thread. The verified default ranker builder is `approx`; `hist` and `exact`
remain available only for diagnostic comparisons.

Install the versions in `requirements-lock.txt` before running the workflow.
Each entry point checks the lock before training and stops on a mismatch.

## Commands

```powershell
python train.py
python main.py --export_csvs
python T6_main.py
python Fig_main.py
python Appendix_Fig_main.py
```

The training commands require the original source files under `data/` and
create fresh `temp/`, `model/` and `runs/` artifacts. Those large source data
and checkpoints are intentionally excluded from this GitHub-sized package.
The two small seed ledgers in `data/reproducibility/` are included.

This archive intentionally contains no generated results, intermediate CSVs,
model checkpoints, caches or logs. Run outputs are created under `results/`,
`temp/`, `model/` and `runs/` when the commands above are executed.
