# LTR-DQN reproducibility package

This package contains the minimum Python code and locked configuration used
to train LambdaRank/LambdaMART/DQN, export Tables T3--T7, and generate the
main-text Figures 3--7 plus Appendix Figures C1--C5. Generated files are not
stored in the repository.

## Reproduction environment

- Python 3.9.13
- NumPy 1.21.5
- pandas 1.4.4
- PyTorch 2.0.0
- XGBoost 1.7.6

Install the versions in `requirements.txt` before running the workflow.

## Commands

```powershell
python train.py
python main.py --export_csvs
python Fig_main.py
python Appendix_Fig_main.py --figures C1,C2,C3,C5
python train.py --models all --years 3 --t6 --ranker_tree_method gpu_hist --t6_require_gpu
python T6_main.py
python Appendix_Fig_main.py --figures C4
```

The training commands require the original source files under `data/` and
create fresh `temp/`, `model/`, `runs/` and `results/` artifacts. The two seed
ledgers in `data/reproducibility/` are included. T6's `100%` row reuses the
T4 result from the same run; only the `50%`--`90%` cells run the seed sampling.

This archive intentionally contains no generated results, intermediate CSVs,
model checkpoints, caches or logs. Run outputs are created under `results/`,
`temp/`, `model/` and `runs/` when the commands above are executed.
