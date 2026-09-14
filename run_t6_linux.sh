#!/usr/bin/env bash
set -euo pipefail

export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export PYTHONUTF8="${PYTHONUTF8:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"
export ATEN_CPU_CAPABILITY="${ATEN_CPU_CAPABILITY:-default}"
export MKL_CBWR="${MKL_CBWR:-COMPATIBLE}"

python -c "import platform; from runtime_config import ACTIVE_PLATFORM_PROFILE; assert platform.system() == 'Linux'; assert ACTIVE_PLATFORM_PROFILE == 'linux-emergency-compatibility'; print('Parameter profile:', ACTIVE_PLATFORM_PROFILE)"
python -m unittest discover -s tests -v
python train.py --models all --years 3 --t6 --ranker_tree_method approx
python T6_main.py
python Appendix_Fig_main.py --figures C4 --force
