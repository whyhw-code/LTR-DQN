"""Process-wide reproducibility defaults loaded by Python at startup.

The file is intentionally limited to numerical runtime settings.  It does not
contain data, fitted parameters or generated results.
"""

import os


for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    # Reproduction runs must not inherit a host-specific worker count.
    os.environ[_name] = "1"

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["XGBOOST_BUILD_DOC"] = "0"
os.environ["ATEN_CPU_CAPABILITY"] = "default"
os.environ["MKL_CBWR"] = "COMPATIBLE"

# Set both PyTorch pools before any application module can create a tensor.
# Calling set_num_interop_threads later is unsafe because PyTorch may already
# have initialized its task pool.
try:
    import torch as _torch

    _torch.set_num_threads(1)
    _torch.set_num_interop_threads(1)
except Exception:
    # The runtime lock check will report an unusable PyTorch installation.
    pass
