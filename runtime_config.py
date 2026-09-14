from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from pathlib import Path


for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["ATEN_CPU_CAPABILITY"] = "default"
os.environ["MKL_CBWR"] = "COMPATIBLE"

import numpy as np

DEFAULT_DEVICE = "cpu"

# T7 score cutoffs are computed from the raw cross-sectional ESG.csv file. The
# resulting 25%/50% thresholds (5.52/6.02 for this data) are shared by both
# markets and by NS/PI; the strategies differ only in replenishment behavior.
ESG_QUANTILES = {
    "25%": 0.25,
    "50%": 0.50,
}


DEFAULT_TRAINING_SEEDS = {
    "0060": 40,
    "3068": 50,
}
DEFAULT_EVALUATION_SEED = 1795

LOCKED_RUNTIME = {
    "python": "3.9.13",
    "numpy": "1.21.5",
    "pandas": "1.4.4",
    "torch": "2.0.0+cpu",
    "xgboost": "1.7.6",
}

CODE_DIR = Path(__file__).resolve().parent
PLATFORM_PARAMETER_FILES = {
    "Windows": CODE_DIR / "parameters_windows.txt",
    "Linux": CODE_DIR / "parameters_linux.txt",
}
_MARKETS = {"0060", "3068"}
_YEARS = {"2", "3", "4"}
_STAGES = {"rank", "mart", "dqn", "baseline", "evaluation"}
_RANK_FIELDS = {
    "max_depth", "n_estimators", "subsample", "colsample_bytree", "tree_method",
}
_MART_FIELDS = {
    "max_bin", "min_child_weight", "subsample", "colsample_bytree", "tree_method",
}


def parameter_file_for_system(system_name: str | None = None) -> Path:
    """Return the tracked parameter file selected by the host OS."""
    detected = platform.system() if system_name is None else str(system_name)
    try:
        return PLATFORM_PARAMETER_FILES[detected]
    except KeyError as exc:
        supported = ", ".join(sorted(PLATFORM_PARAMETER_FILES))
        raise RuntimeError(
            f"Unsupported operating system {detected!r}; expected one of: {supported}"
        ) from exc


def _validate_market_years(section: object, name: str, fields: set[str]) -> None:
    if not isinstance(section, dict) or set(section) != _MARKETS:
        raise ValueError(f"{name} must define exactly the markets {_MARKETS}")
    for market, years in section.items():
        if not isinstance(years, dict) or set(years) != _YEARS:
            raise ValueError(f"{name}/{market} must define exactly years {_YEARS}")
        for year, values in years.items():
            if not isinstance(values, dict) or set(values) != fields:
                raise ValueError(
                    f"{name}/{market}/{year} must define exactly {sorted(fields)}"
                )


def _validate_platform_parameters(config: object, path: Path, system_name: str) -> None:
    if not isinstance(config, dict):
        raise ValueError(f"platform parameter file must contain a JSON object: {path}")
    required = {
        "schema_version", "profile", "system", "purpose",
        "stage_seeds", "rank_config", "mart_config",
    }
    if set(config) != required:
        raise ValueError(
            f"platform parameter keys must be exactly {sorted(required)}: {path}"
        )
    if config["schema_version"] != 1:
        raise ValueError(f"unsupported platform parameter schema: {path}")
    if config["system"] != system_name:
        raise ValueError(
            f"parameter file system mismatch: expected {system_name}, got {config['system']!r}"
        )
    if not isinstance(config["profile"], str) or not config["profile"].strip():
        raise ValueError(f"platform profile name is empty: {path}")
    _validate_market_years(config["stage_seeds"], "stage_seeds", _STAGES)
    _validate_market_years(config["rank_config"], "rank_config", _RANK_FIELDS)
    _validate_market_years(config["mart_config"], "mart_config", _MART_FIELDS)
    for market, years in config["stage_seeds"].items():
        for year, stages in years.items():
            if any(
                not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
                for seed in stages.values()
            ):
                raise ValueError(f"invalid stage seed in {path}: {market}/{year}")
    for section_name in ("rank_config", "mart_config"):
        for market, years in config[section_name].items():
            for year, values in years.items():
                if values["tree_method"] not in {"hist", "exact", "approx"}:
                    raise ValueError(
                        f"invalid tree_method in {path}: {section_name}/{market}/{year}"
                    )
                for name in ("subsample", "colsample_bytree"):
                    value = values[name]
                    if (
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not 0 < float(value) <= 1
                    ):
                        raise ValueError(
                            f"invalid {name} in {path}: {section_name}/{market}/{year}"
                        )
    for market, years in config["rank_config"].items():
        for year, values in years.items():
            for name in ("max_depth", "n_estimators"):
                if not isinstance(values[name], int) or values[name] <= 0:
                    raise ValueError(f"invalid {name} in {path}: {market}/{year}")
    for market, years in config["mart_config"].items():
        for year, values in years.items():
            if not isinstance(values["max_bin"], int) or values["max_bin"] < 2:
                raise ValueError(f"invalid max_bin in {path}: {market}/{year}")
            weight = values["min_child_weight"]
            if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight < 0:
                raise ValueError(f"invalid min_child_weight in {path}: {market}/{year}")


def load_platform_parameters(system_name: str | None = None) -> dict:
    """Detect the host OS and load its tracked JSON-formatted text profile."""
    detected = platform.system() if system_name is None else str(system_name)
    path = parameter_file_for_system(detected)
    if not path.is_file():
        raise FileNotFoundError(f"platform parameter file not found: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in platform parameter file: {path}") from exc
    _validate_platform_parameters(config, path, detected)
    return config


ACTIVE_SYSTEM = platform.system()
ACTIVE_PARAMETER_FILE = parameter_file_for_system(ACTIVE_SYSTEM)
ACTIVE_PLATFORM_PARAMETERS = load_platform_parameters(ACTIVE_SYSTEM)
ACTIVE_PLATFORM_PROFILE = ACTIVE_PLATFORM_PARAMETERS["profile"]
ACTIVE_PARAMETER_SHA256 = hashlib.sha256(ACTIVE_PARAMETER_FILE.read_bytes()).hexdigest()

# These three sections are the only platform-dependent defaults. Parameters
# reported by the paper remain locked in experiment_core.py and train.py.
DEFAULT_STAGE_SEEDS = json.loads(json.dumps(ACTIVE_PLATFORM_PARAMETERS["stage_seeds"]))
DEFAULT_RANK_CONFIG = json.loads(json.dumps(ACTIVE_PLATFORM_PARAMETERS["rank_config"]))
DEFAULT_MART_CONFIG = json.loads(json.dumps(ACTIVE_PLATFORM_PARAMETERS["mart_config"]))


def active_parameter_metadata() -> dict[str, str]:
    return {
        "system": ACTIVE_SYSTEM,
        "profile": ACTIVE_PLATFORM_PROFILE,
        "file": ACTIVE_PARAMETER_FILE.name,
        "sha256": ACTIVE_PARAMETER_SHA256,
    }


def load_stage_seed_config(path: str | Path | None) -> dict:
    """Load an optional JSON seed map; omitted entries use defaults."""
    if path is None:
        return DEFAULT_STAGE_SEEDS
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("seed config must be a JSON object")
    merged = json.loads(json.dumps(DEFAULT_STAGE_SEEDS))
    for code, years in config.items():
        if code not in merged or not isinstance(years, dict):
            raise ValueError(f"invalid seed config market: {code}")
        for year, stages in years.items():
            if str(year) not in merged[code] or not isinstance(stages, dict):
                raise ValueError(f"invalid seed config year: {code}/{year}")
            for stage, seed in stages.items():
                if stage not in merged[code][str(year)] or not isinstance(seed, int):
                    raise ValueError(f"invalid seed config entry: {code}/{year}/{stage}")
                merged[code][str(year)][stage] = seed
    return merged


def load_rank_config(path: str | Path | None) -> dict:
    """Load LambdaRank parameters not specified by the paper."""
    if path is None:
        return DEFAULT_RANK_CONFIG
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    merged = json.loads(json.dumps(DEFAULT_RANK_CONFIG))
    for code, years in config.items():
        if code not in merged or not isinstance(years, dict):
            raise ValueError(f"invalid rank config market: {code}")
        for year, params in years.items():
            if str(year) not in merged[code] or not isinstance(params, dict):
                raise ValueError(f"invalid rank config year: {code}/{year}")
            unknown = set(params) - {
                "max_depth", "n_estimators", "subsample", "colsample_bytree",
                "tree_method",
            }
            if unknown:
                raise ValueError(
                    f"paper-fixed or unknown LambdaRank parameters cannot be overridden: {sorted(unknown)}"
                )
            for name, value in params.items():
                if name == "tree_method":
                    if value not in {"hist", "exact", "approx"}:
                        raise ValueError(f"invalid rank tree method: {code}/{year}/{value}")
                elif name in {"max_depth", "n_estimators"}:
                    if not isinstance(value, int) or value <= 0:
                        raise ValueError(f"invalid rank config entry: {code}/{year}/{name}")
                elif (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not 0 < float(value) <= 1
                ):
                    raise ValueError(f"invalid rank sampling entry: {code}/{year}/{name}")
                merged[code][str(year)][name] = value
    return merged


def load_mart_config(path: str | Path | None) -> dict:
    """Load CPU-only LambdaMART implementation parameters."""
    if path is None:
        return DEFAULT_MART_CONFIG
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    merged = json.loads(json.dumps(DEFAULT_MART_CONFIG))
    for code, years in config.items():
        if code not in merged or not isinstance(years, dict):
            raise ValueError(f"invalid MART config market: {code}")
        for year, params in years.items():
            if str(year) not in merged[code] or not isinstance(params, dict):
                raise ValueError(f"invalid MART config year: {code}/{year}")
            unknown = set(params) - {
                "max_bin", "min_child_weight", "subsample", "colsample_bytree",
                "tree_method",
            }
            if unknown:
                raise ValueError(
                    f"paper-fixed or unknown LambdaMART parameters cannot be overridden: {sorted(unknown)}"
                )
            value = params.get("max_bin", merged[code][str(year)]["max_bin"])
            if not isinstance(value, int) or value < 2:
                raise ValueError(f"invalid MART max_bin: {code}/{year}/{value}")
            merged[code][str(year)]["max_bin"] = value
            child_weight = params.get(
                "min_child_weight",
                merged[code][str(year)]["min_child_weight"],
            )
            if not isinstance(child_weight, (int, float)) or child_weight < 0:
                raise ValueError(
                    f"invalid MART min_child_weight: {code}/{year}/{child_weight}"
                )
            merged[code][str(year)]["min_child_weight"] = float(child_weight)
            tree_method = params.get(
                "tree_method", merged[code][str(year)]["tree_method"]
            )
            if tree_method not in {"hist", "exact", "approx"}:
                raise ValueError(f"invalid MART tree method: {code}/{year}/{tree_method}")
            merged[code][str(year)]["tree_method"] = tree_method
            for name in ("subsample", "colsample_bytree"):
                sampling = params.get(name, merged[code][str(year)][name])
                if (
                    not isinstance(sampling, (int, float))
                    or isinstance(sampling, bool)
                    or not 0 < float(sampling) <= 1
                ):
                    raise ValueError(f"invalid MART sampling entry: {code}/{year}/{name}")
                merged[code][str(year)][name] = float(sampling)
    return merged


def stage_seed(
    bankuaicode: str,
    train_year: int,
    stage: str,
    config: dict | None = None,
    override: int | None = None,
) -> int:
    """Resolve one deterministic seed for a market/year/stage."""
    if override is not None:
        return int(override)
    config = DEFAULT_STAGE_SEEDS if config is None else config
    try:
        return int(config[bankuaicode][str(train_year)][stage])
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"missing seed for {bankuaicode}/{train_year}/{stage}"
        ) from exc


def market_seed(bankuaicode: str) -> int:
    """Return the validated default seed for a market."""
    try:
        return DEFAULT_TRAINING_SEEDS[bankuaicode]
    except KeyError as exc:
        raise ValueError(f"Unsupported market code: {bankuaicode}") from exc


def set_global_determinism(seed: int) -> None:
    """Configure Python, NumPy and PyTorch for repeatable execution."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def configure_torch_threads(torch_module=None) -> None:
    """Set the safe PyTorch worker pool used by the DQN path."""
    if torch_module is None:
        import torch as torch_module
    torch_module.set_num_threads(1)


def torch_device():
    """Return the fixed single-CPU device used by the reproduction."""
    import torch
    return torch.device("cpu")
