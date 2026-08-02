"""
GPU capability detection for model training/optimization.

Traditional/boosted models are created via `ModelRegistry.create(name, params)`
throughout the codebase (PSO optimization, nested retuning, FS embedded
methods) with only numeric hyperparameters in scope - no `Config` object is
passed down to that call site. GPU usage is a hardware/deployment setting,
not a tunable hyperparameter, so it is threaded through as a process-wide
flag (`QDECEM_USE_GPU` env var, set once from `config.model.use_gpu` in
`Config.__post_init__`) rather than added to every PSO param space or call
site. Each backend is probed at most once per process (`lru_cache`) and
falls back to CPU with a logged warning if the installed build has no GPU
support, so `use_gpu: true` is always safe to leave on regardless of the
machine actually running the pipeline.
"""
import functools
import os

from .logging_utils import get_logger

_ENV_VAR = "QDECEM_USE_GPU"


def set_gpu_requested(enabled: bool) -> None:
    os.environ[_ENV_VAR] = "1" if enabled else "0"


def gpu_requested() -> bool:
    return os.environ.get(_ENV_VAR, "0") == "1"


@functools.lru_cache(maxsize=1)
def _catboost_gpu_available() -> bool:
    if not gpu_requested():
        return False
    try:
        import numpy as np
        from catboost import CatBoostRegressor

        probe = CatBoostRegressor(
            task_type="GPU", devices="0", iterations=2,
            verbose=False, allow_writing_files=False,
        )
        rng = np.random.default_rng(0)
        probe.fit(rng.random((10, 2)), rng.random(10))
        return True
    except Exception as exc:
        get_logger().warning(f"CatBoost GPU probe failed ({exc}); using CPU.")
        return False


@functools.lru_cache(maxsize=1)
def _lightgbm_gpu_available() -> bool:
    if not gpu_requested():
        return False
    import numpy as np
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    X_probe, y_probe = rng.random((10, 2)), rng.random(10)
    for device_type in ("cuda", "gpu"):
        try:
            probe = lgb.LGBMRegressor(device_type=device_type, n_estimators=2, verbosity=-1)
            probe.fit(X_probe, y_probe)
            return True
        except Exception:
            continue
    get_logger().warning(
        "LightGBM has no GPU-enabled build installed; using CPU. The pip "
        "wheel ships CPU-only - install a CUDA/OpenCL build (e.g. build "
        "from source with -DUSE_CUDA=1, or a conda-forge lightgbm=*=cuda* "
        "package) to enable GPU training."
    )
    return False


def catboost_gpu_kwargs() -> dict:
    if _catboost_gpu_available():
        return {"task_type": "GPU", "devices": "0"}
    return {}


def lightgbm_gpu_kwargs() -> dict:
    if _lightgbm_gpu_available():
        return {"device_type": "cuda"}
    return {}


def torch_device_str() -> str:
    if not gpu_requested():
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
