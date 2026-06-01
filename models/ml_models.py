"""
Machine-learning regressors used in the benchmark.

The paper reports CPU/GPU timing separately, so the training helpers must be
able to run on ordinary CPU-only machines. GPU acceleration is used only when
the installed library and hardware support it; otherwise the same estimator is
trained with CPU parameters.
"""

import logging
import threading
import warnings
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

warnings.filterwarnings("ignore")
logging.getLogger("lightgbm").setLevel(logging.ERROR)
logging.getLogger("xgboost").setLevel(logging.ERROR)
logging.getLogger("cuml").setLevel(logging.ERROR)

try:
    from cuml.ensemble import RandomForestRegressor as cuRandomForestRegressor
    from cuml.linear_model import LinearRegression as cuLinearRegression

    GPU_RF_AVAILABLE = True
    GPU_LINEAR_AVAILABLE = True
except Exception:
    cuRandomForestRegressor = None
    cuLinearRegression = None
    GPU_RF_AVAILABLE = False
    GPU_LINEAR_AVAILABLE = False

try:
    from xgboost import XGBRegressor

    XGB_AVAILABLE = True
except ImportError:
    XGBRegressor = None
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor

    LGBM_AVAILABLE = True
except ImportError:
    LGBMRegressor = None
    LGBM_AVAILABLE = False


gpu_lock = threading.Lock()


def _clean_training_arrays(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Replace invalid numeric values before fitting sklearn-style estimators."""
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
    if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)
    return X_train, y_train


def _cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _require_package(package_name: str, is_available: bool) -> None:
    if not is_available:
        raise ImportError(
            f"{package_name} is required for this model. Install it with "
            f"`pip install {package_name}` or install the full requirements file."
        )


def train_rf(X_train, y_train, params: Dict):
    """Train a Random Forest regressor with cuML when available, else sklearn."""
    X_train, y_train = _clean_training_arrays(X_train, y_train)

    rf_params = {
        "n_estimators": int(params.get("n_estimators", 30)),
        "max_depth": int(params.get("max_depth", 3)) if params.get("max_depth") is not None else None,
        "random_state": int(params.get("random_state", 42)),
    }

    if GPU_RF_AVAILABLE and _cuda_available():
        base = cuRandomForestRegressor(**rf_params)
        return MultiOutputRegressor(base, n_jobs=1).fit(X_train, y_train)

    base = RandomForestRegressor(**rf_params, n_jobs=-1)
    return MultiOutputRegressor(base, n_jobs=1).fit(X_train, y_train)


def train_xgb(X_train, y_train, params: Dict):
    """Train an XGBoost regressor with CPU fallback."""
    _require_package("xgboost", XGB_AVAILABLE)
    X_train, y_train = _clean_training_arrays(X_train, y_train)

    base_params = {
        "n_estimators": int(params.get("n_estimators", 30)),
        "max_depth": int(params.get("max_depth", 3)) if params.get("max_depth") is not None else None,
        "learning_rate": float(params.get("learning_rate", 0.1)),
        "verbosity": int(params.get("verbosity", 0)),
        "random_state": int(params.get("random_state", 42)),
        "n_jobs": 1,
    }

    if _cuda_available():
        base_params.update({"tree_method": "hist", "device": "cuda"})
    else:
        base_params.update({"tree_method": "hist", "device": "cpu"})

    base = XGBRegressor(**base_params)
    return MultiOutputRegressor(base, n_jobs=1).fit(X_train, y_train)


def train_lgbm(X_train, y_train, params: Dict):
    """Train a LightGBM regressor with CPU fallback."""
    _require_package("lightgbm", LGBM_AVAILABLE)
    X_train, y_train = _clean_training_arrays(X_train, y_train)

    base_params = {
        "n_estimators": int(params.get("n_estimators", 30)),
        "max_depth": int(params.get("max_depth", 3)) if params.get("max_depth") is not None else -1,
        "learning_rate": float(params.get("learning_rate", 0.1)),
        "random_state": int(params.get("random_state", 42)),
        "verbose": int(params.get("verbosity", -1)),
        "n_jobs": 1,
    }

    if _cuda_available():
        base_params.update({"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})

    base = LGBMRegressor(**base_params)
    try:
        return MultiOutputRegressor(base, n_jobs=1).fit(X_train, y_train)
    except Exception as exc:
        if _cuda_available() and "GPU" in str(exc):
            base_params.pop("device", None)
            base_params.pop("gpu_platform_id", None)
            base_params.pop("gpu_device_id", None)
            base = LGBMRegressor(**base_params)
            return MultiOutputRegressor(base, n_jobs=1).fit(X_train, y_train)
        raise


def train_linear(X_train, y_train, params: Dict):
    """Train Linear Regression with cuML when available, else sklearn."""
    del params
    X_train, y_train = _clean_training_arrays(X_train, y_train)

    with gpu_lock:
        if GPU_LINEAR_AVAILABLE and _cuda_available():
            try:
                model = cuLinearRegression()
                model.fit(X_train, y_train)
                return model
            except Exception:
                pass

        model = LinearRegression()
        model.fit(X_train, y_train)
        return model
