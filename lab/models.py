from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .utils import assert_finite


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    assert y_true.shape == y_pred.shape
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    assert y_true.shape == y_pred.shape
    if float(np.std(y_true)) == 0.0 or float(np.std(y_pred)) == 0.0:
        return float("nan")
    return float(pearsonr(y_true, y_pred).statistic)


@dataclass(frozen=True)
class ModelResult:
    y_pred: np.ndarray  # predictions for all samples
    metrics_test: dict[str, float]
    model: Any


def fit_predict_ridge(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    ridge_alpha: float,
) -> ModelResult:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    assert_finite("X", X)
    assert_finite("y", y)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(f"y must be (n,), got {y.shape} for X {X.shape}")

    pipe = make_pipeline(StandardScaler(), Ridge(alpha=float(ridge_alpha)))
    pipe.fit(X[train_idx], y[train_idx])
    y_pred = pipe.predict(X)
    assert_finite("y_pred", y_pred)

    y_test = y[test_idx]
    p_test = y_pred[test_idx]
    rmse_v = rmse(y_test, p_test)
    std_y = float(np.std(y_test))
    rel = float(rmse_v / std_y) if std_y > 0 else float("nan")
    metrics = {
        "pearson": safe_pearson(y_test, p_test),
        "rmse": rmse_v,
        "relRMSE": rel,
    }
    return ModelResult(y_pred=y_pred, metrics_test=metrics, model=pipe)


def deltas_vs_baseline(baseline: dict[str, float], other: dict[str, float]) -> dict[str, float]:
    return {
        "delta_pearson": float(other["pearson"] - baseline["pearson"]),
        "delta_relRMSE": float(other["relRMSE"] - baseline["relRMSE"]),
    }

