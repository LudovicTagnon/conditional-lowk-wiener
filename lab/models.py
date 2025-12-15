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


def fit_eval_ridge(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    ridge_alpha: float,
) -> tuple[np.ndarray, dict[str, float], Any]:
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
    y_pred_test = pipe.predict(X[test_idx])
    assert_finite("y_pred_test", y_pred_test)

    y_test = y[test_idx]
    rmse_v = rmse(y_test, y_pred_test)
    std_y = float(np.std(y_test))
    rel = float(rmse_v / std_y) if std_y > 0 else float("nan")
    metrics = {
        "pearson": safe_pearson(y_test, y_pred_test),
        "rmse": rmse_v,
        "relRMSE": rel,
    }
    return y_pred_test, metrics, pipe


def deltas_vs_baseline(baseline: dict[str, float], other: dict[str, float]) -> dict[str, float]:
    return {
        "delta_pearson": float(other["pearson"] - baseline["pearson"]),
        "delta_relRMSE": float(other["relRMSE"] - baseline["relRMSE"]),
    }


@dataclass(frozen=True)
class ResidualTestResult:
    metrics_residual_test: dict[str, float]
    metrics_total_test: dict[str, float]
    deltas_total_vs_B: dict[str, float]
    y_pred_B: np.ndarray
    r_pred: np.ndarray
    y_pred_total: np.ndarray


def _shuffle_columns_inplace(X: np.ndarray, idx: np.ndarray, rng: np.random.Generator) -> None:
    for j in range(X.shape[1]):
        X[idx, j] = X[idx, j][rng.permutation(idx.size)]


def residual_test(
    *,
    X_B: np.ndarray,
    X_topo: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    ridge_alpha: float,
    metrics_B_test: dict[str, float],
    y_pred_B: np.ndarray | None = None,
    rng: np.random.Generator,
    include_mass_interactions: bool = False,
    mass: np.ndarray | None = None,
    placebo_shuffle_train: bool = False,
    placebo_shuffle_test: bool = False,
) -> ResidualTestResult:
    """
    Test whether topo features explain residuals of model B.

    Steps:
    - Fit B on y, get y_pred_B
    - r = y - y_pred_B
    - Fit ridge on r using topo features (optionally add mass*topo interactions)
    - Return residual metrics and total metrics for y_total = y_pred_B + r_pred
    """
    X_B = np.asarray(X_B, dtype=np.float64)
    X_topo = np.asarray(X_topo, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    assert_finite("X_B", X_B)
    assert_finite("X_topo", X_topo)
    assert_finite("y", y)
    if X_B.shape[0] != y.shape[0] or X_topo.shape[0] != y.shape[0]:
        raise ValueError("X_B/X_topo/y sample count mismatch")
    if X_topo.ndim != 2 or X_topo.shape[1] < 1:
        raise ValueError("X_topo must be (n, p>=1)")
    if include_mass_interactions and mass is None:
        raise ValueError("mass is required when include_mass_interactions=True")

    # Fit B on y (unless provided)
    if y_pred_B is None:
        res_B = fit_predict_ridge(X_B, y, train_idx, test_idx, ridge_alpha=ridge_alpha)
        y_pred_B = res_B.y_pred
    else:
        y_pred_B = np.asarray(y_pred_B, dtype=np.float64)
        if y_pred_B.shape != y.shape:
            raise ValueError(f"y_pred_B must be (n,), got {y_pred_B.shape}")
        assert_finite("y_pred_B", y_pred_B)
    r = y - y_pred_B

    # Build residual-design matrix
    Xr = X_topo.astype(np.float64, copy=True)
    if include_mass_interactions:
        mass_v = np.asarray(mass, dtype=np.float64)
        if mass_v.shape != (y.shape[0],):
            raise ValueError(f"mass must be (n,), got {mass_v.shape}")
        inter = X_topo * mass_v[:, None]
        Xr = np.concatenate([Xr, inter], axis=1)
    if placebo_shuffle_train or placebo_shuffle_test:
        if placebo_shuffle_train:
            _shuffle_columns_inplace(Xr, train_idx, rng)
        if placebo_shuffle_test:
            _shuffle_columns_inplace(Xr, test_idx, rng)

    # Fit residual model
    res_r = fit_predict_ridge(Xr, r, train_idx, test_idx, ridge_alpha=ridge_alpha)
    r_pred = res_r.y_pred

    y_pred_total = y_pred_B + r_pred
    y_test = y[test_idx]
    total_test = y_pred_total[test_idx]
    rmse_total = rmse(y_test, total_test)
    std_y = float(np.std(y_test))
    total_metrics = {
        "pearson": safe_pearson(y_test, total_test),
        "rmse": float(rmse_total),
        "relRMSE": float(rmse_total / std_y) if std_y > 0 else float("nan"),
    }

    deltas_total_vs_B = {
        "delta_pearson": float(total_metrics["pearson"] - float(metrics_B_test["pearson"])),
        "delta_relRMSE": float(total_metrics["relRMSE"] - float(metrics_B_test["relRMSE"])),
    }

    return ResidualTestResult(
        metrics_residual_test=res_r.metrics_test,
        metrics_total_test=total_metrics,
        deltas_total_vs_B=deltas_total_vs_B,
        y_pred_B=y_pred_B,
        r_pred=r_pred,
        y_pred_total=y_pred_total,
    )
