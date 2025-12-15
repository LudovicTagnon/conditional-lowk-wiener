from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> np.random.Generator:
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed)}")
    if seed < 0:
        raise ValueError("seed must be >= 0")
    rng = np.random.default_rng(seed)
    np.random.seed(seed % (2**32 - 1))
    os.environ["PYTHONHASHSEED"] = str(seed)
    return rng


def read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError("YAML config must be a mapping at top-level")
        return dict(data)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("JSON config must be an object at top-level")
        return data
    raise ValueError("Config must be .yaml/.yml or .json")


def write_yaml(path: Path, data: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(data), f, sort_keys=False)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    config_used: Path
    metrics_json: Path
    pred_vs_true_csv: Path
    pred_vs_true_png: Path
    residuals_hist_png: Path


def make_run_paths(output_root: Path, exp_name: str) -> RunPaths:
    run_dir = output_root / "runs" / f"{utc_timestamp()}_{exp_name}"
    ensure_dir(run_dir)
    return RunPaths(
        run_dir=run_dir,
        config_used=run_dir / "config_used.yaml",
        metrics_json=run_dir / "metrics.json",
        pred_vs_true_csv=run_dir / "pred_vs_true.csv",
        pred_vs_true_png=run_dir / "pred_vs_true.png",
        residuals_hist_png=run_dir / "residuals_hist.png",
    )


def assert_finite(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN/Inf")


def normalize_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    assert_finite("normalize_01(x)", x)
    mn = float(x.min())
    mx = float(x.max())
    if mx == mn:
        return np.zeros_like(x)
    y = (x - mn) / (mx - mn)
    y = np.clip(y, 0.0, 1.0)
    return y


def save_pred_vs_true(
    path: Path,
    y: np.ndarray,
    split_is_test: np.ndarray,
    mass: np.ndarray,
    preds: Mapping[str, np.ndarray],
) -> None:
    y = np.asarray(y, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    split_is_test = np.asarray(split_is_test, dtype=bool)
    assert y.shape == mass.shape == split_is_test.shape
    for k, v in preds.items():
        v = np.asarray(v, dtype=np.float64)
        if v.shape != y.shape:
            raise ValueError(f"pred {k} shape mismatch: {v.shape} vs {y.shape}")

    header = ["y_true", "split", "mass"] + [f"y_pred_{k}" for k in preds.keys()]
    rows: list[list[Any]] = []
    for i in range(y.shape[0]):
        split = "test" if split_is_test[i] else "train"
        rows.append([float(y[i]), split, float(mass[i])] + [float(preds[k][i]) for k in preds.keys()])
    write_csv(path, header, rows)


def plot_pred_vs_true(path: Path, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    assert y_true.shape == y_pred.shape

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=8, alpha=0.6)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_residuals_hist(path: Path, residuals: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    residuals = np.asarray(residuals, dtype=np.float64)
    assert residuals.ndim == 1

    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=50, alpha=0.85)
    plt.xlabel("residual = y_pred - y_true")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
