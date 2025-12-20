from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .data_synth import (
    band_split_poisson_2d,
    generate_1overf_field_2d,
    generate_1overf_field_3d,
    solve_poisson_periodic_fft,
)
from .features import b0_counts, grad_energy, patch_basic_stats
from .models import deltas_vs_baseline, fit_eval_ridge, fit_predict_ridge, residual_test
from .utils import (
    RunPaths,
    assert_finite,
    make_run_paths,
    plot_pred_vs_true,
    plot_residuals_hist,
    read_config,
    save_pred_vs_true,
    set_global_seed,
    write_csv,
    write_json,
    write_yaml,
)


def _require_odd(name: str, n: int) -> int:
    if int(n) != n:
        raise ValueError(f"{name} must be int, got {n}")
    n = int(n)
    if n <= 0 or n % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer, got {n}")
    return n


def _split_indices(n: int, test_frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < float(test_frac) < 1.0):
        raise ValueError("test_frac must be in (0,1)")
    idx = np.arange(n)
    perm = rng.permutation(idx)
    n_test = int(round(n * float(test_frac)))
    n_test = max(1, min(n - 1, n_test))
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    return train_idx, test_idx


def _infer_type(value: str) -> Any:
    v = value.strip()
    low = v.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        if any(c in v for c in [".", "e", "E"]):
            return float(v)
        return int(v)
    except ValueError:
        return v


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    out = dict(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--override expects key=value, got: {item}")
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"Invalid override key: {item}")
        if k in {"N", "n"}:
            k = "n_patches"
        out[k] = _infer_type(v)
    out["_overrides"] = list(overrides)
    return out


def _sample_patches_3d(
    rho: np.ndarray,
    g: np.ndarray,
    n_patches: int,
    patch_size: int,
    thresholds_b0: list[float],
    rng: np.random.Generator,
    shuffle_b0: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rho = np.asarray(rho, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    assert rho.shape == g.shape
    assert rho.ndim == 3
    assert_finite("rho", rho)
    assert_finite("g", g)

    ps = _require_odd("patch_size", patch_size)
    r = ps // 2
    nx, ny, nz = rho.shape
    if any(s < ps for s in (nx, ny, nz)):
        raise ValueError(f"grid too small for patch_size={ps}: {rho.shape}")

    centers = np.column_stack(
        [
            rng.integers(r, nx - r, size=n_patches),
            rng.integers(r, ny - r, size=n_patches),
            rng.integers(r, nz - r, size=n_patches),
        ]
    )

    mass = np.empty(n_patches, dtype=np.float64)
    mass2 = np.empty(n_patches, dtype=np.float64)
    var = np.empty(n_patches, dtype=np.float64)
    mx = np.empty(n_patches, dtype=np.float64)
    ge = np.empty(n_patches, dtype=np.float64)
    b0 = np.empty((n_patches, len(thresholds_b0)), dtype=np.float64)
    y = np.empty(n_patches, dtype=np.float64)

    for i, (cx, cy, cz) in enumerate(centers):
        patch = rho[cx - r : cx + r + 1, cy - r : cy + r + 1, cz - r : cz + r + 1]
        if patch.shape != (ps, ps, ps):
            raise RuntimeError(f"bad patch shape at {i}: {patch.shape}")
        if patch[r, r, r] != rho[int(cx), int(cy), int(cz)]:
            raise RuntimeError("patch center does not match rho center index (bug in sampling)")
        m, m2, v, xmx = patch_basic_stats(patch)
        mass[i] = m
        mass2[i] = m2
        var[i] = v
        mx[i] = xmx
        ge[i] = grad_energy(patch)
        b0[i, :] = np.asarray(b0_counts(patch, thresholds_b0, adjacency=6), dtype=np.float64)
        y_center = float(g[int(cx), int(cy), int(cz)])
        y[i] = y_center

    if shuffle_b0 and b0.shape[1] > 0:
        for j in range(b0.shape[1]):
            b0[:, j] = b0[rng.permutation(n_patches), j]

    feats = {
        "mass": mass,
        "mass2": mass2,
        "var": var,
        "max": mx,
        "grad_energy": ge,
    }
    for j, t in enumerate(thresholds_b0):
        feats[f"b0_{t:.1f}"] = b0[:, j]
        feats[f"b0_t{int(round(float(t) * 10)):02d}"] = b0[:, j]
    return feats, y


def _prefix_sum_3d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    ps = x.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
    out = np.zeros((x.shape[0] + 1, x.shape[1] + 1, x.shape[2] + 1), dtype=np.float64)
    out[1:, 1:, 1:] = ps
    return out


def _prefix_sum_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    ps = x.cumsum(axis=0).cumsum(axis=1)
    out = np.zeros((x.shape[0] + 1, x.shape[1] + 1), dtype=np.float64)
    out[1:, 1:] = ps
    return out


def _box_sum_2d(pref: np.ndarray, x0: np.ndarray, x1: np.ndarray, y0: np.ndarray, y1: np.ndarray) -> np.ndarray:
    return pref[x1, y1] - pref[x0, y1] - pref[x1, y0] + pref[x0, y0]


def _sample_features_2d_fast_topo(
    field: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    n_patches: int,
    patch_size: int,
    topo_mode: str,
    topo_thresholds: list[float] | None,
    thresholds_pos_sigma: list[float] | None,
    thresholds_neg_sigma: list[float] | None,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Sample patches from a 2D field (float), compute B features on that field,
    and topo features according to:
      - topo_mode='quantile_per_field': topo_thresholds must be provided (counts patch > t)
      - topo_mode='sigma': thresholds_pos_sigma / thresholds_neg_sigma are used on z-scored field
    Returns feats dict, y (n,2) (gx,gy), and centers (n,2) int.
    """
    from scipy import ndimage

    field = np.asarray(field, dtype=np.float64)
    gx = np.asarray(gx, dtype=np.float64)
    gy = np.asarray(gy, dtype=np.float64)
    assert field.shape == gx.shape == gy.shape
    assert field.ndim == 2
    assert_finite("field", field)
    assert_finite("gx", gx)
    assert_finite("gy", gy)

    ps = _require_odd("patch_size", patch_size)
    r = ps // 2
    nx, ny = field.shape
    if any(s < ps for s in (nx, ny)):
        raise ValueError(f"grid too small for patch_size={ps}: {field.shape}")

    cx = rng.integers(r, nx - r, size=n_patches, dtype=np.int64)
    cy = rng.integers(r, ny - r, size=n_patches, dtype=np.int64)

    pref1 = _prefix_sum_2d(field)
    pref2 = _prefix_sum_2d(field * field)
    x0 = (cx - r).astype(np.int64)
    x1 = (cx + r + 1).astype(np.int64)
    y0 = (cy - r).astype(np.int64)
    y1 = (cy + r + 1).astype(np.int64)

    mass = _box_sum_2d(pref1, x0, x1, y0, y1)
    nvox = float(ps * ps)
    mean = mass / nvox
    sumsq = _box_sum_2d(pref2, x0, x1, y0, y1)
    var = np.maximum(0.0, sumsq / nvox - mean * mean)
    mass2 = mass * mass

    max_grid = ndimage.maximum_filter(field, size=ps, mode="constant", cval=-np.inf)
    mx = max_grid[cx, cy]

    gx_f, gy_f = np.gradient(field)
    egrid = gx_f * gx_f + gy_f * gy_f
    eavg = ndimage.uniform_filter(egrid, size=ps, mode="constant", cval=0.0)
    ge = eavg[cx, cy]

    feats: dict[str, np.ndarray] = {
        "mass": mass.astype(np.float64, copy=False),
        "mass2": mass2.astype(np.float64, copy=False),
        "var": var.astype(np.float64, copy=False),
        "max": mx.astype(np.float64, copy=False),
        "grad_energy": ge.astype(np.float64, copy=False),
    }

    structure = ndimage.generate_binary_structure(2, 1)  # 4-neigh
    topo_mode = str(topo_mode).lower()
    if topo_mode == "quantile_per_field":
        if topo_thresholds is None:
            raise ValueError("topo_thresholds required for quantile_per_field")
        thresholds = [float(t) for t in topo_thresholds]
        names = [f"q{int(round(q * 100)):02d}" for q in [0.6, 0.7, 0.8, 0.9]][: len(thresholds)]
        b0 = np.empty((n_patches, len(thresholds)), dtype=np.float64)
        for i in range(n_patches):
            patch = field[cx[i] - r : cx[i] + r + 1, cy[i] - r : cy[i] + r + 1]
            for j, t in enumerate(thresholds):
                binary = patch > t
                if not binary.any():
                    b0[i, j] = 0.0
                else:
                    _, num = ndimage.label(binary, structure=structure)
                    b0[i, j] = float(num)
        for j, nm in enumerate(names):
            feats[f"b0_{nm}"] = b0[:, j]
    elif topo_mode == "sigma":
        if thresholds_pos_sigma is None or thresholds_neg_sigma is None:
            raise ValueError("thresholds_pos_sigma and thresholds_neg_sigma required for sigma")
        # z-score the whole field for topo thresholding
        mu = float(field.mean())
        sd = float(field.std())
        field_z = (field - mu) / sd if sd > 0 else (field - mu)
        tpos = [float(t) for t in thresholds_pos_sigma]
        tneg = [float(t) for t in thresholds_neg_sigma]
        b0p = np.empty((n_patches, len(tpos)), dtype=np.float64)
        b0n = np.empty((n_patches, len(tneg)), dtype=np.float64)
        names_p = [f"t{int(round(t * 10)):02d}" for t in tpos]
        names_n = [f"t{int(round(t * 10)):02d}" for t in tneg]
        for i in range(n_patches):
            patch = field_z[cx[i] - r : cx[i] + r + 1, cy[i] - r : cy[i] + r + 1]
            for j, t in enumerate(tpos):
                binary = patch > t
                if not binary.any():
                    b0p[i, j] = 0.0
                else:
                    _, num = ndimage.label(binary, structure=structure)
                    b0p[i, j] = float(num)
            for j, t in enumerate(tneg):
                binary = patch < (-t)
                if not binary.any():
                    b0n[i, j] = 0.0
                else:
                    _, num = ndimage.label(binary, structure=structure)
                    b0n[i, j] = float(num)
        for j, nm in enumerate(names_p):
            feats[f"b0_pos_{nm}"] = b0p[:, j]
        for j, nm in enumerate(names_n):
            feats[f"b0_neg_{nm}"] = b0n[:, j]
    else:
        raise ValueError("topo_mode must be quantile_per_field or sigma")

    y = np.column_stack([gx[cx, cy], gy[cx, cy]]).astype(np.float64, copy=False)
    centers = np.column_stack([cx, cy]).astype(np.int64, copy=False)
    return feats, y, centers


def _ridge_fit_predict_multi(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize X using train stats, center y, solve ridge for multi-output.
    Returns (y_pred_train, y_pred_test, w, x_mu, x_sd, y_mean).
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    if y_train.ndim != 2:
        raise ValueError("y_train must be (n,2)")
    x_mu = X_train.mean(axis=0)
    x_sd = X_train.std(axis=0)
    x_sd = np.where(x_sd > 0, x_sd, 1.0)
    Xtr = (X_train - x_mu) / x_sd
    Xte = (X_test - x_mu) / x_sd
    y_mean = y_train.mean(axis=0)
    yc = y_train - y_mean
    XtX = Xtr.T @ Xtr
    Xty = Xtr.T @ yc
    w = np.linalg.solve(XtX + float(alpha) * np.eye(XtX.shape[0]), Xty)
    y_pred_train = Xtr @ w + y_mean
    y_pred_test = Xte @ w + y_mean
    return y_pred_train, y_pred_test, w, x_mu, x_sd, y_mean


def _box_sum_3d(
    pref: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    z0: np.ndarray,
    z1: np.ndarray,
) -> np.ndarray:
    # pref is padded (nx+1, ny+1, nz+1), indices use [x0,x1), etc.
    return (
        pref[x1, y1, z1]
        - pref[x0, y1, z1]
        - pref[x1, y0, z1]
        - pref[x1, y1, z0]
        + pref[x0, y0, z1]
        + pref[x0, y1, z0]
        + pref[x1, y0, z0]
        - pref[x0, y0, z0]
    )


def _compute_b0_thresholds_for_run(cfg: dict[str, Any], field01: np.ndarray) -> tuple[dict[str, float], dict[str, Any]]:
    """
    threshold_mode:
      - fixed: thresholds_b0=[0.4,0.5,0.6,0.7] -> names t04,t05,t06,t07
      - quantile_global: quantiles_b0=[0.6,0.7,0.8,0.9] -> names q60,q70,q80,q90 (thresholds derived from field)
    """
    mode = str(cfg.get("topo_threshold_mode", cfg.get("threshold_mode", "fixed"))).lower()
    if mode == "fixed":
        thresholds = [float(x) for x in cfg.get("thresholds_b0", [0.4, 0.5, 0.6, 0.7])]
        names = [f"t{int(round(t * 10)):02d}" for t in thresholds]
        return dict(zip(names, thresholds)), {"threshold_mode": "fixed", "thresholds": thresholds}
    if mode in {"quantile_global", "quantile_per_field"}:
        qs = [float(x) for x in cfg.get("quantiles_b0", [0.6, 0.7, 0.8, 0.9])]
        if any((q <= 0.0) or (q >= 1.0) for q in qs):
            raise ValueError("quantiles_b0 must be in (0,1)")
        field01 = np.asarray(field01, dtype=np.float64)
        if field01.ndim < 1:
            raise ValueError("field01 must be array-like")
        if (field01.min() < -1e-6) or (field01.max() > 1.0 + 1e-6):
            raise ValueError("field01 must be in [0,1] for quantile_global thresholds")
        thr = [float(np.quantile(field01.reshape(-1), q)) for q in qs]
        names = [f"q{int(round(q * 100)):02d}" for q in qs]
        return dict(zip(names, thr)), {"threshold_mode": mode, "quantiles": qs, "thresholds": thr}
    raise ValueError(f"Unknown threshold_mode: {mode} (expected fixed|quantile_global|quantile_per_field)")


def _sample_features_3d_fast(
    rho: np.ndarray,
    g: np.ndarray,
    n_patches: int,
    patch_size: int,
    topo_thresholds_by_name: dict[str, float],
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Faster sampler: compute mass/var/max/grad_energy via filters/prefix sums; loop only for b0 labeling.
    """
    from scipy import ndimage

    rho = np.asarray(rho, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    assert rho.shape == g.shape
    assert rho.ndim == 3
    assert_finite("rho", rho)
    assert_finite("g", g)

    ps = _require_odd("patch_size", patch_size)
    r = ps // 2
    nx, ny, nz = rho.shape
    if any(s < ps for s in (nx, ny, nz)):
        raise ValueError(f"grid too small for patch_size={ps}: {rho.shape}")

    cx = rng.integers(r, nx - r, size=n_patches, dtype=np.int64)
    cy = rng.integers(r, ny - r, size=n_patches, dtype=np.int64)
    cz = rng.integers(r, nz - r, size=n_patches, dtype=np.int64)

    # Prefix sums for mass and variance
    pref1 = _prefix_sum_3d(rho)
    pref2 = _prefix_sum_3d(rho * rho)
    x0 = (cx - r).astype(np.int64)
    x1 = (cx + r + 1).astype(np.int64)
    y0 = (cy - r).astype(np.int64)
    y1 = (cy + r + 1).astype(np.int64)
    z0 = (cz - r).astype(np.int64)
    z1 = (cz + r + 1).astype(np.int64)

    mass = _box_sum_3d(pref1, x0, x1, y0, y1, z0, z1)
    nvox = float(ps * ps * ps)
    mean = mass / nvox
    sumsq = _box_sum_3d(pref2, x0, x1, y0, y1, z0, z1)
    var = np.maximum(0.0, sumsq / nvox - mean * mean)
    mass2 = mass * mass

    # Max via maximum_filter
    max_grid = ndimage.maximum_filter(rho, size=ps, mode="constant", cval=-np.inf)
    mx = max_grid[cx, cy, cz]

    # Grad energy via global gradients + uniform_filter
    gx, gy, gz = np.gradient(rho)
    egrid = gx * gx + gy * gy + gz * gz
    egrid = np.asarray(egrid, dtype=np.float64)
    eavg = ndimage.uniform_filter(egrid, size=ps, mode="constant", cval=0.0)
    ge = eavg[cx, cy, cz]

    # b0 (loop)
    names = list(topo_thresholds_by_name.keys())
    thresholds = [float(topo_thresholds_by_name[n]) for n in names]
    b0 = np.empty((n_patches, len(names)), dtype=np.float64)
    for i in range(n_patches):
        patch = rho[cx[i] - r : cx[i] + r + 1, cy[i] - r : cy[i] + r + 1, cz[i] - r : cz[i] + r + 1]
        if patch.shape != (ps, ps, ps):
            raise RuntimeError(f"bad patch shape at {i}: {patch.shape}")
        if i < 3 and patch[r, r, r] != rho[int(cx[i]), int(cy[i]), int(cz[i])]:
            raise RuntimeError("patch center does not match rho center index (bug in sampling)")
        b0[i, :] = np.asarray(b0_counts(patch, thresholds, adjacency=6), dtype=np.float64)

    y = g[cx, cy, cz].astype(np.float64, copy=False)
    assert_finite("y", y)

    feats: dict[str, np.ndarray] = {
        "mass": mass.astype(np.float64, copy=False),
        "mass2": mass2.astype(np.float64, copy=False),
        "var": var.astype(np.float64, copy=False),
        "max": mx.astype(np.float64, copy=False),
        "grad_energy": ge.astype(np.float64, copy=False),
    }
    for j, n in enumerate(names):
        feats[f"b0_{n}"] = b0[:, j]
    return feats, y


def _sample_features_3d_fast_signed_topo(
    rho: np.ndarray,
    g: np.ndarray,
    n_patches: int,
    patch_size: int,
    thresholds_pos: list[float],
    thresholds_neg: list[float],
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Like _sample_features_3d_fast, but compute topo features:
    - b0_pos_tXX = #CC(patch > t) for t in thresholds_pos (including t=0.0)
    - b0_neg_tXX = #CC(patch < -t) for t in thresholds_neg
    """
    from scipy import ndimage

    rho = np.asarray(rho, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    assert rho.shape == g.shape
    assert rho.ndim == 3
    assert_finite("rho", rho)
    assert_finite("g", g)

    ps = _require_odd("patch_size", patch_size)
    r = ps // 2
    nx, ny, nz = rho.shape
    if any(s < ps for s in (nx, ny, nz)):
        raise ValueError(f"grid too small for patch_size={ps}: {rho.shape}")

    cx = rng.integers(r, nx - r, size=n_patches, dtype=np.int64)
    cy = rng.integers(r, ny - r, size=n_patches, dtype=np.int64)
    cz = rng.integers(r, nz - r, size=n_patches, dtype=np.int64)

    # Prefix sums for mass and variance
    pref1 = _prefix_sum_3d(rho)
    pref2 = _prefix_sum_3d(rho * rho)
    x0 = (cx - r).astype(np.int64)
    x1 = (cx + r + 1).astype(np.int64)
    y0 = (cy - r).astype(np.int64)
    y1 = (cy + r + 1).astype(np.int64)
    z0 = (cz - r).astype(np.int64)
    z1 = (cz + r + 1).astype(np.int64)

    mass = _box_sum_3d(pref1, x0, x1, y0, y1, z0, z1)
    nvox = float(ps * ps * ps)
    mean = mass / nvox
    sumsq = _box_sum_3d(pref2, x0, x1, y0, y1, z0, z1)
    var = np.maximum(0.0, sumsq / nvox - mean * mean)
    mass2 = mass * mass

    max_grid = ndimage.maximum_filter(rho, size=ps, mode="constant", cval=-np.inf)
    mx = max_grid[cx, cy, cz]

    gx, gy, gz = np.gradient(rho)
    egrid = gx * gx + gy * gy + gz * gz
    egrid = np.asarray(egrid, dtype=np.float64)
    eavg = ndimage.uniform_filter(egrid, size=ps, mode="constant", cval=0.0)
    ge = eavg[cx, cy, cz]

    structure = ndimage.generate_binary_structure(3, 1)  # 6-neigh
    thresholds_pos = [float(t) for t in thresholds_pos]
    thresholds_neg = [float(t) for t in thresholds_neg]
    names_pos = [f"t{int(round(t * 10)):02d}" for t in thresholds_pos]
    names_neg = [f"t{int(round(t * 10)):02d}" for t in thresholds_neg]
    b0p = np.empty((n_patches, len(names_pos)), dtype=np.float64)
    b0n = np.empty((n_patches, len(names_neg)), dtype=np.float64)
    for i in range(n_patches):
        patch = rho[cx[i] - r : cx[i] + r + 1, cy[i] - r : cy[i] + r + 1, cz[i] - r : cz[i] + r + 1]
        if patch.shape != (ps, ps, ps):
            raise RuntimeError(f"bad patch shape at {i}: {patch.shape}")
        for j, t in enumerate(thresholds_pos):
            binary = patch > t
            if not binary.any():
                b0p[i, j] = 0.0
            else:
                _, num = ndimage.label(binary, structure=structure)
                b0p[i, j] = float(num)
        for j, t in enumerate(thresholds_neg):
            binary = patch < (-t)
            if not binary.any():
                b0n[i, j] = 0.0
            else:
                _, num = ndimage.label(binary, structure=structure)
                b0n[i, j] = float(num)

    y = g[cx, cy, cz].astype(np.float64, copy=False)
    assert_finite("y", y)

    feats: dict[str, np.ndarray] = {
        "mass": mass.astype(np.float64, copy=False),
        "mass2": mass2.astype(np.float64, copy=False),
        "var": var.astype(np.float64, copy=False),
        "max": mx.astype(np.float64, copy=False),
        "grad_energy": ge.astype(np.float64, copy=False),
    }
    for j, n in enumerate(names_pos):
        feats[f"b0_pos_{n}"] = b0p[:, j]
    for j, n in enumerate(names_neg):
        feats[f"b0_neg_{n}"] = b0n[:, j]
    return feats, y


def _zscore_field(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = float(x.mean())
    sd = float(x.std())
    if sd <= 0:
        return x - mu
    return (x - mu) / sd


def _ridge_metrics_from_blocks(
    B_train: np.ndarray,
    T_train: np.ndarray | None,
    y_train: np.ndarray,
    B_test: np.ndarray,
    T_test: np.ndarray | None,
    y_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Closed-form ridge with train-standardization and y centering.
    If T_* is None: fits only on B.
    Returns y_pred_test and metrics dict.
    """
    from .models import rmse, safe_pearson

    B_train = np.asarray(B_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    B_test = np.asarray(B_test, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)
    if T_train is not None:
        T_train = np.asarray(T_train, dtype=np.float64)
        T_test = np.asarray(T_test, dtype=np.float64)

    def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = train.mean(axis=0)
        sd = train.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        return (train - mu) / sd, (test - mu) / sd

    y_mean = float(y_train.mean())
    yc = y_train - y_mean

    Btr, Bte = standardize(B_train, B_test)
    if T_train is None:
        Xtr = Btr
        Xte = Bte
    else:
        Ttr, Tte = standardize(T_train, T_test)
        Xtr = np.concatenate([Btr, Ttr], axis=1)
        Xte = np.concatenate([Bte, Tte], axis=1)

    XtX = Xtr.T @ Xtr
    Xty = Xtr.T @ yc
    w = np.linalg.solve(XtX + float(alpha) * np.eye(XtX.shape[0]), Xty)
    y_pred = Xte @ w + y_mean
    rm = rmse(y_test, y_pred)
    std_y = float(np.std(y_test))
    rel = float(rm / std_y) if std_y > 0 else float("nan")
    metrics = {"pearson": safe_pearson(y_test, y_pred), "rmse": float(rm), "relRMSE": rel}
    return y_pred, metrics


def _sample_patches_2d(
    sigma: np.ndarray,
    g_proj: np.ndarray,
    n_patches: int,
    patch_size: int,
    thresholds_b0: list[float],
    rng: np.random.Generator,
    shuffle_b0: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    sigma = np.asarray(sigma, dtype=np.float64)
    g_proj = np.asarray(g_proj, dtype=np.float64)
    assert sigma.shape == g_proj.shape
    assert sigma.ndim == 2
    assert_finite("sigma", sigma)
    assert_finite("g_proj", g_proj)

    ps = _require_odd("patch_size", patch_size)
    r = ps // 2
    nx, ny = sigma.shape
    if any(s < ps for s in (nx, ny)):
        raise ValueError(f"grid too small for patch_size={ps}: {sigma.shape}")

    centers = np.column_stack(
        [rng.integers(r, nx - r, size=n_patches), rng.integers(r, ny - r, size=n_patches)]
    )

    mass = np.empty(n_patches, dtype=np.float64)
    mass2 = np.empty(n_patches, dtype=np.float64)
    var = np.empty(n_patches, dtype=np.float64)
    mx = np.empty(n_patches, dtype=np.float64)
    ge = np.empty(n_patches, dtype=np.float64)
    b0 = np.empty((n_patches, len(thresholds_b0)), dtype=np.float64)
    y = np.empty(n_patches, dtype=np.float64)

    for i, (cx, cy) in enumerate(centers):
        patch = sigma[cx - r : cx + r + 1, cy - r : cy + r + 1]
        if patch.shape != (ps, ps):
            raise RuntimeError(f"bad patch shape at {i}: {patch.shape}")
        if patch[r, r] != sigma[int(cx), int(cy)]:
            raise RuntimeError("patch center does not match sigma center index (bug in sampling)")
        m, m2, v, xmx = patch_basic_stats(patch)
        mass[i] = m
        mass2[i] = m2
        var[i] = v
        mx[i] = xmx
        ge[i] = grad_energy(patch)
        b0[i, :] = np.asarray(b0_counts(patch, thresholds_b0, adjacency=4), dtype=np.float64)
        y_center = float(g_proj[int(cx), int(cy)])
        y[i] = y_center

    if shuffle_b0 and b0.shape[1] > 0:
        for j in range(b0.shape[1]):
            b0[:, j] = b0[rng.permutation(n_patches), j]

    feats = {
        "mass": mass,
        "mass2": mass2,
        "var": var,
        "max": mx,
        "grad_energy": ge,
    }
    for j, t in enumerate(thresholds_b0):
        feats[f"b0_{t:.1f}"] = b0[:, j]
        feats[f"b0_t{int(round(float(t) * 10)):02d}"] = b0[:, j]
    return feats, y


def _build_feature_matrix(feats: dict[str, np.ndarray], keys: list[str]) -> np.ndarray:
    X = np.column_stack([np.asarray(feats[k], dtype=np.float64) for k in keys])
    assert_finite("X", X)
    return X


def _run_models(
    feats: dict[str, np.ndarray],
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    thresholds_b0: list[float],
    ridge_alpha: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    y = np.asarray(y, dtype=np.float64)
    assert_finite("y", y)
    n = y.shape[0]
    for v in feats.values():
        if np.asarray(v).shape[0] != n:
            raise ValueError("feature length mismatch")

    keys_A = ["mass", "mass2"]
    keys_B = keys_A + ["var", "max", "grad_energy"]
    keys_C = keys_B + [f"b0_{t:.1f}" for t in thresholds_b0]

    res: dict[str, Any] = {"models": {}, "deltas_vs_A": {}}
    preds: dict[str, np.ndarray] = {}

    X_A = _build_feature_matrix(feats, keys_A)
    rA = fit_predict_ridge(X_A, y, train_idx, test_idx, ridge_alpha=ridge_alpha)
    res["models"]["A"] = rA.metrics_test
    preds["A"] = rA.y_pred

    X_B = _build_feature_matrix(feats, keys_B)
    rB = fit_predict_ridge(X_B, y, train_idx, test_idx, ridge_alpha=ridge_alpha)
    res["models"]["B"] = rB.metrics_test
    res["deltas_vs_A"]["B"] = deltas_vs_baseline(rA.metrics_test, rB.metrics_test)
    preds["B"] = rB.y_pred

    X_C = _build_feature_matrix(feats, keys_C)
    rC = fit_predict_ridge(X_C, y, train_idx, test_idx, ridge_alpha=ridge_alpha)
    res["models"]["C"] = rC.metrics_test
    res["deltas_vs_A"]["C"] = deltas_vs_baseline(rA.metrics_test, rC.metrics_test)
    preds["C"] = rC.y_pred
    return res, preds


def _metrics_by_mass_bins(
    mass: np.ndarray,
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    test_idx: np.ndarray,
    n_bins: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    from .models import rmse, safe_pearson

    mass = np.asarray(mass, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    test_mass = mass[test_idx]
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(test_mass, qs)
    edges[0] -= 1e-12
    edges[-1] += 1e-12

    rows: list[dict[str, Any]] = []
    bin_ids = np.full(test_mass.shape, -1, dtype=int)
    for b in range(n_bins):
        lo, hi = float(edges[b]), float(edges[b + 1])
        mask = (test_mass > lo) & (test_mass <= hi)
        bin_ids[mask] = b
        idx_b = test_idx[mask]
        if idx_b.size == 0:
            continue
        yb = y_true[idx_b]
        std_y = float(np.std(yb))
        row: dict[str, Any] = {
            "bin": b,
            "mass_q_low": float(qs[b]),
            "mass_q_high": float(qs[b + 1]),
            "mass_edge_low": lo,
            "mass_edge_high": hi,
            "n": int(idx_b.size),
        }
        for m in ["A", "B", "C"]:
            pb = preds[m][idx_b]
            row[f"pearson_{m}"] = safe_pearson(yb, pb)
            r = rmse(yb, pb)
            row[f"relRMSE_{m}"] = float(r / std_y) if std_y > 0 else float("nan")
        rows.append(row)
    return rows, bin_ids


def _plot_metrics_by_mass_bin(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    bins = [r["bin"] for r in rows]
    pearson_A = [r["pearson_A"] for r in rows]
    pearson_B = [r["pearson_B"] for r in rows]
    pearson_C = [r["pearson_C"] for r in rows]
    rel_A = [r["relRMSE_A"] for r in rows]
    rel_B = [r["relRMSE_B"] for r in rows]
    rel_C = [r["relRMSE_C"] for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    ax[0].plot(bins, pearson_A, marker="o", label="A")
    ax[0].plot(bins, pearson_B, marker="o", label="B")
    ax[0].plot(bins, pearson_C, marker="o", label="C")
    ax[0].set_title("Pearson (test) par bin de masse")
    ax[0].set_xlabel("bin (quantiles)")
    ax[0].set_ylabel("Pearson")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(bins, rel_A, marker="o", label="A")
    ax[1].plot(bins, rel_B, marker="o", label="B")
    ax[1].plot(bins, rel_C, marker="o", label="C")
    ax[1].set_title("relRMSE (test) par bin de masse")
    ax[1].set_xlabel("bin (quantiles)")
    ax[1].set_ylabel("relRMSE")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_experiment(cfg: dict[str, Any]) -> RunPaths:
    experiment = str(cfg.get("experiment", "")).lower()
    exp_name = str(cfg.get("exp_name", experiment or "exp"))
    seed = int(cfg.get("seed", 0))
    rng = set_global_seed(seed)
    split_seed = int(cfg.get("split_seed", seed))
    rng_split = np.random.default_rng(split_seed)
    placebo_seed = int(cfg.get("placebo_seed", split_seed))
    rng_placebo = np.random.default_rng(placebo_seed)

    output_root = Path(cfg.get("output_root", "outputs"))
    paths = make_run_paths(output_root=output_root, exp_name=exp_name)
    write_yaml(paths.config_used, cfg)

    thresholds_b0 = [float(x) for x in cfg.get("thresholds_b0", [0.4, 0.5, 0.6, 0.7])]
    n_patches = int(cfg.get("n_patches", 10_000))
    patch_size = int(cfg.get("patch_size", cfg.get("window_size", 9)))
    test_frac = float(cfg.get("test_frac", 0.3))
    ridge_alpha = float(cfg.get("ridge_alpha", 1.0))

    if experiment in {"e0", "e1", "e2"}:
        grid_size = int(cfg.get("grid_size", 64))
        alpha = float(cfg.get("alpha", 2.0))

        rho = generate_1overf_field_3d((grid_size, grid_size, grid_size), alpha=alpha, rng=rng)
        if (rho.min() < -1e-9) or (rho.max() > 1.0 + 1e-9):
            raise ValueError("rho expected in [0,1] after normalization")

        sol = solve_poisson_periodic_fft(rho)
        shuffle_b0 = experiment == "e1"
        feats, y = _sample_patches_3d(
            rho=sol.rho,
            g=sol.g,
            n_patches=n_patches,
            patch_size=patch_size,
            thresholds_b0=thresholds_b0,
            rng=rng,
            shuffle_b0=shuffle_b0,
        )

        train_idx, test_idx = _split_indices(n_patches, test_frac=test_frac, rng=rng_split)
        metrics, preds = _run_models(
            feats=feats,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            thresholds_b0=thresholds_b0,
            ridge_alpha=ridge_alpha,
        )
        metrics.update(
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "split_seed": split_seed,
                "n_patches": n_patches,
                "patch_size": patch_size,
                "test_frac": test_frac,
                "ridge_alpha": ridge_alpha,
                "grid_size": grid_size,
                "alpha": alpha,
                "thresholds_b0": thresholds_b0,
                "shuffle_b0": shuffle_b0,
            }
        )

        split_is_test = np.zeros(n_patches, dtype=bool)
        split_is_test[test_idx] = True
        save_pred_vs_true(
            paths.pred_vs_true_csv,
            y=y,
            split_is_test=split_is_test,
            mass=np.asarray(feats["mass"], dtype=np.float64),
            preds=preds,
        )

        y_test = y[test_idx]
        pC_test = preds["C"][test_idx]
        plot_pred_vs_true(
            paths.pred_vs_true_png,
            y_true=y_test,
            y_pred=pC_test,
            title=f"{exp_name} (model C, test)",
        )
        plot_residuals_hist(
            paths.residuals_hist_png,
            residuals=(pC_test - y_test),
            title=f"{exp_name} residuals (model C, test)",
        )

        if experiment == "e2":
            n_bins = int(cfg.get("mass_bins", 5))
            rows, _ = _metrics_by_mass_bins(
                mass=np.asarray(feats["mass"], dtype=np.float64),
                y_true=y,
                preds=preds,
                test_idx=test_idx,
                n_bins=n_bins,
            )
            header = list(rows[0].keys()) if rows else []
            csv_rows = [[r[h] for h in header] for r in rows]
            write_csv(paths.run_dir / "metrics_by_mass_bin.csv", header, csv_rows)
            _plot_metrics_by_mass_bin(paths.run_dir / "metric_vs_mass_bin.png", rows)
            metrics["mass_bins"] = n_bins

        write_json(paths.metrics_json, metrics)
        return paths

    if experiment in {"e4", "e5"}:
        grid_size = int(cfg.get("grid_size", 64))
        alpha = float(cfg.get("alpha", 2.0))

        rho = generate_1overf_field_3d((grid_size, grid_size, grid_size), alpha=alpha, rng=rng)
        if (rho.min() < -1e-9) or (rho.max() > 1.0 + 1e-9):
            raise ValueError("rho expected in [0,1] after normalization")
        sol = solve_poisson_periodic_fft(rho)

        feats, y = _sample_patches_3d(
            rho=sol.rho,
            g=sol.g,
            n_patches=n_patches,
            patch_size=patch_size,
            thresholds_b0=thresholds_b0,
            rng=rng,
            shuffle_b0=False,
        )
        train_idx, test_idx = _split_indices(n_patches, test_frac=test_frac, rng=rng_split)

        base_metrics, base_preds = _run_models(
            feats=feats,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            thresholds_b0=thresholds_b0,
            ridge_alpha=ridge_alpha,
        )

        keys_A = ["mass", "mass2"]
        keys_B = keys_A + ["var", "max", "grad_energy"]
        X_B = _build_feature_matrix(feats, keys_B)

        include_mass_interactions = bool(cfg.get("include_mass_interactions", False))
        if experiment == "e4":
            b0_threshold = float(cfg.get("b0_threshold", 0.5))
            topo_keys = [f"b0_t{int(round(b0_threshold * 10)):02d}"]
        else:
            topo_keys = [f"b0_t{int(round(float(t) * 10)):02d}" for t in thresholds_b0]

        X_topo = _build_feature_matrix(feats, topo_keys)
        metrics_B_test = base_metrics["models"]["B"]
        y_pred_B = base_preds["B"]

        real = residual_test(
            X_B=X_B,
            X_topo=X_topo,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            ridge_alpha=ridge_alpha,
            metrics_B_test=metrics_B_test,
            y_pred_B=y_pred_B,
            rng=rng_placebo,
            include_mass_interactions=include_mass_interactions,
            mass=np.asarray(feats["mass"], dtype=np.float64),
            placebo_shuffle_train=False,
            placebo_shuffle_test=False,
        )
        placebo = residual_test(
            X_B=X_B,
            X_topo=X_topo,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            ridge_alpha=ridge_alpha,
            metrics_B_test=metrics_B_test,
            y_pred_B=y_pred_B,
            rng=rng_placebo,
            include_mass_interactions=include_mass_interactions,
            mass=np.asarray(feats["mass"], dtype=np.float64),
            placebo_shuffle_train=True,
            placebo_shuffle_test=False,
        )

        metrics: dict[str, Any] = dict(base_metrics)
        metrics["experiment"] = experiment
        metrics["exp_name"] = exp_name
        metrics["seed"] = seed
        metrics["split_seed"] = split_seed
        metrics["placebo_seed"] = placebo_seed
        metrics["n_patches"] = n_patches
        metrics["patch_size"] = patch_size
        metrics["test_frac"] = test_frac
        metrics["ridge_alpha"] = ridge_alpha
        metrics["grid_size"] = grid_size
        metrics["alpha"] = alpha
        metrics["thresholds_b0"] = thresholds_b0
        metrics["include_mass_interactions"] = include_mass_interactions
        metrics["topo_keys"] = topo_keys
        metrics["residual_test"] = {
            "real": {
                "metrics_residual_test": real.metrics_residual_test,
                "metrics_total_test": real.metrics_total_test,
                "deltas_total_vs_B": real.deltas_total_vs_B,
            },
            "placebo": {
                "metrics_residual_test": placebo.metrics_residual_test,
                "metrics_total_test": placebo.metrics_total_test,
                "deltas_total_vs_B": placebo.deltas_total_vs_B,
            },
        }

        split_is_test = np.zeros(n_patches, dtype=bool)
        split_is_test[test_idx] = True
        preds = dict(base_preds)
        preds["total_real"] = real.y_pred_total
        preds["total_placebo"] = placebo.y_pred_total
        save_pred_vs_true(
            paths.pred_vs_true_csv,
            y=y,
            split_is_test=split_is_test,
            mass=np.asarray(feats["mass"], dtype=np.float64),
            preds=preds,
        )

        y_test = y[test_idx]
        p_total_test = real.y_pred_total[test_idx]
        plot_pred_vs_true(
            paths.pred_vs_true_png,
            y_true=y_test,
            y_pred=p_total_test,
            title=f"{exp_name} (B+residual, test)",
        )
        plot_residuals_hist(
            paths.residuals_hist_png,
            residuals=(p_total_test - y_test),
            title=f"{exp_name} total residuals (test)",
        )

        write_json(paths.metrics_json, metrics)
        return paths

    if experiment in {"e6", "e7"}:
        grid_size = int(cfg.get("grid_size", 64))
        alpha = float(cfg.get("alpha", 2.0))
        topo_thresholds_by_name, topo_thr_meta = _compute_b0_thresholds_for_run(cfg, field01=np.zeros((1, 1, 1)))

        rho = generate_1overf_field_3d((grid_size, grid_size, grid_size), alpha=alpha, rng=rng)
        sol = solve_poisson_periodic_fft(rho)
        if (sol.rho.min() < -1e-9) or (sol.rho.max() > 1.0 + 1e-9):
            raise ValueError("rho expected in [0,1] after normalization")

        # If quantile_global, recompute thresholds from the actual field.
        topo_thresholds_by_name, topo_thr_meta = _compute_b0_thresholds_for_run(cfg, field01=sol.rho)

        feats, y = _sample_features_3d_fast(
            rho=sol.rho,
            g=sol.g,
            n_patches=n_patches,
            patch_size=patch_size,
            topo_thresholds_by_name=topo_thresholds_by_name,
            rng=rng,
        )
        train_idx, test_idx = _split_indices(n_patches, test_frac=test_frac, rng=rng_split)

        # Fit B
        keys_B = ["mass", "mass2", "var", "max", "grad_energy"]
        X_B = _build_feature_matrix(feats, keys_B)
        res_B = fit_predict_ridge(X_B, y, train_idx, test_idx, ridge_alpha=ridge_alpha)
        metrics_B_test = res_B.metrics_test

        topo_keys = [f"b0_{n}" for n in topo_thresholds_by_name.keys()]
        X_topo = _build_feature_matrix(feats, topo_keys)

        real = residual_test(
            X_B=X_B,
            X_topo=X_topo,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            ridge_alpha=ridge_alpha,
            metrics_B_test=metrics_B_test,
            y_pred_B=res_B.y_pred,
            rng=rng_placebo,
            include_mass_interactions=False,
            placebo_shuffle_train=False,
            placebo_shuffle_test=False,
        )
        placebo = residual_test(
            X_B=X_B,
            X_topo=X_topo,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            ridge_alpha=ridge_alpha,
            metrics_B_test=metrics_B_test,
            y_pred_B=res_B.y_pred,
            rng=rng_placebo,
            include_mass_interactions=False,
            placebo_shuffle_train=True,
            placebo_shuffle_test=False,
        )

        metrics: dict[str, Any] = {
            "experiment": experiment,
            "exp_name": exp_name,
            "seed": seed,
            "split_seed": split_seed,
            "placebo_seed": placebo_seed,
            "n_patches": n_patches,
            "patch_size": patch_size,
            "test_frac": test_frac,
            "ridge_alpha": ridge_alpha,
            "grid_size": grid_size,
            "alpha": alpha,
            "topo_keys": topo_keys,
            "topo_thresholds": topo_thr_meta,
            "model_B_test": metrics_B_test,
            "residual_test": {
                "real": {
                    "metrics_residual_test": real.metrics_residual_test,
                    "metrics_total_test": real.metrics_total_test,
                    "deltas_total_vs_B": real.deltas_total_vs_B,
                },
                "placebo": {
                    "metrics_residual_test": placebo.metrics_residual_test,
                    "metrics_total_test": placebo.metrics_total_test,
                    "deltas_total_vs_B": placebo.deltas_total_vs_B,
                },
            },
        }

        split_is_test = np.zeros(n_patches, dtype=bool)
        split_is_test[test_idx] = True
        save_pred_vs_true(
            paths.pred_vs_true_csv,
            y=y,
            split_is_test=split_is_test,
            mass=np.asarray(feats["mass"], dtype=np.float64),
            preds={"B": res_B.y_pred, "total_real": real.y_pred_total, "total_placebo": placebo.y_pred_total},
        )
        y_test = y[test_idx]
        p_total_test = real.y_pred_total[test_idx]
        plot_pred_vs_true(
            paths.pred_vs_true_png,
            y_true=y_test,
            y_pred=p_total_test,
            title=f"{exp_name} (B+topo residual, test)",
        )
        plot_residuals_hist(
            paths.residuals_hist_png,
            residuals=(p_total_test - y_test),
            title=f"{exp_name} total residuals (test)",
        )
        write_json(paths.metrics_json, metrics)
        return paths

    if experiment == "e8":
        # LOFO generalization across independent 1/f field realizations
        grid_size = int(cfg.get("grid_size", 64))
        alpha = float(cfg.get("alpha", 2.0))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 20_000))
        n_patches_total = int(n_fields * patches_per_field)
        if n_fields <= 1:
            raise ValueError("n_fields must be >= 2 for LOFO")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")

        rho_fields: list[np.ndarray] = []
        g_fields: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho = generate_1overf_field_3d((grid_size, grid_size, grid_size), alpha=alpha, rng=rng_field)
            sol = solve_poisson_periodic_fft(rho)
            rho_fields.append(sol.rho)
            g_fields.append(sol.g)

        mode = str(cfg.get("topo_threshold_mode", cfg.get("threshold_mode", "fixed"))).lower()
        topo_thresholds_by_name_global: dict[str, float] | None = None
        topo_thr_meta_global: dict[str, Any] | None = None
        per_field_thresholds: dict[int, dict[str, float]] = {}
        per_field_thresholds_meta: dict[int, dict[str, Any]] = {}

        if mode != "quantile_per_field":
            # thresholds computed once per run from the global field values (all realizations)
            field_stack = np.concatenate([rf.reshape(-1) for rf in rho_fields], axis=0)
            topo_thresholds_by_name_global, topo_thr_meta_global = _compute_b0_thresholds_for_run(cfg, field01=field_stack)

        # sample per field
        feats_lists: dict[str, list[np.ndarray]] = {}
        ys: list[np.ndarray] = []
        field_ids: list[np.ndarray] = []
        for field_id in range(n_fields):
            if mode == "quantile_per_field":
                topo_thresholds_by_name, topo_thr_meta = _compute_b0_thresholds_for_run(cfg, field01=rho_fields[field_id])
                per_field_thresholds[field_id] = topo_thresholds_by_name
                per_field_thresholds_meta[field_id] = topo_thr_meta
            else:
                assert topo_thresholds_by_name_global is not None and topo_thr_meta_global is not None
                topo_thresholds_by_name = topo_thresholds_by_name_global
                topo_thr_meta = topo_thr_meta_global

            feats_f, y_f = _sample_features_3d_fast(
                rho=rho_fields[field_id],
                g=g_fields[field_id],
                n_patches=patches_per_field,
                patch_size=patch_size,
                topo_thresholds_by_name=topo_thresholds_by_name,
                rng=rng,
            )
            for k, v in feats_f.items():
                feats_lists.setdefault(k, []).append(np.asarray(v, dtype=np.float64))
            ys.append(np.asarray(y_f, dtype=np.float64))
            field_ids.append(np.full((patches_per_field,), field_id, dtype=np.int32))

        feats = {k: np.concatenate(vs, axis=0) for k, vs in feats_lists.items()}
        y = np.concatenate(ys, axis=0)
        fid = np.concatenate(field_ids, axis=0)
        if y.shape[0] != n_patches_total or fid.shape[0] != n_patches_total:
            raise RuntimeError("LOFO dataset size mismatch")

        topo_keys = sorted([k for k in feats.keys() if k.startswith("b0_")])
        if not topo_keys:
            raise RuntimeError("No topo features found (expected b0_*)")

        keys_B = ["mass", "mass2", "var", "max", "grad_energy"]
        X_B = _build_feature_matrix(feats, keys_B)
        X_topo = _build_feature_matrix(feats, topo_keys)
        X_C = np.concatenate([X_B, X_topo], axis=1)

        # placebo: shuffle topo within each field to preserve per-field marginals
        X_topo_pl = X_topo.copy()
        for field_id in range(n_fields):
            idx = np.nonzero(fid == field_id)[0]
            perm = rng_placebo.permutation(idx.size)
            for j in range(X_topo_pl.shape[1]):
                X_topo_pl[idx, j] = X_topo_pl[idx[perm], j]
        X_C_pl = np.concatenate([X_B, X_topo_pl], axis=1)

        # inter-field shift stats
        field_stats: list[dict[str, Any]] = []
        for field_id in range(n_fields):
            idx = np.nonzero(fid == field_id)[0]
            b0_mean = float(np.mean(X_topo[idx].mean(axis=1)))
            b0_std = float(np.std(X_topo[idx].mean(axis=1)))
            field_stats.append(
                {
                    "field_id": int(field_id),
                    "n": int(idx.size),
                    "mass_mean": float(np.mean(feats["mass"][idx])),
                    "mass_std": float(np.std(feats["mass"][idx])),
                    "var_mean": float(np.mean(feats["var"][idx])),
                    "var_std": float(np.std(feats["var"][idx])),
                    "max_mean": float(np.mean(feats["max"][idx])),
                    "max_std": float(np.std(feats["max"][idx])),
                    "grad_energy_mean": float(np.mean(feats["grad_energy"][idx])),
                    "grad_energy_std": float(np.std(feats["grad_energy"][idx])),
                    "b0_mean": b0_mean,
                    "b0_std": b0_std,
                    "y_mean": float(np.mean(y[idx])),
                    "y_std": float(np.std(y[idx])),
                }
            )

        fold_rows: list[dict[str, Any]] = []
        dP_real: list[float] = []
        dR_real: list[float] = []
        dP_pl: list[float] = []
        dR_pl: list[float] = []
        dP_diff: list[float] = []
        for field_id in range(n_fields):
            test_idx = np.nonzero(fid == field_id)[0]
            train_idx = np.nonzero(fid != field_id)[0]

            yB_test, mB, _ = fit_eval_ridge(X_B, y, train_idx, test_idx, ridge_alpha=ridge_alpha)
            yC_test, mC, _ = fit_eval_ridge(X_C, y, train_idx, test_idx, ridge_alpha=ridge_alpha)
            yCp_test, mCp, _ = fit_eval_ridge(X_C_pl, y, train_idx, test_idx, ridge_alpha=ridge_alpha)

            dp = float(mC["pearson"] - mB["pearson"])
            dr = float(mC["relRMSE"] - mB["relRMSE"])
            dpp = float(mCp["pearson"] - mB["pearson"])
            drp = float(mCp["relRMSE"] - mB["relRMSE"])

            # direction check: corr(b0_mean, residual_B) on the test field
            y_test = y[test_idx]
            resid_B = y_test - yB_test
            b0_mean_test = X_topo[test_idx].mean(axis=1)
            from .models import safe_pearson

            corr_b0_resid = float(safe_pearson(b0_mean_test, resid_B))

            fold_rows.append(
                {
                    "field_id": int(field_id),
                    "n_test": int(test_idx.size),
                    "pearson_B": float(mB["pearson"]),
                    "relRMSE_B": float(mB["relRMSE"]),
                    "pearson_C": float(mC["pearson"]),
                    "relRMSE_C": float(mC["relRMSE"]),
                    "pearson_C_placebo": float(mCp["pearson"]),
                    "relRMSE_C_placebo": float(mCp["relRMSE"]),
                    "delta_pearson_real": dp,
                    "delta_relRMSE_real": dr,
                    "delta_pearson_placebo": dpp,
                    "delta_relRMSE_placebo": drp,
                    "corr_b0mean_residB": corr_b0_resid,
                }
            )
            dP_real.append(dp)
            dR_real.append(dr)
            dP_pl.append(dpp)
            dR_pl.append(drp)
            dP_diff.append(dp - dpp)

        dP_real = np.asarray(dP_real, dtype=np.float64)
        dR_real = np.asarray(dR_real, dtype=np.float64)
        dP_pl = np.asarray(dP_pl, dtype=np.float64)
        dR_pl = np.asarray(dR_pl, dtype=np.float64)
        dP_diff = np.asarray(dP_diff, dtype=np.float64)
        snr = float(dP_diff.mean() / dP_diff.std(ddof=1)) if float(dP_diff.std(ddof=1)) > 0 else float("inf")

        mean_dp = float(dP_real.mean())
        mean_dr = float(dR_real.mean())
        verdict = bool((mean_dp > 0.0) and (mean_dr < 0.0) and (snr > 2.0))

        metrics: dict[str, Any] = {
            "experiment": experiment,
            "exp_name": exp_name,
            "seed": seed,
            "split_seed": split_seed,
            "placebo_seed": placebo_seed,
            "n_fields": n_fields,
            "patches_per_field": patches_per_field,
            "n_patches_total": n_patches_total,
            "patch_size": patch_size,
            "grid_size": grid_size,
            "alpha": alpha,
            "ridge_alpha": ridge_alpha,
            "topo_keys": topo_keys,
            "topo_thresholds": topo_thr_meta_global if topo_thr_meta_global is not None else {"threshold_mode": "quantile_per_field"},
            "topo_thresholds_per_field": per_field_thresholds_meta,
            "field_stats": field_stats,
            "lofo_folds": fold_rows,
            "lofo_agg": {
                "delta_pearson_real_mean": float(dP_real.mean()),
                "delta_pearson_real_std": float(dP_real.std(ddof=1)),
                "delta_relRMSE_real_mean": float(dR_real.mean()),
                "delta_relRMSE_real_std": float(dR_real.std(ddof=1)),
                "delta_pearson_placebo_mean": float(dP_pl.mean()),
                "delta_pearson_placebo_std": float(dP_pl.std(ddof=1)),
                "delta_relRMSE_placebo_mean": float(dR_pl.mean()),
                "delta_relRMSE_placebo_std": float(dR_pl.std(ddof=1)),
                "snr_delta_pearson": snr,
                "verdict_pass": verdict,
            },
        }

        # save per-fold CSV for convenience
        header = list(fold_rows[0].keys()) if fold_rows else []
        rows = [[r[h] for h in header] for r in fold_rows]
        write_csv(paths.run_dir / "lofo_by_field.csv", header, rows)
        write_json(paths.metrics_json, metrics)
        return paths

    if experiment == "e11":
        # LOFO with per-field z-scoring and sigma-threshold topo; permutation-based placebo per fold.
        grid_size = int(cfg.get("grid_size", 64))
        alpha = float(cfg.get("alpha", 2.0))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 20_000))
        n_patches_total = int(n_fields * patches_per_field)
        if n_fields <= 1:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        P = int(cfg.get("placebo_permutations", 30))
        if P <= 0:
            raise ValueError("placebo_permutations must be > 0")

        normalize_y_per_field = bool(cfg.get("normalize_y_per_field", True))
        thresholds_pos = [float(x) for x in cfg.get("thresholds_pos_sigma", [0.0, 0.5, 1.0, 1.5])]
        thresholds_neg = [float(x) for x in cfg.get("thresholds_neg_sigma", [0.5, 1.0, 1.5])]

        feats_lists: dict[str, list[np.ndarray]] = {}
        ys: list[np.ndarray] = []
        field_ids: list[np.ndarray] = []
        field_stats: list[dict[str, Any]] = []
        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_3d((grid_size, grid_size, grid_size), alpha=alpha, rng=rng_field)
            rho_z = _zscore_field(rho01)
            sol = solve_poisson_periodic_fft(rho_z)
            g = sol.g
            if normalize_y_per_field:
                g = _zscore_field(g)

            feats_f, y_f = _sample_features_3d_fast_signed_topo(
                rho=sol.rho,
                g=g,
                n_patches=patches_per_field,
                patch_size=patch_size,
                thresholds_pos=thresholds_pos,
                thresholds_neg=thresholds_neg,
                rng=rng,
            )
            for k, v in feats_f.items():
                feats_lists.setdefault(k, []).append(np.asarray(v, dtype=np.float64))
            y_f = np.asarray(y_f, dtype=np.float64)
            ys.append(y_f)
            field_ids.append(np.full((patches_per_field,), field_id, dtype=np.int32))

            # quick shift stats
            b0_cols = [kk for kk in feats_f.keys() if kk.startswith("b0_")]
            b0_mean = float(np.mean(np.column_stack([feats_f[k] for k in b0_cols]).mean(axis=1))) if b0_cols else float("nan")
            b0_std = float(np.std(np.column_stack([feats_f[k] for k in b0_cols]).mean(axis=1))) if b0_cols else float("nan")
            field_stats.append(
                {
                    "field_id": int(field_id),
                    "n": int(patches_per_field),
                    "mass_mean": float(np.mean(feats_f["mass"])),
                    "mass_std": float(np.std(feats_f["mass"])),
                    "var_mean": float(np.mean(feats_f["var"])),
                    "var_std": float(np.std(feats_f["var"])),
                    "max_mean": float(np.mean(feats_f["max"])),
                    "max_std": float(np.std(feats_f["max"])),
                    "grad_energy_mean": float(np.mean(feats_f["grad_energy"])),
                    "grad_energy_std": float(np.std(feats_f["grad_energy"])),
                    "b0_mean": b0_mean,
                    "b0_std": b0_std,
                    "y_mean": float(np.mean(y_f)),
                    "y_std": float(np.std(y_f)),
                }
            )

        feats = {k: np.concatenate(vs, axis=0) for k, vs in feats_lists.items()}
        y = np.concatenate(ys, axis=0)
        fid = np.concatenate(field_ids, axis=0)
        if y.shape[0] != n_patches_total:
            raise RuntimeError("dataset size mismatch")

        keys_B = ["mass", "mass2", "var", "max", "grad_energy"]
        topo_keys = sorted([k for k in feats.keys() if k.startswith("b0_")])
        X_B = _build_feature_matrix(feats, keys_B)
        X_T = _build_feature_matrix(feats, topo_keys)

        from .models import safe_pearson
        from scipy.stats import chi2

        folds: list[dict[str, Any]] = []
        deltas_real_p: list[float] = []
        deltas_real_r: list[float] = []
        zscores_p: list[float] = []
        pvals_p: list[float] = []
        pvals_r: list[float] = []
        n_pos_p = 0

        for field_id in range(n_fields):
            test_idx = np.nonzero(fid == field_id)[0]
            train_idx = np.nonzero(fid != field_id)[0]

            y_train = y[train_idx]
            y_test = y[test_idx]
            B_train = X_B[train_idx]
            B_test = X_B[test_idx]
            T_train = X_T[train_idx]
            T_test = X_T[test_idx]

            _, mB = _ridge_metrics_from_blocks(B_train, None, y_train, B_test, None, y_test, alpha=ridge_alpha)
            yC_pred, mC = _ridge_metrics_from_blocks(B_train, T_train, y_train, B_test, T_test, y_test, alpha=ridge_alpha)

            dp_real = float(mC["pearson"] - mB["pearson"])
            dr_real = float(mC["relRMSE"] - mB["relRMSE"])
            deltas_real_p.append(dp_real)
            deltas_real_r.append(dr_real)
            if dp_real > 0:
                n_pos_p += 1

            # direction check on test fold: corr(mean topo, residual_B)
            # compute B-only predictions to get residuals
            yB_pred, _ = _ridge_metrics_from_blocks(B_train, None, y_train, B_test, None, y_test, alpha=ridge_alpha)
            resid_B = y_test - yB_pred
            corr_dir = float(safe_pearson(T_test.mean(axis=1), resid_B))

            # permutation null
            null_dp: list[float] = []
            null_dr: list[float] = []
            # Precompute train standardization for blocks for speed: standardize B and T with train stats.
            muB = B_train.mean(axis=0)
            sdB = np.where(B_train.std(axis=0) > 0, B_train.std(axis=0), 1.0)
            muT = T_train.mean(axis=0)
            sdT = np.where(T_train.std(axis=0) > 0, T_train.std(axis=0), 1.0)
            Btr = (B_train - muB) / sdB
            Bte = (B_test - muB) / sdB
            Ttr = (T_train - muT) / sdT
            Tte = (T_test - muT) / sdT
            y_mean = float(y_train.mean())
            yc = y_train - y_mean

            # constants
            SBB = Btr.T @ Btr
            STT = Ttr.T @ Ttr
            SBy = Btr.T @ yc
            pB = Btr.shape[1]
            pT = Ttr.shape[1]
            I = np.eye(pB + pT)

            for _ in range(P):
                perm_tr = rng_placebo.permutation(train_idx.size)
                inv_tr = np.argsort(perm_tr)
                # topo permuted => use inv permutation on B and y when forming cross terms
                B_perm = Btr[inv_tr]
                y_perm = yc[inv_tr]
                SBT = B_perm.T @ Ttr
                STy = Ttr.T @ y_perm
                XtX = np.block([[SBB, SBT], [SBT.T, STT]])
                Xty = np.concatenate([SBy, STy], axis=0)
                w = np.linalg.solve(XtX + float(ridge_alpha) * I, Xty)
                wB = w[:pB]
                wT = w[pB:]

                perm_te = rng_placebo.permutation(test_idx.size)
                Tte_perm = Tte[perm_te]
                y_pred = Bte @ wB + Tte_perm @ wT + y_mean
                # metrics
                pear = float(safe_pearson(y_test, y_pred))
                rm = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
                std_y = float(np.std(y_test))
                rel = float(rm / std_y) if std_y > 0 else float("nan")
                null_dp.append(pear - float(mB["pearson"]))
                null_dr.append(rel - float(mB["relRMSE"]))

            null_dp_a = np.asarray(null_dp, dtype=np.float64)
            null_dr_a = np.asarray(null_dr, dtype=np.float64)
            ndp_mean = float(null_dp_a.mean())
            ndp_std = float(null_dp_a.std(ddof=1)) if null_dp_a.size > 1 else 0.0
            ndr_mean = float(null_dr_a.mean())
            ndr_std = float(null_dr_a.std(ddof=1)) if null_dr_a.size > 1 else 0.0

            z_p = float((dp_real - ndp_mean) / ndp_std) if ndp_std > 0 else float("inf")
            z_r = float((dr_real - ndr_mean) / ndr_std) if ndr_std > 0 else float("inf")
            # empirical p with +1 smoothing
            p_emp_p = float((np.sum(null_dp_a >= dp_real) + 1) / (P + 1))
            p_emp_r = float((np.sum(null_dr_a <= dr_real) + 1) / (P + 1))

            zscores_p.append(z_p)
            pvals_p.append(p_emp_p)
            pvals_r.append(p_emp_r)

            folds.append(
                {
                    "field_id": int(field_id),
                    "n_test": int(test_idx.size),
                    "pearson_B": float(mB["pearson"]),
                    "relRMSE_B": float(mB["relRMSE"]),
                    "pearson_C": float(mC["pearson"]),
                    "relRMSE_C": float(mC["relRMSE"]),
                    "delta_pearson_real": dp_real,
                    "delta_relRMSE_real": dr_real,
                    "null_pearson_mean": ndp_mean,
                    "null_pearson_std": ndp_std,
                    "null_relRMSE_mean": ndr_mean,
                    "null_relRMSE_std": ndr_std,
                    "zscore_pearson": z_p,
                    "zscore_relRMSE": z_r,
                    "p_emp_pearson": p_emp_p,
                    "p_emp_relRMSE": p_emp_r,
                    "corr(b0_mean,resid_B)": corr_dir,
                }
            )

        deltas_real_p = np.asarray(deltas_real_p, dtype=np.float64)
        deltas_real_r = np.asarray(deltas_real_r, dtype=np.float64)
        zscores_p = np.asarray(zscores_p, dtype=np.float64)
        pvals_p = np.asarray(pvals_p, dtype=np.float64)
        pvals_r = np.asarray(pvals_r, dtype=np.float64)

        # Fisher combine p-values
        eps = 1e-12
        stat_p = float(-2.0 * np.sum(np.log(np.clip(pvals_p, eps, 1.0))))
        stat_r = float(-2.0 * np.sum(np.log(np.clip(pvals_r, eps, 1.0))))
        fisher_p = float(chi2.sf(stat_p, 2 * n_fields))
        fisher_r = float(chi2.sf(stat_r, 2 * n_fields))

        verdict = bool(
            (float(deltas_real_p.mean()) > 0.0)
            and (float(deltas_real_r.mean()) < 0.0)
            and (fisher_p < 0.01)
            and (n_pos_p >= 7)
        )

        metrics: dict[str, Any] = {
            "experiment": experiment,
            "exp_name": exp_name,
            "seed": seed,
            "split_seed": split_seed,
            "placebo_seed": placebo_seed,
            "n_fields": n_fields,
            "patches_per_field": patches_per_field,
            "n_patches_total": n_patches_total,
            "patch_size": patch_size,
            "grid_size": grid_size,
            "alpha": alpha,
            "ridge_alpha": ridge_alpha,
            "normalize_y_per_field": normalize_y_per_field,
            "topo_keys": topo_keys,
            "thresholds_pos_sigma": thresholds_pos,
            "thresholds_neg_sigma": thresholds_neg,
            "placebo_permutations": P,
            "field_stats": field_stats,
            "lofo_folds": folds,
            "lofo_agg": {
                "delta_pearson_real_mean": float(deltas_real_p.mean()),
                "delta_pearson_real_std": float(deltas_real_p.std(ddof=1)),
                "delta_relRMSE_real_mean": float(deltas_real_r.mean()),
                "delta_relRMSE_real_std": float(deltas_real_r.std(ddof=1)),
                "mean_zscore_pearson": float(np.mean(zscores_p)),
                "fisher_p_pearson": fisher_p,
                "fisher_p_relRMSE": fisher_r,
                "n_fields": int(n_fields),
                "n_pos_delta_pearson": int(n_pos_p),
                "verdict_pass": verdict,
            },
        }

        header = list(folds[0].keys()) if folds else []
        rows = [[r[h] for h in header] for r in folds]
        write_csv(paths.run_dir / "lofo_by_field.csv", header, rows)
        write_json(paths.metrics_json, metrics)
        return paths

    if experiment == "e12":
        # 2D LOFO with band-splitting and vector target y=(gx,gy) (high-k by default).
        grid_size = int(cfg.get("grid_size", 128))
        alpha = float(cfg.get("alpha", 2.0))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 20_000))
        if n_fields <= 1:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        k0_frac = float(cfg.get("k0_frac", 0.15))
        P = int(cfg.get("placebo_permutations", 200))
        if P <= 0:
            raise ValueError("placebo_permutations must be > 0")

        feature_field = str(cfg.get("feature_field", "rho_high")).lower()  # rho or rho_high
        target_field = str(cfg.get("target_field", "high")).lower()  # high or full
        topo_mode = str(cfg.get("topo_mode", "quantile_per_field")).lower()  # quantile_per_field or sigma

        quantiles = [float(x) for x in cfg.get("quantiles_b0", [0.6, 0.7, 0.8, 0.9])]
        thresholds_pos_sigma = [float(x) for x in cfg.get("thresholds_pos_sigma", [0.0, 0.5, 1.0, 1.5])]
        thresholds_neg_sigma = [float(x) for x in cfg.get("thresholds_neg_sigma", [0.5, 1.0, 1.5])]

        # build dataset per field
        feats_lists: dict[str, list[np.ndarray]] = {}
        ys: list[np.ndarray] = []
        field_ids: list[np.ndarray] = []
        split_checks: list[dict[str, float]] = []
        topo_thr_per_field: dict[int, list[float]] = {}

        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
            split = band_split_poisson_2d(rho01, k0_frac=k0_frac)
            split_checks.append({"field_id": float(field_id), "rel_err_gx": split.rel_err_gx, "rel_err_gy": split.rel_err_gy})

            if feature_field == "rho":
                field_feat = rho01
            elif feature_field == "rho_high":
                field_feat = split.rho_high
            else:
                raise ValueError("feature_field must be 'rho' or 'rho_high'")

            if target_field == "high":
                gx = split.high.gx
                gy = split.high.gy
            elif target_field == "full":
                gx = split.full.gx
                gy = split.full.gy
            else:
                raise ValueError("target_field must be 'high' or 'full'")

            topo_thresholds: list[float] | None = None
            if topo_mode == "quantile_per_field":
                topo_thresholds = [float(np.quantile(field_feat.reshape(-1), q)) for q in quantiles]
                topo_thr_per_field[field_id] = topo_thresholds

            feats_f, y_f, _ = _sample_features_2d_fast_topo(
                field=field_feat,
                gx=gx,
                gy=gy,
                n_patches=patches_per_field,
                patch_size=patch_size,
                topo_mode=topo_mode,
                topo_thresholds=topo_thresholds,
                thresholds_pos_sigma=thresholds_pos_sigma if topo_mode == "sigma" else None,
                thresholds_neg_sigma=thresholds_neg_sigma if topo_mode == "sigma" else None,
                rng=rng,
            )
            for k, v in feats_f.items():
                feats_lists.setdefault(k, []).append(np.asarray(v, dtype=np.float64))
            ys.append(np.asarray(y_f, dtype=np.float64))
            field_ids.append(np.full((patches_per_field,), field_id, dtype=np.int32))

        feats = {k: np.concatenate(vs, axis=0) for k, vs in feats_lists.items()}
        y = np.concatenate(ys, axis=0)
        fid = np.concatenate(field_ids, axis=0)
        n_total = int(n_fields * patches_per_field)
        if y.shape != (n_total, 2):
            raise RuntimeError(f"y shape mismatch: {y.shape}")

        keys_B = ["mass", "mass2", "var", "max", "grad_energy"]
        topo_keys = sorted([k for k in feats.keys() if k.startswith("b0_")])
        if not topo_keys:
            raise RuntimeError("no topo features found")
        X_B = _build_feature_matrix(feats, keys_B)
        X_T = _build_feature_matrix(feats, topo_keys)

        def metrics_vec(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
            from .models import rmse, safe_pearson

            out = {}
            for i, name in enumerate(["gx", "gy"]):
                yt = y_true[:, i]
                yp = y_pred[:, i]
                out[f"pearson_{name}"] = float(safe_pearson(yt, yp))
                rm = float(rmse(yt, yp))
                std = float(np.std(yt))
                out[f"relRMSE_{name}"] = float(rm / std) if std > 0 else float("nan")
            out["pearson_mean"] = float(np.mean([out["pearson_gx"], out["pearson_gy"]]))
            out["relRMSE_mean"] = float(np.mean([out["relRMSE_gx"], out["relRMSE_gy"]]))
            return out

        from scipy.stats import chi2

        fold_rows: list[dict[str, Any]] = []
        deltas_p: list[float] = []
        deltas_r: list[float] = []
        pvals: list[float] = []
        n_pos = 0

        for field_id in range(n_fields):
            test_idx = np.nonzero(fid == field_id)[0]
            train_idx = np.nonzero(fid != field_id)[0]
            Btr = X_B[train_idx]
            Bte = X_B[test_idx]
            Ttr = X_T[train_idx]
            Tte = X_T[test_idx]
            ytr = y[train_idx]
            yte = y[test_idx]

            # Fit B once
            yB_tr_pred, yB_te_pred, wB, muB, sdB, yB_mean = _ridge_fit_predict_multi(Btr, ytr, Bte, ridge_alpha)
            mB = metrics_vec(yte, yB_te_pred)
            rtr = ytr - yB_tr_pred
            # residual model uses topo only
            muT = Ttr.mean(axis=0)
            sdT = Ttr.std(axis=0)
            sdT = np.where(sdT > 0, sdT, 1.0)
            Ttr_s = (Ttr - muT) / sdT
            Tte_s = (Tte - muT) / sdT

            # Fit residual ridge (real)
            r_mean = rtr.mean(axis=0)
            rc = rtr - r_mean
            TT = Ttr_s.T @ Ttr_s
            Tr = Ttr_s.T @ rc
            wR = np.linalg.solve(TT + float(ridge_alpha) * np.eye(TT.shape[0]), Tr)
            r_te_pred = Tte_s @ wR + r_mean
            yC_te_pred = yB_te_pred + r_te_pred
            mC = metrics_vec(yte, yC_te_pred)

            dp = float(mC["pearson_mean"] - mB["pearson_mean"])
            dr = float(mC["relRMSE_mean"] - mB["relRMSE_mean"])
            deltas_p.append(dp)
            deltas_r.append(dr)
            if dp > 0:
                n_pos += 1

            # placebo permutations: permute topo in train and test, refit residual model
            null_dp: list[float] = []
            null_dr: list[float] = []
            for _ in range(P):
                perm_tr = rng_placebo.permutation(train_idx.size)
                perm_te = rng_placebo.permutation(test_idx.size)
                Ttr_p = Ttr_s[perm_tr]
                Tte_p = Tte_s[perm_te]
                Tr_p = Ttr_p.T @ rc
                wR_p = np.linalg.solve(TT + float(ridge_alpha) * np.eye(TT.shape[0]), Tr_p)
                r_te_p = Tte_p @ wR_p + r_mean
                yC_p = yB_te_pred + r_te_p
                mp = metrics_vec(yte, yC_p)
                null_dp.append(float(mp["pearson_mean"] - mB["pearson_mean"]))
                null_dr.append(float(mp["relRMSE_mean"] - mB["relRMSE_mean"]))

            null_dp_a = np.asarray(null_dp, dtype=np.float64)
            null_dr_a = np.asarray(null_dr, dtype=np.float64)
            ndp_mean = float(null_dp_a.mean())
            ndp_std = float(null_dp_a.std(ddof=1)) if null_dp_a.size > 1 else 0.0
            ndr_mean = float(null_dr_a.mean())
            ndr_std = float(null_dr_a.std(ddof=1)) if null_dr_a.size > 1 else 0.0
            z = float((dp - ndp_mean) / ndp_std) if ndp_std > 0 else float("inf")
            # empirical one-sided p for Pearson improvement
            p_emp = float((np.sum(null_dp_a >= dp) + 1) / (P + 1))
            pvals.append(p_emp)

            fold_rows.append(
                {
                    "field_id": int(field_id),
                    "n_test": int(test_idx.size),
                    "pearson_B_mean": float(mB["pearson_mean"]),
                    "relRMSE_B_mean": float(mB["relRMSE_mean"]),
                    "pearson_C_mean": float(mC["pearson_mean"]),
                    "relRMSE_C_mean": float(mC["relRMSE_mean"]),
                    "delta_pearson_mean": dp,
                    "delta_relRMSE_mean": dr,
                    "null_delta_pearson_mean": ndp_mean,
                    "null_delta_pearson_std": ndp_std,
                    "null_delta_relRMSE_mean": ndr_mean,
                    "null_delta_relRMSE_std": ndr_std,
                    "zscore_delta_pearson": z,
                    "p_emp_delta_pearson": p_emp,
                }
            )

        deltas_p = np.asarray(deltas_p, dtype=np.float64)
        deltas_r = np.asarray(deltas_r, dtype=np.float64)
        pvals = np.asarray(pvals, dtype=np.float64)

        eps = 1e-12
        fisher_stat = float(-2.0 * np.sum(np.log(np.clip(pvals, eps, 1.0))))
        fisher_p = float(chi2.sf(fisher_stat, 2 * n_fields))
        verdict = bool((float(deltas_p.mean()) > 0.0) and (float(deltas_r.mean()) < 0.0) and (fisher_p < 0.01) and (n_pos >= 7))

        metrics: dict[str, Any] = {
            "experiment": experiment,
            "exp_name": exp_name,
            "seed": seed,
            "split_seed": split_seed,
            "placebo_seed": placebo_seed,
            "grid_size": grid_size,
            "alpha": alpha,
            "k0_frac": k0_frac,
            "band_split_check": split_checks,
            "n_fields": n_fields,
            "patches_per_field": patches_per_field,
            "patch_size": patch_size,
            "feature_field": feature_field,
            "target_field": target_field,
            "topo_mode": topo_mode,
            "topo_keys": topo_keys,
            "quantiles_b0": quantiles,
            "topo_thresholds_per_field": topo_thr_per_field if topo_mode == "quantile_per_field" else {},
            "thresholds_pos_sigma": thresholds_pos_sigma,
            "thresholds_neg_sigma": thresholds_neg_sigma,
            "ridge_alpha": ridge_alpha,
            "placebo_permutations": P,
            "lofo_folds": fold_rows,
            "lofo_agg": {
                "delta_pearson_mean": float(deltas_p.mean()),
                "delta_pearson_std": float(deltas_p.std(ddof=1)),
                "delta_relRMSE_mean": float(deltas_r.mean()),
                "delta_relRMSE_std": float(deltas_r.std(ddof=1)),
                "fisher_p_delta_pearson": fisher_p,
                "n_pos_delta_pearson": int(n_pos),
                "verdict_pass": verdict,
            },
        }

        header = list(fold_rows[0].keys()) if fold_rows else []
        rows = [[r[h] for h in header] for r in fold_rows]
        write_csv(paths.run_dir / "lofo_by_field.csv", header, rows)
        write_json(paths.metrics_json, metrics)
        return paths

    if experiment in {"e13", "e14"}:
        # Kernel learning: linear model on raw patch pixels to predict y=(gx_high,gy_high) under LOFO.
        from numpy.lib.stride_tricks import sliding_window_view
        from sklearn.linear_model import SGDRegressor
        from sklearn.multioutput import MultiOutputRegressor
        import matplotlib.pyplot as plt

        grid_size = int(cfg.get("grid_size", 128))
        alpha = float(cfg.get("alpha", 2.0))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 20_000))
        if n_fields <= 1:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        ridge_alpha = float(cfg.get("ridge_alpha", 1.0))
        sgd_alpha = float(cfg.get("sgd_alpha", 1e-4))
        sgd_max_iter = int(cfg.get("sgd_max_iter", 10))

        k0_list = [float(cfg.get("k0_frac", 0.15))]
        w_list = [int(patch_size)]
        if experiment == "e14":
            k0_list = [float(x) for x in cfg.get("k0_list", [0.10, 0.15, 0.25])]
            w_list = [int(x) for x in cfg.get("w_list", [9, 17, 33])]

        pixel_field = str(cfg.get("pixel_field", "rho_high")).lower()  # rho|rho_high
        baseline_field = str(cfg.get("baseline_field", pixel_field)).lower()
        placebo_mode = str(cfg.get("placebo_mode", "permute_y_train")).lower()

        def vec_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
            from .models import rmse, safe_pearson

            out = {}
            for i, name in enumerate(["gx", "gy"]):
                yt = y_true[:, i]
                yp = y_pred[:, i]
                out[f"pearson_{name}"] = float(safe_pearson(yt, yp))
                rm = float(rmse(yt, yp))
                std = float(np.std(yt))
                out[f"relRMSE_{name}"] = float(rm / std) if std > 0 else float("nan")
            out["pearson_mean"] = float(np.mean([out["pearson_gx"], out["pearson_gy"]]))
            out["relRMSE_mean"] = float(np.mean([out["relRMSE_gx"], out["relRMSE_gy"]]))
            return out

        def fit_pixel_model(Xtr: np.ndarray, ytr: np.ndarray) -> Any:
            base = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(sgd_alpha),
                max_iter=int(sgd_max_iter),
                tol=1e-3,
                random_state=int(placebo_seed),
            )
            model = MultiOutputRegressor(base)
            model.fit(Xtr, ytr)
            return model

        sweep_rows: list[dict[str, Any]] = []
        kernel_artifacts: dict[str, Any] = {}

        for k0_frac in k0_list:
            for w in w_list:
                ps = _require_odd("window_size", w)
                r = ps // 2

                # generate per-field datasets (kept in memory per config point)
                Xpix_by_field: list[np.ndarray] = []
                Xb_by_field: list[np.ndarray] = []
                y_by_field: list[np.ndarray] = []
                split_checks: list[dict[str, float]] = []

                for field_id in range(n_fields):
                    rng_field = np.random.default_rng(seed + field_id)
                    rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
                    split = band_split_poisson_2d(rho01, k0_frac=float(k0_frac))
                    split_checks.append({"field_id": float(field_id), "rel_err_gx": split.rel_err_gx, "rel_err_gy": split.rel_err_gy})

                    def select(name: str) -> np.ndarray:
                        if name == "rho":
                            return rho01
                        if name == "rho_high":
                            return split.rho_high
                        raise ValueError("field must be rho or rho_high")

                    field_pix = select(pixel_field)
                    field_b = select(baseline_field)

                    # z-score per field (requested)
                    field_pix = _zscore_field(field_pix)
                    field_b = _zscore_field(field_b)

                    target_field = str(cfg.get("target_field", "high")).lower()
                    if target_field == "high":
                        gx = split.high.gx
                        gy = split.high.gy
                    elif target_field == "full":
                        gx = split.full.gx
                        gy = split.full.gy
                    else:
                        raise ValueError("target_field must be high or full")
                    yfield = np.stack([gx, gy], axis=2)

                    if any(s < ps for s in field_pix.shape):
                        raise ValueError("grid too small for window")

                    # sample centers
                    cx = rng.integers(r, grid_size - r, size=patches_per_field, dtype=np.int64)
                    cy = rng.integers(r, grid_size - r, size=patches_per_field, dtype=np.int64)

                    # pixel patches (fast sliding window view)
                    win = sliding_window_view(field_pix, (ps, ps))  # (nx-ps+1, ny-ps+1, ps, ps)
                    patches = win[cx - r, cy - r].reshape(patches_per_field, -1).astype(np.float32, copy=False)

                    # baseline B features on field_b
                    pref1 = _prefix_sum_2d(field_b)
                    pref2 = _prefix_sum_2d(field_b * field_b)
                    x0 = (cx - r).astype(np.int64)
                    x1 = (cx + r + 1).astype(np.int64)
                    y0 = (cy - r).astype(np.int64)
                    y1 = (cy + r + 1).astype(np.int64)
                    mass = _box_sum_2d(pref1, x0, x1, y0, y1)
                    nvox = float(ps * ps)
                    mean = mass / nvox
                    sumsq = _box_sum_2d(pref2, x0, x1, y0, y1)
                    var = np.maximum(0.0, sumsq / nvox - mean * mean)
                    mass2 = mass * mass
                    from scipy import ndimage

                    max_grid = ndimage.maximum_filter(field_b, size=ps, mode="constant", cval=-np.inf)
                    mx = max_grid[cx, cy]
                    gx_f, gy_f = np.gradient(field_b)
                    egrid = gx_f * gx_f + gy_f * gy_f
                    eavg = ndimage.uniform_filter(egrid, size=ps, mode="constant", cval=0.0)
                    ge = eavg[cx, cy]
                    Xb = np.column_stack([mass, mass2, var, mx, ge]).astype(np.float64, copy=False)

                    yv = yfield[cx, cy].reshape(patches_per_field, 2).astype(np.float64, copy=False)

                    Xpix_by_field.append(patches)
                    Xb_by_field.append(Xb)
                    y_by_field.append(yv)

                # LOFO evaluation
                fold_rows: list[dict[str, Any]] = []
                kernels_gx: list[np.ndarray] = []
                kernels_gy: list[np.ndarray] = []
                for test_field in range(n_fields):
                    train_fields = [i for i in range(n_fields) if i != test_field]
                    Xpix_tr = np.concatenate([Xpix_by_field[i] for i in train_fields], axis=0)
                    Xb_tr = np.concatenate([Xb_by_field[i] for i in train_fields], axis=0)
                    y_tr = np.concatenate([y_by_field[i] for i in train_fields], axis=0)

                    Xpix_te = Xpix_by_field[test_field]
                    Xb_te = Xb_by_field[test_field]
                    y_te = y_by_field[test_field]

                    # baseline B (ridge closed-form)
                    _, yB_te_pred, _, _, _, _ = _ridge_fit_predict_multi(Xb_tr, y_tr, Xb_te, ridge_alpha)
                    mB = vec_metrics(y_te, yB_te_pred)

                    # pixel model P
                    modelP = fit_pixel_model(Xpix_tr, y_tr)
                    yP_te_pred = modelP.predict(Xpix_te)
                    mP = vec_metrics(y_te, yP_te_pred)

                    dp = float(mP["pearson_mean"] - mB["pearson_mean"])
                    dr = float(mP["relRMSE_mean"] - mB["relRMSE_mean"])

                    # placebo
                    if placebo_mode == "permute_y_train":
                        perm = rng_placebo.permutation(y_tr.shape[0])
                        y_tr_p = y_tr[perm]
                        modelPl = fit_pixel_model(Xpix_tr, y_tr_p)
                        yPl_pred = modelPl.predict(Xpix_te)
                    elif placebo_mode == "permute_pixels":
                        perm = rng_placebo.permutation(Xpix_tr.shape[1])
                        modelPl = fit_pixel_model(Xpix_tr[:, perm], y_tr)
                        yPl_pred = modelPl.predict(Xpix_te[:, perm])
                    else:
                        raise ValueError("placebo_mode must be permute_y_train or permute_pixels")
                    mPl = vec_metrics(y_te, yPl_pred)
                    dp_pl = float(mPl["pearson_mean"] - mB["pearson_mean"])
                    dr_pl = float(mPl["relRMSE_mean"] - mB["relRMSE_mean"])

                    fold_rows.append(
                        {
                            "field_id": int(test_field),
                            "pearson_B_mean": float(mB["pearson_mean"]),
                            "relRMSE_B_mean": float(mB["relRMSE_mean"]),
                            "pearson_P_mean": float(mP["pearson_mean"]),
                            "relRMSE_P_mean": float(mP["relRMSE_mean"]),
                            "delta_pearson_P_minus_B": dp,
                            "delta_relRMSE_P_minus_B": dr,
                            "delta_pearson_placebo_minus_B": dp_pl,
                            "delta_relRMSE_placebo_minus_B": dr_pl,
                        }
                    )

                    # kernel weights for w=17 fixed experiment (E13 only)
                    if experiment == "e13":
                        coef_gx = np.asarray(modelP.estimators_[0].coef_, dtype=np.float64)
                        coef_gy = np.asarray(modelP.estimators_[1].coef_, dtype=np.float64)
                        kernels_gx.append(coef_gx.reshape(ps, ps))
                        kernels_gy.append(coef_gy.reshape(ps, ps))

                # aggregate per config point
                dp_vals = np.asarray([r["delta_pearson_P_minus_B"] for r in fold_rows], dtype=np.float64)
                dr_vals = np.asarray([r["delta_relRMSE_P_minus_B"] for r in fold_rows], dtype=np.float64)
                dp_pl_vals = np.asarray([r["delta_pearson_placebo_minus_B"] for r in fold_rows], dtype=np.float64)
                dr_pl_vals = np.asarray([r["delta_relRMSE_placebo_minus_B"] for r in fold_rows], dtype=np.float64)
                sweep_rows.append(
                    {
                        "w": int(ps),
                        "k0_frac": float(k0_frac),
                        "pixel_field": pixel_field,
                        "baseline_field": baseline_field,
                        "delta_pearson_mean": float(dp_vals.mean()),
                        "delta_pearson_std": float(dp_vals.std(ddof=1)),
                        "delta_relRMSE_mean": float(dr_vals.mean()),
                        "delta_relRMSE_std": float(dr_vals.std(ddof=1)),
                        "pearson_P_mean": float(np.mean([r["pearson_P_mean"] for r in fold_rows])),
                        "relRMSE_P_mean": float(np.mean([r["relRMSE_P_mean"] for r in fold_rows])),
                        "pearson_B_mean": float(np.mean([r["pearson_B_mean"] for r in fold_rows])),
                        "relRMSE_B_mean": float(np.mean([r["relRMSE_B_mean"] for r in fold_rows])),
                        "delta_pearson_placebo_mean": float(dp_pl_vals.mean()),
                        "delta_relRMSE_placebo_mean": float(dr_pl_vals.mean()),
                        "n_pos_delta_pearson": int(np.sum(dp_vals > 0)),
                    }
                )

                # write per-fold CSV per config point
                tag = f"w{ps}_k0{int(round(k0_frac*100)):02d}"
                header = list(fold_rows[0].keys()) if fold_rows else []
                write_csv(paths.run_dir / f"lofo_by_field_{tag}.csv", header, [[r[h] for h in header] for r in fold_rows])

                if experiment == "e13":
                    kgx = np.mean(np.stack(kernels_gx, axis=0), axis=0)
                    kgy = np.mean(np.stack(kernels_gy, axis=0), axis=0)
                    # symmetry checks
                    def corr(a: np.ndarray, b: np.ndarray) -> float:
                        a = a.reshape(-1)
                        b = b.reshape(-1)
                        a = a - a.mean()
                        b = b - b.mean()
                        denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
                        return float((a @ b) / denom)

                    gx_antisym_x = corr(kgx, -kgx[::-1, :])
                    gy_antisym_y = corr(kgy, -kgy[:, ::-1])

                    for arr, name in [(kgx, "kernel_gx.png"), (kgy, "kernel_gy.png")]:
                        vmax = float(np.max(np.abs(arr))) + 1e-12
                        plt.figure(figsize=(4, 4))
                        plt.imshow(arr, cmap="coolwarm", vmin=-vmax, vmax=vmax)
                        plt.colorbar()
                        plt.title(name.replace(".png", ""))
                        plt.tight_layout()
                        plt.savefig(paths.run_dir / name, dpi=150)
                        plt.close()

                    kernel_artifacts = {
                        "gx_antisym_x_corr": gx_antisym_x,
                        "gy_antisym_y_corr": gy_antisym_y,
                    }

        # write sweep results
        header = list(sweep_rows[0].keys()) if sweep_rows else []
        write_csv(paths.run_dir / "sweep_results.csv", header, [[r[h] for h in header] for r in sweep_rows])

        metrics = {
            "experiment": experiment,
            "exp_name": exp_name,
            "seed": seed,
            "split_seed": split_seed,
            "placebo_seed": placebo_seed,
            "grid_size": grid_size,
            "alpha": alpha,
            "patches_per_field": patches_per_field,
            "n_fields": n_fields,
            "ridge_alpha": ridge_alpha,
            "sgd_alpha": sgd_alpha,
            "sgd_max_iter": sgd_max_iter,
            "placebo_mode": placebo_mode,
            "sweep_rows": sweep_rows,
            "kernel_artifacts": kernel_artifacts,
        }

        # simple heatmap for E14
        if experiment == "e14" and sweep_rows:
            ws = sorted({int(r["w"]) for r in sweep_rows})
            ks = sorted({float(r["k0_frac"]) for r in sweep_rows})
            pear = np.full((len(ws), len(ks)), np.nan)
            rel = np.full((len(ws), len(ks)), np.nan)
            for r in sweep_rows:
                i = ws.index(int(r["w"]))
                j = ks.index(float(r["k0_frac"]))
                pear[i, j] = float(r["pearson_P_mean"])
                rel[i, j] = float(r["relRMSE_P_mean"])

            def plot_grid(mat: np.ndarray, title: str, fname: str) -> None:
                plt.figure(figsize=(6, 3))
                plt.imshow(mat, aspect="auto", cmap="viridis")
                plt.xticks(range(len(ks)), [f"{k:.2f}" for k in ks])
                plt.yticks(range(len(ws)), [str(w) for w in ws])
                plt.xlabel("k0_frac")
                plt.ylabel("w")
                plt.title(title)
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(paths.run_dir / fname, dpi=150)
                plt.close()

            plot_grid(pear, "mean Pearson agg (P)", "heatmap_pearson.png")
            plot_grid(rel, "mean relRMSE agg (P)", "heatmap_relRMSE.png")

        write_json(paths.metrics_json, metrics)
        return paths

    if experiment == "e15":
        # Compare learned linear kernel to theoretical impulse-response kernel for bandpassed operator.
        import matplotlib.pyplot as plt
        from numpy.lib.stride_tricks import sliding_window_view
        from sklearn.linear_model import SGDRegressor
        from sklearn.multioutput import MultiOutputRegressor

        grid_size = int(cfg.get("grid_size", 128))
        alpha = float(cfg.get("alpha", 2.0))
        k0_frac = float(cfg.get("k0_frac", 0.15))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 20_000))
        ps = _require_odd("window_size", int(cfg.get("window_size", patch_size)))
        r = ps // 2
        pixel_field = str(cfg.get("pixel_field", "rho_high")).lower()  # rho|rho_high
        sgd_alpha = float(cfg.get("sgd_alpha", 1e-4))
        sgd_max_iter = int(cfg.get("sgd_max_iter", 10))

        # theoretical kernel via delta impulse in rho (within [0,1])
        rho_delta = np.zeros((grid_size, grid_size), dtype=np.float64)
        c = grid_size // 2
        rho_delta[c, c] = 1.0
        split_th = band_split_poisson_2d(rho_delta, k0_frac=k0_frac)
        rho_high_th = split_th.rho_high
        gx_high_th = split_th.high.gx
        gy_high_th = split_th.high.gy
        # kernel coeff at offset = response at -offset => flip both axes
        Kth_gx = gx_high_th[c - r : c + r + 1, c - r : c + r + 1][::-1, ::-1]
        Kth_gy = gy_high_th[c - r : c + r + 1, c - r : c + r + 1][::-1, ::-1]

        # train per-fold and compare
        per_fold: list[dict[str, Any]] = []
        learned_gx: list[np.ndarray] = []
        learned_gy: list[np.ndarray] = []

        def corr(a: np.ndarray, b: np.ndarray) -> float:
            a = a.reshape(-1).astype(np.float64)
            b = b.reshape(-1).astype(np.float64)
            a = a - a.mean()
            b = b - b.mean()
            denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
            return float((a @ b) / denom)

        # per-field cached data
        X_by_field: list[np.ndarray] = []
        y_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
            split = band_split_poisson_2d(rho01, k0_frac=k0_frac)
            if pixel_field == "rho":
                field = rho01
            elif pixel_field == "rho_high":
                field = split.rho_high
            else:
                raise ValueError("pixel_field must be rho or rho_high")

            win = sliding_window_view(field, (ps, ps))
            cx = rng.integers(r, grid_size - r, size=patches_per_field, dtype=np.int64)
            cy = rng.integers(r, grid_size - r, size=patches_per_field, dtype=np.int64)
            X = win[cx - r, cy - r].reshape(patches_per_field, -1).astype(np.float32, copy=False)
            y = np.column_stack([split.high.gx[cx, cy], split.high.gy[cx, cy]]).astype(np.float64, copy=False)
            X_by_field.append(X)
            y_by_field.append(y)

        for test_field in range(n_fields):
            train_fields = [i for i in range(n_fields) if i != test_field]
            Xtr = np.concatenate([X_by_field[i] for i in train_fields], axis=0)
            ytr = np.concatenate([y_by_field[i] for i in train_fields], axis=0)
            Xte = X_by_field[test_field]
            yte = y_by_field[test_field]

            base = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(sgd_alpha),
                max_iter=int(sgd_max_iter),
                tol=1e-3,
                random_state=int(placebo_seed),
            )
            model = MultiOutputRegressor(base)
            model.fit(Xtr, ytr)

            coef_gx = np.asarray(model.estimators_[0].coef_, dtype=np.float64).reshape(ps, ps)
            coef_gy = np.asarray(model.estimators_[1].coef_, dtype=np.float64).reshape(ps, ps)
            learned_gx.append(coef_gx)
            learned_gy.append(coef_gy)

            corr_gx = corr(coef_gx, Kth_gx)
            corr_gy = corr(coef_gy, Kth_gy)
            rel_l2_gx = float(np.linalg.norm(coef_gx - Kth_gx) / (np.linalg.norm(Kth_gx) + 1e-12))
            rel_l2_gy = float(np.linalg.norm(coef_gy - Kth_gy) / (np.linalg.norm(Kth_gy) + 1e-12))

            gx_antisym_x = corr(coef_gx, -coef_gx[::-1, :])
            gy_antisym_y = corr(coef_gy, -coef_gy[:, ::-1])

            per_fold.append(
                {
                    "field_id": int(test_field),
                    "corr_gx": corr_gx,
                    "corr_gy": corr_gy,
                    "relL2_gx": rel_l2_gx,
                    "relL2_gy": rel_l2_gy,
                    "gx_antisym_x_corr": gx_antisym_x,
                    "gy_antisym_y_corr": gy_antisym_y,
                }
            )

        Kgx = np.mean(np.stack(learned_gx, axis=0), axis=0)
        Kgy = np.mean(np.stack(learned_gy, axis=0), axis=0)

        # Save weights
        np.save(paths.run_dir / "kernel_theoretical_gx.npy", Kth_gx)
        np.save(paths.run_dir / "kernel_theoretical_gy.npy", Kth_gy)
        np.save(paths.run_dir / "kernel_learned_gx.npy", Kgx)
        np.save(paths.run_dir / "kernel_learned_gy.npy", Kgy)

        def save_triplet(th: np.ndarray, le: np.ndarray, name: str) -> None:
            vmax = float(np.max(np.abs(th))) + 1e-12
            plt.figure(figsize=(9, 3))
            plt.subplot(1, 3, 1)
            plt.imshow(th, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            plt.title("theory")
            plt.colorbar(fraction=0.046)
            plt.subplot(1, 3, 2)
            plt.imshow(le, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            plt.title("learned")
            plt.colorbar(fraction=0.046)
            plt.subplot(1, 3, 3)
            plt.imshow(le - th, cmap="coolwarm")
            plt.title("diff")
            plt.colorbar(fraction=0.046)
            plt.tight_layout()
            plt.savefig(paths.run_dir / name, dpi=150)
            plt.close()

        save_triplet(Kth_gx, Kgx, "kernel_compare_gx.png")
        save_triplet(Kth_gy, Kgy, "kernel_compare_gy.png")

        header = list(per_fold[0].keys()) if per_fold else []
        write_csv(paths.run_dir / "kernel_compare_by_field.csv", header, [[r[h] for h in header] for r in per_fold])

        metrics = {
            "experiment": experiment,
            "exp_name": exp_name,
            "seed": seed,
            "grid_size": grid_size,
            "alpha": alpha,
            "k0_frac": k0_frac,
            "window_size": ps,
            "pixel_field": pixel_field,
            "n_fields": n_fields,
            "patches_per_field": patches_per_field,
            "sgd_alpha": sgd_alpha,
            "sgd_max_iter": sgd_max_iter,
            "per_field": per_fold,
            "agg": {
                "corr_gx_mean": float(np.mean([r["corr_gx"] for r in per_fold])),
                "corr_gy_mean": float(np.mean([r["corr_gy"] for r in per_fold])),
                "relL2_gx_mean": float(np.mean([r["relL2_gx"] for r in per_fold])),
                "relL2_gy_mean": float(np.mean([r["relL2_gy"] for r in per_fold])),
            },
        }
        write_json(paths.metrics_json, metrics)
        return paths

    if experiment == "e16":
        # LOFO pixels-only on full gx/gy (no bandpass target) to probe non-locality.
        from numpy.lib.stride_tricks import sliding_window_view
        import matplotlib.pyplot as plt
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        grid_size = int(cfg.get("grid_size", 128))
        alpha = float(cfg.get("alpha", 2.0))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 5_000))
        if n_fields <= 1:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")

        w_list = [int(x) for x in cfg.get("w_list", [17, 33])]
        k0_frac = float(cfg.get("k0_frac", 0.15))  # used only to compute split/full consistently
        pixel_field = str(cfg.get("pixel_field", "rho")).lower()
        ridge_alpha = float(cfg.get("ridge_alpha", 1.0))
        ridge_max_iter = int(cfg.get("ridge_max_iter", 500))

        from .models import rmse, safe_pearson

        def vec_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
            pears = []
            rels = []
            for i in range(2):
                yt = y_true[:, i]
                yp = y_pred[:, i]
                pears.append(float(safe_pearson(yt, yp)))
                rm = float(rmse(yt, yp))
                std = float(np.std(yt))
                rels.append(float(rm / std) if std > 0 else float("nan"))
            return float(np.mean(pears)), float(np.mean(rels))

        rows: list[dict[str, Any]] = []
        for w in w_list:
            ps = _require_odd("window_size", w)
            r = ps // 2
            X_by_field: list[np.ndarray] = []
            y_by_field: list[np.ndarray] = []
            for field_id in range(n_fields):
                rng_field = np.random.default_rng(seed + field_id)
                rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
                split = band_split_poisson_2d(rho01, k0_frac=k0_frac)
                field = rho01 if pixel_field == "rho" else split.rho_high
                field = _zscore_field(field)
                win = sliding_window_view(field, (ps, ps))
                cx = rng.integers(r, grid_size - r, size=patches_per_field, dtype=np.int64)
                cy = rng.integers(r, grid_size - r, size=patches_per_field, dtype=np.int64)
                X = win[cx - r, cy - r].reshape(patches_per_field, -1).astype(np.float32, copy=False)
                y = np.column_stack([split.full.gx[cx, cy], split.full.gy[cx, cy]]).astype(np.float64, copy=False)
                X_by_field.append(X)
                y_by_field.append(y)

            fold_pears = []
            fold_rels = []
            for test_field in range(n_fields):
                train_fields = [i for i in range(n_fields) if i != test_field]
                Xtr = np.concatenate([X_by_field[i] for i in train_fields], axis=0)
                ytr = np.concatenate([y_by_field[i] for i in train_fields], axis=0)
                Xte = X_by_field[test_field]
                yte = y_by_field[test_field]

                scaler = StandardScaler(with_mean=True, with_std=True)
                Xtr_s = scaler.fit_transform(Xtr)
                Xte_s = scaler.transform(Xte)
                model = Ridge(alpha=float(ridge_alpha), solver="sag", max_iter=int(ridge_max_iter), random_state=int(placebo_seed))
                model.fit(Xtr_s, ytr)
                ypred = model.predict(Xte_s)
                p, rrm = vec_metrics(yte, ypred)
                fold_pears.append(p)
                fold_rels.append(rrm)

            rows.append(
                {
                    "w": int(ps),
                    "k0_frac": float(k0_frac),
                    "pixel_field": pixel_field,
                    "pearson_mean": float(np.mean(fold_pears)),
                    "pearson_std": float(np.std(fold_pears, ddof=1)),
                    "relRMSE_mean": float(np.mean(fold_rels)),
                    "relRMSE_std": float(np.std(fold_rels, ddof=1)),
                }
            )

        header = list(rows[0].keys()) if rows else []
        write_csv(paths.run_dir / "fullg_sweep.csv", header, [[r[h] for h in header] for r in rows])

        # simple plot
        ws = [r["w"] for r in rows]
        pears = [r["pearson_mean"] for r in rows]
        rels = [r["relRMSE_mean"] for r in rows]
        plt.figure(figsize=(8, 3))
        plt.subplot(1, 2, 1)
        plt.plot(ws, pears, marker="o")
        plt.xlabel("w")
        plt.ylabel("mean Pearson")
        plt.title("full gx/gy (LOFO)")
        plt.grid(True, alpha=0.3)
        plt.subplot(1, 2, 2)
        plt.plot(ws, rels, marker="o")
        plt.xlabel("w")
        plt.ylabel("mean relRMSE")
        plt.title("full gx/gy (LOFO)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(paths.run_dir / "fullg_locality.png", dpi=150)
        plt.close()

        metrics = {
            "experiment": experiment,
            "exp_name": exp_name,
            "seed": seed,
            "grid_size": grid_size,
            "alpha": alpha,
            "n_fields": n_fields,
            "patches_per_field": patches_per_field,
            "k0_frac": k0_frac,
            "pixel_field": pixel_field,
            "ridge_alpha": ridge_alpha,
            "ridge_max_iter": ridge_max_iter,
            "rows": rows,
        }
        write_json(paths.metrics_json, metrics)
        return paths

    if experiment == "e15b":
        # Kernel diagnostics: scale/shift correction + ridge_alpha sweep vs theory.
        from sklearn.linear_model import SGDRegressor

        run_dirs = [Path(p) for p in cfg.get("kernel_run_dirs", [])]
        if not run_dirs:
            raise ValueError("kernel_run_dirs must list e15 run folders to compare")

        def corr(a: np.ndarray, b: np.ndarray) -> float:
            a = a.reshape(-1).astype(np.float64)
            b = b.reshape(-1).astype(np.float64)
            a = a - a.mean()
            b = b - b.mean()
            denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
            return float((a @ b) / denom)

        def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))

        def scale_best(klearn: np.ndarray, kth: np.ndarray) -> tuple[float, np.ndarray]:
            num = float(np.sum(klearn * kth))
            den = float(np.sum(klearn * klearn)) + 1e-12
            a = num / den
            return a, a * klearn

        def shift_zero(x: np.ndarray, dx: int, dy: int) -> np.ndarray:
            out = np.zeros_like(x)
            xs = slice(max(0, dx), x.shape[0] + min(0, dx))
            ys = slice(max(0, dy), x.shape[1] + min(0, dy))
            xo = slice(max(0, -dx), x.shape[0] + min(0, -dx))
            yo = slice(max(0, -dy), x.shape[1] + min(0, -dy))
            out[xs, ys] = x[xo, yo]
            return out

        diag_rows: list[dict[str, Any]] = []
        for rd in run_dirs:
            kth_gx = np.load(rd / "kernel_theoretical_gx.npy")
            kth_gy = np.load(rd / "kernel_theoretical_gy.npy")
            kl_gx = np.load(rd / "kernel_learned_gx.npy")
            kl_gy = np.load(rd / "kernel_learned_gy.npy")

            a_gx, kl_gx_s = scale_best(kl_gx, kth_gx)
            a_gy, kl_gy_s = scale_best(kl_gy, kth_gy)

            best = {"corr": -1e9, "rel": 1e9, "dx": 0, "dy": 0}
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    x = shift_zero(kl_gx, dx, dy)
                    y = shift_zero(kl_gy, dx, dy)
                    ax, xs = scale_best(x, kth_gx)
                    ay, ys = scale_best(y, kth_gy)
                    c = 0.5 * (corr(xs, kth_gx) + corr(ys, kth_gy))
                    r = 0.5 * (rel_l2(xs, kth_gx) + rel_l2(ys, kth_gy))
                    if c > best["corr"]:
                        best = {"corr": c, "rel": r, "dx": dx, "dy": dy}

            diag_rows.append(
                {
                    "run_dir": str(rd),
                    "w": int(kth_gx.shape[0]),
                    "corr_gx": corr(kl_gx, kth_gx),
                    "corr_gy": corr(kl_gy, kth_gy),
                    "relL2_gx": rel_l2(kl_gx, kth_gx),
                    "relL2_gy": rel_l2(kl_gy, kth_gy),
                    "scale_a_gx": float(a_gx),
                    "scale_a_gy": float(a_gy),
                    "relL2_scaled_gx": rel_l2(kl_gx_s, kth_gx),
                    "relL2_scaled_gy": rel_l2(kl_gy_s, kth_gy),
                    "bestshift_dx": int(best["dx"]),
                    "bestshift_dy": int(best["dy"]),
                    "corr_bestshift_mean": float(best["corr"]),
                    "relL2_scaled_bestshift_mean": float(best["rel"]),
                }
            )

        # Ridge sweep vs theory (w=33)
        sweep_w = int(cfg.get("sweep_w", 33))
        sweep_ps = _require_odd("sweep_w", sweep_w)
        grid_size = int(cfg.get("grid_size", 128))
        alpha = float(cfg.get("alpha", 2.0))
        k0_frac = float(cfg.get("k0_frac", 0.15))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 20_000))
        ridge_alphas = [float(x) for x in cfg.get("ridge_alphas", [1e-6, 1e-4, 1e-2, 1.0, 1e2])]
        sgd_epochs = int(cfg.get("sgd_epochs", 3))
        batch_size = int(cfg.get("batch_size", 4096))
        eta0 = float(cfg.get("sgd_eta0", 0.01))
        power_t = float(cfg.get("sgd_power_t", 0.25))

        # theory kernel for sweep_w
        rho_delta = np.zeros((grid_size, grid_size), dtype=np.float64)
        c = grid_size // 2
        r = sweep_ps // 2
        rho_delta[c, c] = 1.0
        split_th = band_split_poisson_2d(rho_delta, k0_frac=k0_frac)
        kth_gx = split_th.high.gx[c - r : c + r + 1, c - r : c + r + 1][::-1, ::-1]
        kth_gy = split_th.high.gy[c - r : c + r + 1, c - r : c + r + 1][::-1, ::-1]

        from numpy.lib.stride_tricks import sliding_window_view

        # Train all ridge_alphas in one streamed pass over data (avoids repeating FFT work per alpha).
        models: dict[float, tuple[SGDRegressor, SGDRegressor]] = {}
        for i, ra in enumerate(ridge_alphas):
            # SGDRegressor is a scalable ridge-like learner (L2 penalty).
            # Note: `alpha` here is SGD's L2 strength; it's not numerically identical to `Ridge(alpha=...)`,
            # but it is suitable for diagnosing shrinkage vs theory without materializing X_all.
            reg_gx = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(ra),
                learning_rate="invscaling",
                eta0=float(eta0),
                power_t=float(power_t),
                max_iter=1,
                tol=None,
                fit_intercept=True,
                random_state=int(placebo_seed) + 10_000 * i + 0,
            )
            reg_gy = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(ra),
                learning_rate="invscaling",
                eta0=float(eta0),
                power_t=float(power_t),
                max_iter=1,
                tol=None,
                fit_intercept=True,
                random_state=int(placebo_seed) + 10_000 * i + 1,
            )
            models[float(ra)] = (reg_gx, reg_gy)

        for epoch in range(sgd_epochs):
            for field_id in range(n_fields):
                rng_field = np.random.default_rng(seed + field_id)
                rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
                split = band_split_poisson_2d(rho01, k0_frac=k0_frac)
                field = split.rho_high
                win = sliding_window_view(field, (sweep_ps, sweep_ps))
                cx = rng.integers(r, grid_size - r, size=patches_per_field, dtype=np.int64)
                cy = rng.integers(r, grid_size - r, size=patches_per_field, dtype=np.int64)
                X = win[cx - r, cy - r].reshape(patches_per_field, -1).astype(np.float32, copy=False)
                y = np.column_stack([split.high.gx[cx, cy], split.high.gy[cx, cy]]).astype(np.float64, copy=False)

                perm = rng_placebo.permutation(patches_per_field)
                for start in range(0, patches_per_field, batch_size):
                    idx = perm[start : start + batch_size]
                    Xb = X[idx]
                    ygx = y[idx, 0]
                    ygy = y[idx, 1]
                    for reg_gx, reg_gy in models.values():
                        reg_gx.partial_fit(Xb, ygx)
                        reg_gy.partial_fit(Xb, ygy)

                del X, y, win, cx, cy, split, field, rho01

        sweep_rows: list[dict[str, Any]] = []
        for ra in ridge_alphas:
            reg_gx, reg_gy = models[float(ra)]
            kgx = np.asarray(reg_gx.coef_, dtype=np.float64).reshape(sweep_ps, sweep_ps)
            kgy = np.asarray(reg_gy.coef_, dtype=np.float64).reshape(sweep_ps, sweep_ps)
            ax, kgx_s = scale_best(kgx, kth_gx)
            ay, kgy_s = scale_best(kgy, kth_gy)

            best = {"corr": -1e9, "rel": 1e9, "dx": 0, "dy": 0}
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    x = shift_zero(kgx, dx, dy)
                    y_ = shift_zero(kgy, dx, dy)
                    _, xs = scale_best(x, kth_gx)
                    _, ys = scale_best(y_, kth_gy)
                    cmean = 0.5 * (corr(xs, kth_gx) + corr(ys, kth_gy))
                    rmean = 0.5 * (rel_l2(xs, kth_gx) + rel_l2(ys, kth_gy))
                    if cmean > best["corr"]:
                        best = {"corr": cmean, "rel": rmean, "dx": dx, "dy": dy}

            sweep_rows.append(
                {
                    "ridge_alpha": float(ra),
                    "sgd_epochs": int(sgd_epochs),
                    "batch_size": int(batch_size),
                    "sgd_eta0": float(eta0),
                    "sgd_power_t": float(power_t),
                    "corr_gx": corr(kgx, kth_gx),
                    "corr_gy": corr(kgy, kth_gy),
                    "relL2_gx": rel_l2(kgx, kth_gx),
                    "relL2_gy": rel_l2(kgy, kth_gy),
                    "relL2_scaled_gx": rel_l2(kgx_s, kth_gx),
                    "relL2_scaled_gy": rel_l2(kgy_s, kth_gy),
                    "corr_bestshift_mean": float(best["corr"]),
                    "relL2_scaled_bestshift_mean": float(best["rel"]),
                    "bestshift_dx": int(best["dx"]),
                    "bestshift_dy": int(best["dy"]),
                    "scale_a_gx": float(ax),
                    "scale_a_gy": float(ay),
                }
            )

        write_csv(paths.run_dir / "kernel_diag.csv", list(diag_rows[0].keys()), [[r[k] for k in diag_rows[0].keys()] for r in diag_rows])
        write_csv(paths.run_dir / "ridge_sweep.csv", list(sweep_rows[0].keys()), [[r[k] for k in sweep_rows[0].keys()] for r in sweep_rows])

        def fmt(x: float) -> str:
            if not np.isfinite(x):
                return "nan"
            ax = abs(float(x))
            if ax != 0.0 and (ax < 1e-3 or ax >= 1e3):
                return f"{x:.3e}"
            return f"{x:.4f}"

        def md_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r in rows:
                out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
            return "\n".join(out)

        diag_md_rows: list[dict[str, str]] = []
        for r0 in diag_rows:
            diag_md_rows.append(
                {
                    "w": str(r0["w"]),
                    "corr_mean": fmt(0.5 * (float(r0["corr_gx"]) + float(r0["corr_gy"]))),
                    "relL2_mean": fmt(0.5 * (float(r0["relL2_gx"]) + float(r0["relL2_gy"]))),
                    "relL2_scaled_mean": fmt(0.5 * (float(r0["relL2_scaled_gx"]) + float(r0["relL2_scaled_gy"]))),
                    "bestshift_(dx,dy)": f"({int(r0['bestshift_dx'])},{int(r0['bestshift_dy'])})",
                    "corr_bestshift_mean": fmt(float(r0["corr_bestshift_mean"])),
                    "relL2_scaled_bestshift_mean": fmt(float(r0["relL2_scaled_bestshift_mean"])),
                }
            )

        sweep_md_rows: list[dict[str, str]] = []
        for r0 in sweep_rows:
            sweep_md_rows.append(
                {
                    "ridge_alpha": fmt(float(r0["ridge_alpha"])),
                    "corr_mean": fmt(0.5 * (float(r0["corr_gx"]) + float(r0["corr_gy"]))),
                    "corr_bestshift_mean": fmt(float(r0["corr_bestshift_mean"])),
                    "relL2_scaled_mean": fmt(0.5 * (float(r0["relL2_scaled_gx"]) + float(r0["relL2_scaled_gy"]))),
                    "bestshift_(dx,dy)": f"({int(r0['bestshift_dx'])},{int(r0['bestshift_dy'])})",
                }
            )

        summary_md = (
            f"# Kernel diagnostics (E15b)\n\n"
            f"- run: `{paths.run_dir}`\n"
            f"- source kernels: {', '.join(f'`{p}`' for p in run_dirs)}\n\n"
            f"## Direct compare (loaded .npy)\n"
            f"{md_table(diag_md_rows, ['w','corr_mean','relL2_mean','relL2_scaled_mean','bestshift_(dx,dy)','corr_bestshift_mean','relL2_scaled_bestshift_mean'])}\n\n"
            f"## Ridge-alpha sweep (SGD L2)\n"
            f"{md_table(sweep_md_rows, ['ridge_alpha','corr_mean','corr_bestshift_mean','relL2_scaled_mean','bestshift_(dx,dy)'])}\n"
        )
        (paths.run_dir / "summary_kernel_diag.md").write_text(summary_md, encoding="utf-8")
        (paths.run_dir.parent.parent / "summary_kernel_diag.md").write_text(summary_md, encoding="utf-8")

        metrics = {
            "experiment": experiment,
            "exp_name": exp_name,
            "diag": diag_rows,
            "ridge_sweep": sweep_rows,
        }
        write_json(paths.metrics_json, metrics)
        return paths

    if experiment == "e15c":
        # Kernel "ground-truth" via impulse response through the exact discrete pipeline, then compare to learned/theory.
        import matplotlib.pyplot as plt

        kernel_run_dirs = [Path(p) for p in cfg.get("kernel_run_dirs", [])]
        if not kernel_run_dirs:
            raise ValueError("kernel_run_dirs must list e15 run folders containing kernel_{learned,theoretical}_*.npy")

        grid_size = int(cfg.get("grid_size", 256))
        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        k0_frac = float(cfg.get("k0_frac", 0.15))
        ws = [int(x) for x in cfg.get("ws", [17, 33])]
        modes = [str(x).lower() for x in cfg.get("modes", ["rho_high"])]  # rho|rho_high

        # Load kernels from provided run dirs (keyed by w).
        learned_by_w: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        theory_by_w: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for rd in kernel_run_dirs:
            kth_gx = np.load(rd / "kernel_theoretical_gx.npy")
            kth_gy = np.load(rd / "kernel_theoretical_gy.npy")
            kl_gx = np.load(rd / "kernel_learned_gx.npy")
            kl_gy = np.load(rd / "kernel_learned_gy.npy")
            w = int(kth_gx.shape[0])
            learned_by_w[w] = (kl_gx.astype(np.float64), kl_gy.astype(np.float64))
            theory_by_w[w] = (kth_gx.astype(np.float64), kth_gy.astype(np.float64))

        def corr(a: np.ndarray, b: np.ndarray) -> float:
            a = a.reshape(-1).astype(np.float64)
            b = b.reshape(-1).astype(np.float64)
            a = a - a.mean()
            b = b - b.mean()
            denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
            return float((a @ b) / denom)

        def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))

        def scale_best(klearn: np.ndarray, kth: np.ndarray) -> tuple[float, np.ndarray]:
            num = float(np.sum(klearn * kth))
            den = float(np.sum(klearn * klearn)) + 1e-12
            a = num / den
            return a, a * klearn

        def extract_periodic_kernel(field: np.ndarray, w: int) -> np.ndarray:
            # field is the impulse response evaluated on the grid, with the impulse at (0,0).
            # A linear, translation-invariant operator satisfies:
            #   y(center) = sum_{dx,dy} K(-dx,-dy) * x(center+dx,center+dy)
            # so we reverse the impulse patch to match the regression dot-product convention.
            ps = _require_odd("w", w)
            r = ps // 2
            idx = (np.arange(-r, r + 1, dtype=np.int64) % field.shape[0]).astype(np.int64)
            patch = field[np.ix_(idx, idx)]
            return patch[::-1, ::-1].astype(np.float64, copy=False)

        def save_kernel_png(arr: np.ndarray, name: str) -> None:
            vmax = float(np.max(np.abs(arr))) + 1e-12
            plt.figure(figsize=(4, 4))
            plt.imshow(arr, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            plt.colorbar()
            plt.title(name.replace(".png", ""))
            plt.tight_layout()
            plt.savefig(paths.run_dir / name, dpi=150)
            plt.close()

        def save_triplet(a: np.ndarray, b: np.ndarray, name: str, title_a: str, title_b: str) -> None:
            vmax = float(np.max(np.abs(a))) + 1e-12
            plt.figure(figsize=(9, 3))
            plt.subplot(1, 3, 1)
            plt.imshow(a, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            plt.title(title_a)
            plt.colorbar(fraction=0.046)
            plt.subplot(1, 3, 2)
            plt.imshow(b, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            plt.title(title_b)
            plt.colorbar(fraction=0.046)
            plt.subplot(1, 3, 3)
            plt.imshow(b - a, cmap="coolwarm")
            plt.title("diff")
            plt.colorbar(fraction=0.046)
            plt.tight_layout()
            plt.savefig(paths.run_dir / name, dpi=150)
            plt.close()

        # Build impulse (delta at origin) and run the exact same discrete pipeline.
        rho_delta = np.zeros((grid_size, grid_size), dtype=np.float64)
        rho_delta[0, 0] = 1.0

        impulse_by_mode: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for mode in modes:
            if mode == "rho":
                sol = solve_poisson_periodic_fft_2d(rho_delta)
                impulse_by_mode[mode] = (sol.gx, sol.gy)
            elif mode == "rho_high":
                split = band_split_poisson_2d(rho_delta, k0_frac=k0_frac)
                impulse_by_mode[mode] = (split.high.gx, split.high.gy)
            else:
                raise ValueError("modes must be a list of: rho, rho_high")

        results: list[dict[str, Any]] = []
        rows_md: list[dict[str, str]] = []

        for mode, (gx_imp, gy_imp) in impulse_by_mode.items():
            for w in ws:
                kx_imp = extract_periodic_kernel(gx_imp, w)
                ky_imp = extract_periodic_kernel(gy_imp, w)

                np.save(paths.run_dir / f"kernel_impulse_{mode}_w{w}_gx.npy", kx_imp)
                np.save(paths.run_dir / f"kernel_impulse_{mode}_w{w}_gy.npy", ky_imp)

                save_kernel_png(kx_imp, f"kernel_impulse_{mode}_w{w}_gx.png")
                save_kernel_png(ky_imp, f"kernel_impulse_{mode}_w{w}_gy.png")

                def compare_pair(
                    tag: str, kx: np.ndarray, ky: np.ndarray
                ) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
                    ax, kx_s = scale_best(kx, kx_imp)
                    ay, ky_s = scale_best(ky, ky_imp)
                    metrics = {
                        "corr_gx": corr(kx, kx_imp),
                        "corr_gy": corr(ky, ky_imp),
                        "relL2_gx": rel_l2(kx, kx_imp),
                        "relL2_gy": rel_l2(ky, ky_imp),
                        "scale_a_gx": float(ax),
                        "scale_a_gy": float(ay),
                        "relL2_scaled_gx": rel_l2(kx_s, kx_imp),
                        "relL2_scaled_gy": rel_l2(ky_s, ky_imp),
                        "corr_mean": 0.5 * (corr(kx, kx_imp) + corr(ky, ky_imp)),
                        "relL2_scaled_mean": 0.5 * (rel_l2(kx_s, kx_imp) + rel_l2(ky_s, ky_imp)),
                    }
                    np.save(paths.run_dir / f"kernel_{tag}_{mode}_w{w}_gx.npy", kx.astype(np.float64, copy=False))
                    np.save(paths.run_dir / f"kernel_{tag}_{mode}_w{w}_gy.npy", ky.astype(np.float64, copy=False))
                    return metrics, kx_s, ky_s

                # learned vs impulse
                learn_metrics: dict[str, float] | None = None
                if w in learned_by_w:
                    klx, kly = learned_by_w[w]
                    learn_metrics, _, _ = compare_pair("learned", klx, kly)
                    save_triplet(
                        kx_imp,
                        klx,
                        f"kernel_compare_learn_vs_impulse_{mode}_w{w}_gx.png",
                        "impulse",
                        "learned",
                    )
                    save_triplet(
                        ky_imp,
                        kly,
                        f"kernel_compare_learn_vs_impulse_{mode}_w{w}_gy.png",
                        "impulse",
                        "learned",
                    )

                # theory vs impulse
                theory_metrics: dict[str, float] | None = None
                if w in theory_by_w:
                    kthx, kthy = theory_by_w[w]
                    theory_metrics, _, _ = compare_pair("theory", kthx, kthy)
                    save_triplet(
                        kx_imp,
                        kthx,
                        f"kernel_compare_theory_vs_impulse_{mode}_w{w}_gx.png",
                        "impulse",
                        "theory",
                    )
                    save_triplet(
                        ky_imp,
                        kthy,
                        f"kernel_compare_theory_vs_impulse_{mode}_w{w}_gy.png",
                        "impulse",
                        "theory",
                    )

                row = {
                    "mode": mode,
                    "w": int(w),
                    "learn_corr_mean": float("nan") if learn_metrics is None else float(learn_metrics["corr_mean"]),
                    "learn_relL2_scaled_mean": float("nan") if learn_metrics is None else float(learn_metrics["relL2_scaled_mean"]),
                    "theory_corr_mean": float("nan") if theory_metrics is None else float(theory_metrics["corr_mean"]),
                    "theory_relL2_scaled_mean": float("nan") if theory_metrics is None else float(theory_metrics["relL2_scaled_mean"]),
                    "learn_details": learn_metrics,
                    "theory_details": theory_metrics,
                }
                results.append(row)

                def fmt(x: float) -> str:
                    if not np.isfinite(x):
                        return "nan"
                    ax = abs(float(x))
                    if ax != 0.0 and (ax < 1e-3 or ax >= 1e3):
                        return f"{x:.3e}"
                    return f"{x:.4f}"

                rows_md.append(
                    {
                        "mode": mode,
                        "w": str(w),
                        "corr_mean(learn,imp)": "nan" if learn_metrics is None else fmt(float(learn_metrics["corr_mean"])),
                        "relL2_scaled_mean(learn,imp)": "nan" if learn_metrics is None else fmt(float(learn_metrics["relL2_scaled_mean"])),
                        "corr_mean(theory,imp)": "nan" if theory_metrics is None else fmt(float(theory_metrics["corr_mean"])),
                        "relL2_scaled_mean(theory,imp)": "nan" if theory_metrics is None else fmt(float(theory_metrics["relL2_scaled_mean"])),
                    }
                )

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r in rows:
                out.append("| " + " | ".join(r.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        summary_md = (
            "# Impulse-response kernel comparison (E15c)\n\n"
            f"- run: `{paths.run_dir}`\n"
            f"- grid_size: `{grid_size}`\n"
            f"- k0_frac: `{k0_frac}`\n"
            f"- kernel sources: {', '.join(f'`{p}`' for p in kernel_run_dirs)}\n\n"
            "## Summary table\n"
            + md_table(
                rows_md,
                [
                    "mode",
                    "w",
                    "corr_mean(learn,imp)",
                    "relL2_scaled_mean(learn,imp)",
                    "corr_mean(theory,imp)",
                    "relL2_scaled_mean(theory,imp)",
                ],
            )
            + "\n"
        )
        (paths.run_dir / "summary_kernel_impulse.md").write_text(summary_md, encoding="utf-8")

        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "grid_size": grid_size,
                "k0_frac": k0_frac,
                "modes": modes,
                "ws": ws,
                "kernel_run_dirs": [str(p) for p in kernel_run_dirs],
                "results": results,
            },
        )
        return paths

    if experiment == "e17":
        # LOFO moments vs baseline B vs pixels, for gx/gy targets (high or full).
        from numpy.lib.stride_tricks import sliding_window_view
        from sklearn.linear_model import SGDRegressor, Ridge
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.preprocessing import StandardScaler

        grid_size = int(cfg.get("grid_size", 128))
        alpha = float(cfg.get("alpha", 2.0))
        k0_frac = float(cfg.get("k0_frac", 0.15))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 20_000))
        if n_fields <= 1:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")

        feature_field = str(cfg.get("feature_field", "rho_high")).lower()  # rho|rho_high
        target_field = str(cfg.get("target_field", "high")).lower()  # high|full
        include_pixels = bool(cfg.get("include_pixels", True))
        pixel_solver = str(cfg.get("pixel_solver", "sgd")).lower()  # sgd|ridge
        ridge_alpha_m = float(cfg.get("ridge_alpha", 1.0))
        sgd_alpha = float(cfg.get("sgd_alpha", 1e-4))
        sgd_max_iter = int(cfg.get("sgd_max_iter", 10))
        pixel_epochs = int(cfg.get("pixel_epochs", 3))
        pixel_batch_size = int(cfg.get("pixel_batch_size", 4096))
        pixel_eta0 = float(cfg.get("pixel_eta0", 0.01))
        pixel_power_t = float(cfg.get("pixel_power_t", 0.25))
        ridge_max_iter = int(cfg.get("ridge_max_iter", 2000))

        ps = _require_odd("window_size", int(cfg.get("window_size", patch_size)))
        r = ps // 2

        # moment weights
        xs = np.arange(-r, r + 1, dtype=np.float64)
        dx, dy = np.meshgrid(xs, xs, indexing="ij")
        ones = np.ones_like(dx)
        dxx = dx * dx
        dyy = dy * dy
        dxy = dx * dy
        w1 = np.exp(-(dx * dx + dy * dy) / (2.0 * (ps / 6.0) ** 2))
        w2 = np.exp(-(dx * dx + dy * dy) / (2.0 * (ps / 3.0) ** 2))

        def moment_block(patches: np.ndarray, weight: np.ndarray) -> dict[str, np.ndarray]:
            W = weight.reshape(-1)
            X = patches.astype(np.float64, copy=False)
            M0 = X @ (ones.reshape(-1) * W)
            Mx = X @ (dx.reshape(-1) * W)
            My = X @ (dy.reshape(-1) * W)
            Mxx = X @ (dxx.reshape(-1) * W)
            Myy = X @ (dyy.reshape(-1) * W)
            Mxy = X @ (dxy.reshape(-1) * W)
            eps = 1e-12
            inv = 1.0 / (M0 + eps)
            xbar = Mx * inv
            ybar = My * inv
            cov_xx = Mxx * inv - xbar * xbar
            cov_yy = Myy * inv - ybar * ybar
            cov_xy = Mxy * inv - xbar * ybar
            tr = cov_xx + cov_yy + eps
            ellip = (cov_xx - cov_yy) / tr
            theta2 = np.arctan2(2.0 * cov_xy, cov_xx - cov_yy + eps)
            cos2 = np.cos(theta2)
            sin2 = np.sin(theta2)
            return {
                "m0": M0,
                "xbar": xbar,
                "ybar": ybar,
                "dipx": Mx,
                "dipy": My,
                "qxx": Mxx,
                "qyy": Myy,
                "qxy": Mxy,
                "cov_xx": cov_xx,
                "cov_yy": cov_yy,
                "cov_xy": cov_xy,
                "ellip": ellip,
                "cos2": cos2,
                "sin2": sin2,
            }

        def build_M(patches: np.ndarray) -> np.ndarray:
            mA = moment_block(patches, ones)
            mB = moment_block(patches, w1)
            mC = moment_block(patches, w2)
            keys = ["m0", "xbar", "ybar", "dipx", "dipy", "cov_xx", "cov_yy", "cov_xy", "ellip", "cos2", "sin2"]
            cols = [mA[k] for k in keys] + [mB[k] for k in keys] + [mC[k] for k in keys]
            return np.column_stack(cols).astype(np.float64, copy=False)

        from scipy import ndimage

        def build_B(field: np.ndarray, cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
            pref1 = _prefix_sum_2d(field)
            pref2 = _prefix_sum_2d(field * field)
            x0 = (cx - r).astype(np.int64)
            x1 = (cx + r + 1).astype(np.int64)
            y0 = (cy - r).astype(np.int64)
            y1 = (cy + r + 1).astype(np.int64)
            mass = _box_sum_2d(pref1, x0, x1, y0, y1)
            nvox = float(ps * ps)
            mean = mass / nvox
            sumsq = _box_sum_2d(pref2, x0, x1, y0, y1)
            var = np.maximum(0.0, sumsq / nvox - mean * mean)
            mass2 = mass * mass
            max_grid = ndimage.maximum_filter(field, size=ps, mode="constant", cval=-np.inf)
            mx = max_grid[cx, cy]
            gx_f, gy_f = np.gradient(field)
            egrid = gx_f * gx_f + gy_f * gy_f
            eavg = ndimage.uniform_filter(egrid, size=ps, mode="constant", cval=0.0)
            ge = eavg[cx, cy]
            return np.column_stack([mass, mass2, var, mx, ge]).astype(np.float64, copy=False)

        # build per-field datasets
        Xb_by_field: list[np.ndarray] = []
        Xm_by_field: list[np.ndarray] = []
        Xp_by_field: list[np.ndarray] = []
        y_by_field: list[np.ndarray] = []

        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
            split = band_split_poisson_2d(rho01, k0_frac=k0_frac)

            if feature_field == "rho":
                field = rho01
            elif feature_field == "rho_high":
                field = split.rho_high
            else:
                raise ValueError("feature_field must be rho or rho_high")

            if target_field == "high":
                gx = split.high.gx
                gy = split.high.gy
            elif target_field == "full":
                gx = split.full.gx
                gy = split.full.gy
            else:
                raise ValueError("target_field must be high or full")

            win = sliding_window_view(field, (ps, ps))
            cx = rng.integers(r, grid_size - r, size=patches_per_field, dtype=np.int64)
            cy = rng.integers(r, grid_size - r, size=patches_per_field, dtype=np.int64)
            patches = win[cx - r, cy - r].reshape(patches_per_field, -1).astype(np.float32, copy=False)

            Xb = build_B(field, cx, cy)
            Xm = build_M(patches)
            y = np.column_stack([gx[cx, cy], gy[cx, cy]]).astype(np.float64, copy=False)

            Xb_by_field.append(Xb)
            Xm_by_field.append(Xm)
            if include_pixels:
                Xp_by_field.append(patches)
            y_by_field.append(y)

        def vec_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
            from .models import rmse, safe_pearson

            out = {}
            for i, name in enumerate(["gx", "gy"]):
                yt = y_true[:, i]
                yp = y_pred[:, i]
                out[f"pearson_{name}"] = float(safe_pearson(yt, yp))
                rm = float(rmse(yt, yp))
                std = float(np.std(yt))
                out[f"relRMSE_{name}"] = float(rm / std) if std > 0 else float("nan")
            out["pearson_mean"] = float(np.mean([out["pearson_gx"], out["pearson_gy"]]))
            out["relRMSE_mean"] = float(np.mean([out["relRMSE_gx"], out["relRMSE_gy"]]))
            return out

        def fit_predict_small(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
            _, y_pred, _, _, _, _ = _ridge_fit_predict_multi(Xtr, ytr, Xte, ridge_alpha_m)
            return y_pred

        def fit_predict_pixels(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
            if pixel_solver == "sgd":
                base = SGDRegressor(
                    loss="squared_error",
                    penalty="l2",
                    alpha=float(sgd_alpha),
                    max_iter=int(sgd_max_iter),
                    tol=1e-3,
                    random_state=int(placebo_seed),
                )
                model = MultiOutputRegressor(base)
                model.fit(Xtr, ytr)
                return model.predict(Xte)
            if pixel_solver == "ridge":
                scaler = StandardScaler(with_mean=True, with_std=True)
                Xtr_s = scaler.fit_transform(Xtr)
                Xte_s = scaler.transform(Xte)
                model = Ridge(alpha=float(ridge_alpha_m), solver="sag", max_iter=int(ridge_max_iter), random_state=int(placebo_seed))
                model.fit(Xtr_s, ytr)
                return model.predict(Xte_s)
            raise ValueError("pixel_solver must be sgd or ridge")

        def fit_predict_pixels_stream(train_fields: list[int], test_field: int) -> np.ndarray:
            if pixel_solver != "sgd":
                raise ValueError("fit_predict_pixels_stream requires pixel_solver=sgd")
            reg_gx = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(sgd_alpha),
                learning_rate="invscaling",
                eta0=float(pixel_eta0),
                power_t=float(pixel_power_t),
                max_iter=1,
                tol=None,
                fit_intercept=True,
                random_state=int(placebo_seed) + 0,
            )
            reg_gy = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(sgd_alpha),
                learning_rate="invscaling",
                eta0=float(pixel_eta0),
                power_t=float(pixel_power_t),
                max_iter=1,
                tol=None,
                fit_intercept=True,
                random_state=int(placebo_seed) + 1,
            )
            for epoch in range(pixel_epochs):
                for fid in train_fields:
                    X = Xp_by_field[fid]
                    y = y_by_field[fid]
                    perm = rng_placebo.permutation(X.shape[0])
                    for start in range(0, X.shape[0], pixel_batch_size):
                        idx = perm[start : start + pixel_batch_size]
                        Xb = X[idx]
                        reg_gx.partial_fit(Xb, y[idx, 0])
                        reg_gy.partial_fit(Xb, y[idx, 1])
            Xte = Xp_by_field[test_field]
            y_pred = np.empty((Xte.shape[0], 2), dtype=np.float64)
            for start in range(0, Xte.shape[0], pixel_batch_size):
                sl = slice(start, min(Xte.shape[0], start + pixel_batch_size))
                y_pred[sl, 0] = reg_gx.predict(Xte[sl])
                y_pred[sl, 1] = reg_gy.predict(Xte[sl])
            return y_pred

        fold_rows: list[dict[str, Any]] = []
        for test_field in range(n_fields):
            train_fields = [i for i in range(n_fields) if i != test_field]
            ytr = np.concatenate([y_by_field[i] for i in train_fields], axis=0)
            yte = y_by_field[test_field]

            Btr = np.concatenate([Xb_by_field[i] for i in train_fields], axis=0)
            Bte = Xb_by_field[test_field]
            Mtr = np.concatenate([Xm_by_field[i] for i in train_fields], axis=0)
            Mte = Xm_by_field[test_field]
            BMtr = np.concatenate([Btr, Mtr], axis=1)
            BMte = np.concatenate([Bte, Mte], axis=1)

            yB = fit_predict_small(Btr, ytr, Bte)
            yM = fit_predict_small(Mtr, ytr, Mte)
            yBM = fit_predict_small(BMtr, ytr, BMte)

            mB = vec_metrics(yte, yB)
            mM = vec_metrics(yte, yM)
            mBM = vec_metrics(yte, yBM)

            row: dict[str, Any] = {
                "field_id": int(test_field),
                "pearson_B": float(mB["pearson_mean"]),
                "relRMSE_B": float(mB["relRMSE_mean"]),
                "pearson_M": float(mM["pearson_mean"]),
                "relRMSE_M": float(mM["relRMSE_mean"]),
                "pearson_BM": float(mBM["pearson_mean"]),
                "relRMSE_BM": float(mBM["relRMSE_mean"]),
                "deltaP_M_minus_B": float(mM["pearson_mean"] - mB["pearson_mean"]),
                "deltaR_M_minus_B": float(mM["relRMSE_mean"] - mB["relRMSE_mean"]),
                "deltaP_BM_minus_B": float(mBM["pearson_mean"] - mB["pearson_mean"]),
                "deltaR_BM_minus_B": float(mBM["relRMSE_mean"] - mB["relRMSE_mean"]),
            }

            if include_pixels:
                if pixel_solver == "sgd":
                    yP = fit_predict_pixels_stream(train_fields, test_field)
                else:
                    Ptr = np.concatenate([Xp_by_field[i] for i in train_fields], axis=0)
                    Pte = Xp_by_field[test_field]
                    yP = fit_predict_pixels(Ptr, ytr, Pte)
                mP = vec_metrics(yte, yP)
                row.update(
                    {
                        "pearson_P": float(mP["pearson_mean"]),
                        "relRMSE_P": float(mP["relRMSE_mean"]),
                        "deltaP_P_minus_B": float(mP["pearson_mean"] - mB["pearson_mean"]),
                        "deltaR_P_minus_B": float(mP["relRMSE_mean"] - mB["relRMSE_mean"]),
                    }
                )
                fold_rows.append(row)

        header = list(fold_rows[0].keys()) if fold_rows else []
        write_csv(paths.run_dir / "lofo_by_field.csv", header, [[r[h] for h in header] for r in fold_rows])
        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "grid_size": grid_size,
                "alpha": alpha,
                "k0_frac": k0_frac,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "window_size": ps,
                "feature_field": feature_field,
                "target_field": target_field,
                "include_pixels": include_pixels,
                "pixel_solver": pixel_solver,
                "ridge_alpha": ridge_alpha_m,
                "sgd_alpha": sgd_alpha,
                "sgd_max_iter": sgd_max_iter,
                "pixel_epochs": pixel_epochs,
                "pixel_batch_size": pixel_batch_size,
                "pixel_eta0": pixel_eta0,
                "pixel_power_t": pixel_power_t,
                "lofo_folds": fold_rows,
            },
        )
        return paths

    if experiment == "e18":
        # Identify the discrete kernel via cross-spectra (Wiener filter) and compare to impulse/theory kernels.
        import matplotlib.pyplot as plt

        grid_size = int(cfg.get("grid_size", 256))
        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        alpha = float(cfg.get("alpha", 2.0))
        k0_frac = float(cfg.get("k0_frac", 0.15))
        n_fields = int(cfg.get("n_fields", 10))
        if n_fields <= 0:
            raise ValueError("n_fields must be > 0")

        ws = [int(x) for x in cfg.get("ws", [17, 33])]
        mode = str(cfg.get("mode", "rho_high")).lower()
        if mode != "rho_high":
            raise ValueError("E18 currently supports mode=rho_high only (to match E15c rho_high)")

        impulse_run_dir = Path(str(cfg.get("impulse_run_dir", "")))
        if not impulse_run_dir.exists():
            raise FileNotFoundError(f"impulse_run_dir not found: {impulse_run_dir}")
        theory_run_dirs = [Path(p) for p in cfg.get("theory_run_dirs", [])]
        if not theory_run_dirs:
            raise ValueError("theory_run_dirs must list e15 run folders containing kernel_theoretical_*.npy")

        # Load impulse kernels (already extracted in regression convention) from E15c.
        impulse_by_w: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for w in ws:
            gx_path = impulse_run_dir / f"kernel_impulse_rho_high_w{w}_gx.npy"
            gy_path = impulse_run_dir / f"kernel_impulse_rho_high_w{w}_gy.npy"
            if not gx_path.exists() or not gy_path.exists():
                raise FileNotFoundError(f"Impulse kernel missing for w={w} in {impulse_run_dir}")
            impulse_by_w[w] = (np.load(gx_path).astype(np.float64), np.load(gy_path).astype(np.float64))

        # Load theory kernels (regression convention) from E15 run dirs (keyed by w).
        theory_by_w: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for rd in theory_run_dirs:
            kth_gx = np.load(rd / "kernel_theoretical_gx.npy").astype(np.float64)
            kth_gy = np.load(rd / "kernel_theoretical_gy.npy").astype(np.float64)
            w = int(kth_gx.shape[0])
            theory_by_w[w] = (kth_gx, kth_gy)

        def corr(a: np.ndarray, b: np.ndarray) -> float:
            a = a.reshape(-1).astype(np.float64)
            b = b.reshape(-1).astype(np.float64)
            a = a - a.mean()
            b = b - b.mean()
            denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
            return float((a @ b) / denom)

        def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))

        def scale_best(klearn: np.ndarray, kth: np.ndarray) -> tuple[float, np.ndarray]:
            num = float(np.sum(klearn * kth))
            den = float(np.sum(klearn * klearn)) + 1e-12
            a = num / den
            return a, a * klearn

        def extract_periodic_kernel(field: np.ndarray, w: int) -> np.ndarray:
            ps = _require_odd("w", w)
            r = ps // 2
            idx = (np.arange(-r, r + 1, dtype=np.int64) % field.shape[0]).astype(np.int64)
            patch = field[np.ix_(idx, idx)]
            return patch[::-1, ::-1].astype(np.float64, copy=False)

        def save_kernel_png(arr: np.ndarray, name: str) -> None:
            vmax = float(np.max(np.abs(arr))) + 1e-12
            plt.figure(figsize=(4, 4))
            plt.imshow(arr, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            plt.colorbar()
            plt.title(name.replace(".png", ""))
            plt.tight_layout()
            plt.savefig(paths.run_dir / name, dpi=150)
            plt.close()

        def save_triplet(a: np.ndarray, b: np.ndarray, name: str, title_a: str, title_b: str) -> None:
            vmax = float(np.max(np.abs(a))) + 1e-12
            plt.figure(figsize=(9, 3))
            plt.subplot(1, 3, 1)
            plt.imshow(a, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            plt.title(title_a)
            plt.colorbar(fraction=0.046)
            plt.subplot(1, 3, 2)
            plt.imshow(b, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            plt.title(title_b)
            plt.colorbar(fraction=0.046)
            plt.subplot(1, 3, 3)
            plt.imshow(b - a, cmap="coolwarm")
            plt.title("diff")
            plt.colorbar(fraction=0.046)
            plt.tight_layout()
            plt.savefig(paths.run_dir / name, dpi=150)
            plt.close()

        # Wiener identification in Fourier space: H = E[G*conj(R)] / (E[|R|^2] + eps).
        num_x = np.zeros((grid_size, grid_size), dtype=np.complex128)
        num_y = np.zeros((grid_size, grid_size), dtype=np.complex128)
        den = np.zeros((grid_size, grid_size), dtype=np.float64)
        den_means: list[float] = []

        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
            split = band_split_poisson_2d(rho01, k0_frac=k0_frac)

            rho_high = split.rho_high
            gx_high = split.high.gx
            gy_high = split.high.gy
            assert_finite("rho_high", rho_high)
            assert_finite("gx_high", gx_high)
            assert_finite("gy_high", gy_high)

            R = np.fft.fftn(rho_high)
            Gx = np.fft.fftn(gx_high)
            Gy = np.fft.fftn(gy_high)

            p = np.abs(R) ** 2
            den_means.append(float(np.mean(p)))
            den += p
            num_x += Gx * np.conj(R)
            num_y += Gy * np.conj(R)

        num_x /= float(n_fields)
        num_y /= float(n_fields)
        den /= float(n_fields)

        med = float(np.median(np.asarray(den_means, dtype=np.float64))) if den_means else 0.0
        eps = float(cfg.get("eps", 1e-12 * med if med > 0 else 1e-12))
        Hx = num_x / (den + eps)
        Hy = num_y / (den + eps)

        Kx_full = np.fft.ifftn(Hx).real
        Ky_full = np.fft.ifftn(Hy).real
        assert_finite("Kx_full", Kx_full)
        assert_finite("Ky_full", Ky_full)

        results: list[dict[str, Any]] = []
        md_rows: list[dict[str, str]] = []

        def fmt(x: float) -> str:
            if not np.isfinite(x):
                return "nan"
            ax = abs(float(x))
            if ax != 0.0 and (ax < 1e-3 or ax >= 1e3):
                return f"{x:.3e}"
            return f"{x:.4f}"

        # Store full-grid kernels too (for inspection).
        np.save(paths.run_dir / "kernel_wiener_gx_full.npy", Kx_full.astype(np.float64, copy=False))
        np.save(paths.run_dir / "kernel_wiener_gy_full.npy", Ky_full.astype(np.float64, copy=False))

        for w in ws:
            Kx = extract_periodic_kernel(Kx_full, w)
            Ky = extract_periodic_kernel(Ky_full, w)
            np.save(paths.run_dir / f"kernel_wiener_rho_high_w{w}_gx.npy", Kx)
            np.save(paths.run_dir / f"kernel_wiener_rho_high_w{w}_gy.npy", Ky)
            save_kernel_png(Kx, f"kernel_wiener_rho_high_w{w}_gx.png")
            save_kernel_png(Ky, f"kernel_wiener_rho_high_w{w}_gy.png")

            # Also write generic filenames for w=max(ws) for convenience.
            if w == max(ws):
                save_kernel_png(Kx, "kernel_wiener_gx.png")
                save_kernel_png(Ky, "kernel_wiener_gy.png")

            imp_gx, imp_gy = impulse_by_w[w]
            th_gx, th_gy = theory_by_w[w]

            # Save copies (requested: arrays for wiener/impulse/theory in the run dir).
            np.save(paths.run_dir / f"kernel_impulse_rho_high_w{w}_gx.npy", imp_gx)
            np.save(paths.run_dir / f"kernel_impulse_rho_high_w{w}_gy.npy", imp_gy)
            np.save(paths.run_dir / f"kernel_theory_rho_high_w{w}_gx.npy", th_gx)
            np.save(paths.run_dir / f"kernel_theory_rho_high_w{w}_gy.npy", th_gy)

            # Compare Wiener vs impulse
            ax_imp, Kx_s_imp = scale_best(Kx, imp_gx)
            ay_imp, Ky_s_imp = scale_best(Ky, imp_gy)
            corr_imp_mean = 0.5 * (corr(Kx, imp_gx) + corr(Ky, imp_gy))
            rel_imp_scaled_mean = 0.5 * (rel_l2(Kx_s_imp, imp_gx) + rel_l2(Ky_s_imp, imp_gy))

            save_triplet(imp_gx, Kx, f"kernel_compare_wiener_vs_impulse_rho_high_w{w}_gx.png", "impulse", "wiener")
            save_triplet(imp_gy, Ky, f"kernel_compare_wiener_vs_impulse_rho_high_w{w}_gy.png", "impulse", "wiener")

            # Compare Wiener vs theory
            ax_th, Kx_s_th = scale_best(Kx, th_gx)
            ay_th, Ky_s_th = scale_best(Ky, th_gy)
            corr_th_mean = 0.5 * (corr(Kx, th_gx) + corr(Ky, th_gy))
            rel_th_scaled_mean = 0.5 * (rel_l2(Kx_s_th, th_gx) + rel_l2(Ky_s_th, th_gy))

            save_triplet(th_gx, Kx, f"kernel_compare_wiener_vs_theory_rho_high_w{w}_gx.png", "theory", "wiener")
            save_triplet(th_gy, Ky, f"kernel_compare_wiener_vs_theory_rho_high_w{w}_gy.png", "theory", "wiener")

            results.append(
                {
                    "w": int(w),
                    "corr_mean_wiener_impulse": float(corr_imp_mean),
                    "relL2_scaled_mean_wiener_impulse": float(rel_imp_scaled_mean),
                    "corr_mean_wiener_theory": float(corr_th_mean),
                    "relL2_scaled_mean_wiener_theory": float(rel_th_scaled_mean),
                    "scale_a_imp_gx": float(ax_imp),
                    "scale_a_imp_gy": float(ay_imp),
                    "scale_a_th_gx": float(ax_th),
                    "scale_a_th_gy": float(ay_th),
                }
            )
            md_rows.append(
                {
                    "w": str(int(w)),
                    "corr_mean(wiener,imp)": fmt(float(corr_imp_mean)),
                    "relL2_scaled_mean(wiener,imp)": fmt(float(rel_imp_scaled_mean)),
                    "corr_mean(wiener,theory)": fmt(float(corr_th_mean)),
                    "relL2_scaled_mean(wiener,theory)": fmt(float(rel_th_scaled_mean)),
                }
            )

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r in rows:
                out.append("| " + " | ".join(r.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        summary_md = (
            "# Wiener-identified kernel (E18)\n\n"
            f"- run: `{paths.run_dir}`\n"
            f"- grid_size: `{grid_size}`  alpha: `{alpha}`  k0_frac: `{k0_frac}`  n_fields: `{n_fields}`\n"
            f"- impulse_run_dir: `{impulse_run_dir}`\n"
            f"- theory_run_dirs: {', '.join(f'`{p}`' for p in theory_run_dirs)}\n\n"
            "## Table\n"
            + md_table(
                md_rows,
                [
                    "w",
                    "corr_mean(wiener,imp)",
                    "relL2_scaled_mean(wiener,imp)",
                    "corr_mean(wiener,theory)",
                    "relL2_scaled_mean(wiener,theory)",
                ],
            )
            + "\n"
        )
        (paths.run_dir / "summary_e18_wiener_kernel.md").write_text(summary_md, encoding="utf-8")

        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "grid_size": grid_size,
                "alpha": alpha,
                "k0_frac": k0_frac,
                "n_fields": n_fields,
                "mode": mode,
                "ws": ws,
                "eps": eps,
                "impulse_run_dir": str(impulse_run_dir),
                "theory_run_dirs": [str(p) for p in theory_run_dirs],
                "results": results,
            },
        )
        return paths

    if experiment == "e19":
        # Kernel sufficiency test: baseline = truncated impulse kernel convolution, then predict residual with topo in LOFO.
        import math
        import matplotlib.pyplot as plt
        from scipy import stats

        grid_size = int(cfg.get("grid_size", 256))
        alpha = float(cfg.get("alpha", 2.0))
        k0_frac = float(cfg.get("k0_frac", 0.15))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 20_000))
        ws = [int(x) for x in cfg.get("ws", [17, 33])]
        ridge_alpha = float(cfg.get("ridge_alpha", 1.0))
        topo_mode = str(cfg.get("topo_mode", "quantile_per_field")).lower()  # quantile_per_field only here
        quantiles_b0 = [float(x) for x in cfg.get("quantiles_b0", [0.6, 0.7, 0.8, 0.9])]
        n_perms = int(cfg.get("n_perms", 200))
        shuffle_test = bool(cfg.get("placebo_shuffle_test", False))

        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if not (0.0 < k0_frac < 0.5):
            raise ValueError("k0_frac must be in (0,0.5)")
        if n_fields < 2:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        if topo_mode != "quantile_per_field":
            raise ValueError("E19 currently supports topo_mode=quantile_per_field only")
        if any((q <= 0.0) or (q >= 1.0) for q in quantiles_b0):
            raise ValueError("quantiles_b0 must be in (0,1)")
        if n_perms <= 0:
            raise ValueError("n_perms must be > 0")

        impulse_run_dir = Path(str(cfg.get("impulse_run_dir", "")))
        if not impulse_run_dir.exists():
            raise FileNotFoundError(f"impulse_run_dir not found: {impulse_run_dir}")

        # Helpers
        def vec_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
            from .models import rmse, safe_pearson

            out: dict[str, float] = {}
            for i, name in enumerate(["gx", "gy"]):
                yt = np.asarray(y_true[:, i], dtype=np.float64)
                yp = np.asarray(y_pred[:, i], dtype=np.float64)
                out[f"pearson_{name}"] = float(safe_pearson(yt, yp))
                r = float(rmse(yt, yp))
                s = float(np.std(yt))
                out[f"relRMSE_{name}"] = float(r / s) if s > 0 else float("nan")
            out["pearson_mean"] = float(np.mean([out["pearson_gx"], out["pearson_gy"]]))
            out["relRMSE_mean"] = float(np.mean([out["relRMSE_gx"], out["relRMSE_gy"]]))
            return out

        def corr(a: np.ndarray, b: np.ndarray) -> float:
            a = a.reshape(-1).astype(np.float64)
            b = b.reshape(-1).astype(np.float64)
            a = a - a.mean()
            b = b - b.mean()
            denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
            return float((a @ b) / denom)

        def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))

        def scale_best(klearn: np.ndarray, kth: np.ndarray) -> tuple[float, np.ndarray]:
            num = float(np.sum(klearn * kth))
            den = float(np.sum(klearn * klearn)) + 1e-12
            a = num / den
            return a, a * klearn

        def fisher_p(ps: list[float]) -> float:
            # Fisher method with clamping to avoid log(0)
            if not ps:
                return float("nan")
            eps = 1.0 / float(n_perms + 1)
            ps2 = [min(1.0, max(float(p), eps)) for p in ps]
            stat = -2.0 * float(np.sum(np.log(ps2)))
            return float(stats.chi2.sf(stat, 2 * len(ps2)))

        def embed_kernel_patch(h_patch: np.ndarray) -> np.ndarray:
            # h_patch indexed by dx,dy in [-r..r] at [i=r+dx,j=r+dy], placed at wrapped indices.
            w = int(h_patch.shape[0])
            r = w // 2
            out = np.zeros((grid_size, grid_size), dtype=np.float64)
            for i in range(w):
                dx = i - r
                xi = dx % grid_size
                for j in range(w):
                    dy = j - r
                    yj = dy % grid_size
                    out[xi, yj] = float(h_patch[i, j])
            return out

        def fmt_mu_sd(vals: list[float], digits: int = 4) -> str:
            if not vals:
                return "nan"
            v = np.asarray(vals, dtype=np.float64)
            mu = float(np.mean(v))
            sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            return f"{mu:.{digits}f} ± {sd:.{digits}f}"

        def fmt(x: float) -> str:
            if not np.isfinite(x):
                return "nan"
            ax = abs(float(x))
            if ax != 0.0 and (ax < 1e-3 or ax >= 1e3):
                return f"{x:.3e}"
            return f"{x:.4f}"

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r in rows:
                out.append("| " + " | ".join(r.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        # Precompute kernels for each w from impulse patches (stored in regression convention); convert to convolution h via flip.
        kernels: dict[int, dict[str, Any]] = {}
        for w in ws:
            k_reg_gx = np.load(impulse_run_dir / f"kernel_impulse_rho_high_w{w}_gx.npy").astype(np.float64)
            k_reg_gy = np.load(impulse_run_dir / f"kernel_impulse_rho_high_w{w}_gy.npy").astype(np.float64)
            h_gx = k_reg_gx[::-1, ::-1]
            h_gy = k_reg_gy[::-1, ::-1]
            hx_full = embed_kernel_patch(h_gx)
            hy_full = embed_kernel_patch(h_gy)
            Hx = np.fft.fftn(hx_full)
            Hy = np.fft.fftn(hy_full)
            kernels[w] = {
                "k_reg_gx": k_reg_gx,
                "k_reg_gy": k_reg_gy,
                "h_gx": h_gx,
                "h_gy": h_gy,
                "hx_full": hx_full,
                "hy_full": hy_full,
                "Hx": Hx,
                "Hy": Hy,
            }

        # Cache per-field data that is shared across w (rho_high, g_true high).
        rho_high_by_field: list[np.ndarray] = []
        gx_true_by_field: list[np.ndarray] = []
        gy_true_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
            split = band_split_poisson_2d(rho01, k0_frac=k0_frac)
            rho_high_by_field.append(np.asarray(split.rho_high, dtype=np.float64))
            gx_true_by_field.append(np.asarray(split.high.gx, dtype=np.float64))
            gy_true_by_field.append(np.asarray(split.high.gy, dtype=np.float64))

        baseline_rows: list[dict[str, Any]] = []
        residual_summary_by_w: dict[int, dict[str, Any]] = {}

        # Work per window size.
        for w in ws:
            ps = _require_odd("w", w)
            # Baseline kernel prediction (truncated impulse kernel, via FFT convolution)
            pearsons: list[float] = []
            rels: list[float] = []

            # Per-field sampled datasets for LOFO residual modeling.
            B_by_field: list[np.ndarray] = []
            T_by_field: list[np.ndarray] = []
            yres_by_field: list[np.ndarray] = []

            # Identify topo names from the sampler output; keep stable ordering.
            topo_names: list[str] | None = None

            for field_id in range(n_fields):
                rho_high = rho_high_by_field[field_id]
                gx_true = gx_true_by_field[field_id]
                gy_true = gy_true_by_field[field_id]

                # Baseline prediction on full grid.
                R = np.fft.fftn(rho_high)
                gx_pred = np.fft.ifftn(R * kernels[w]["Hx"]).real
                gy_pred = np.fft.ifftn(R * kernels[w]["Hy"]).real
                assert_finite("gx_pred", gx_pred)
                assert_finite("gy_pred", gy_pred)

                # Baseline metrics on the full grid (all pixels).
                m_full = vec_metrics(
                    np.column_stack([gx_true.reshape(-1), gy_true.reshape(-1)]),
                    np.column_stack([gx_pred.reshape(-1), gy_pred.reshape(-1)]),
                )
                baseline_rows.append(
                    {
                        "w": int(w),
                        "field_id": int(field_id),
                        "pearson_mean": float(m_full["pearson_mean"]),
                        "relRMSE_mean": float(m_full["relRMSE_mean"]),
                    }
                )
                pearsons.append(float(m_full["pearson_mean"]))
                rels.append(float(m_full["relRMSE_mean"]))

                # Residual field.
                gx_res = gx_true - gx_pred
                gy_res = gy_true - gy_pred

                # Topo thresholds per field using quantiles on rho_high itself (supports negative values).
                thr = [float(np.quantile(rho_high.reshape(-1), q)) for q in quantiles_b0]

                feats, y_resid, _centers = _sample_features_2d_fast_topo(
                    field=rho_high,
                    gx=gx_res,
                    gy=gy_res,
                    n_patches=patches_per_field,
                    patch_size=ps,
                    topo_mode=topo_mode,
                    topo_thresholds=thr,
                    thresholds_pos_sigma=None,
                    thresholds_neg_sigma=None,
                    rng=rng,
                )
                y_resid = np.asarray(y_resid, dtype=np.float64)
                assert y_resid.shape == (patches_per_field, 2)
                assert_finite("y_resid", y_resid)

                B_names = ["mass", "mass2", "var", "max", "grad_energy"]
                for n in B_names:
                    if n not in feats:
                        raise RuntimeError(f"Missing B feature: {n}")
                B = np.column_stack([np.asarray(feats[n], dtype=np.float64) for n in B_names])

                # topo keys are those starting with b0_
                tnames = sorted([k for k in feats.keys() if k.startswith("b0_")])
                if topo_names is None:
                    topo_names = tnames
                elif topo_names != tnames:
                    raise RuntimeError(f"Topo feature names mismatch across fields: {topo_names} vs {tnames}")
                T = np.column_stack([np.asarray(feats[n], dtype=np.float64) for n in topo_names])

                B_by_field.append(B)
                T_by_field.append(T)
                yres_by_field.append(y_resid)

            if topo_names is None:
                raise RuntimeError("No topo features found")

            # LOFO residual modeling: compare B vs C=B+T, with topo placebo permutations.
            fold_rows: list[dict[str, Any]] = []
            pvals_pearson: list[float] = []
            pvals_rel: list[float] = []

            for test_field in range(n_fields):
                train_fields = [i for i in range(n_fields) if i != test_field]
                Btr = np.concatenate([B_by_field[i] for i in train_fields], axis=0)
                Ttr = np.concatenate([T_by_field[i] for i in train_fields], axis=0)
                ytr = np.concatenate([yres_by_field[i] for i in train_fields], axis=0)

                Bte = B_by_field[test_field]
                Tte = T_by_field[test_field]
                yte = yres_by_field[test_field]

                # Standardize using train stats (perm invariant per column).
                B_mu = Btr.mean(axis=0)
                B_sd = np.where(Btr.std(axis=0) > 0, Btr.std(axis=0), 1.0)
                T_mu = Ttr.mean(axis=0)
                T_sd = np.where(Ttr.std(axis=0) > 0, Ttr.std(axis=0), 1.0)

                Btr_s = (Btr - B_mu) / B_sd
                Bte_s = (Bte - B_mu) / B_sd
                Ttr_s = (Ttr - T_mu) / T_sd
                Tte_s = (Tte - T_mu) / T_sd

                y_mean = ytr.mean(axis=0)
                yc = ytr - y_mean

                # Fit B
                XtX_B = Btr_s.T @ Btr_s
                Xty_B = Btr_s.T @ yc
                wB = np.linalg.solve(XtX_B + ridge_alpha * np.eye(XtX_B.shape[0]), Xty_B)
                yB = Bte_s @ wB + y_mean
                mB = vec_metrics(yte, yB)

                # Fit C (real)
                XtrC = np.concatenate([Btr_s, Ttr_s], axis=1)
                XteC = np.concatenate([Bte_s, Tte_s], axis=1)
                XtX_C = XtrC.T @ XtrC
                Xty_C = XtrC.T @ yc
                wC = np.linalg.solve(XtX_C + ridge_alpha * np.eye(XtX_C.shape[0]), Xty_C)
                yC = XteC @ wC + y_mean
                mC = vec_metrics(yte, yC)

                deltaP_real = float(mC["pearson_mean"] - mB["pearson_mean"])
                deltaR_real = float(mC["relRMSE_mean"] - mB["relRMSE_mean"])

                # Permutation null: shuffle topo rows, refit C each time, compare to fixed B metrics.
                # Optimization: B^T B and T^T T are invariant to row permutation; only cross-terms and T^T y change.
                A = XtX_B  # B^T B
                Cmat = Ttr_s.T @ Ttr_s
                u = Xty_B  # B^T yc

                deltasP_null: list[float] = []
                deltasR_null: list[float] = []
                for _p in range(n_perms):
                    perm_tr = rng_placebo.permutation(Ttr_s.shape[0])
                    Ttr_p = Ttr_s[perm_tr]
                    if shuffle_test:
                        perm_te = rng_placebo.permutation(Tte_s.shape[0])
                        Tte_p = Tte_s[perm_te]
                    else:
                        Tte_p = Tte_s

                    D = Btr_s.T @ Ttr_p
                    v = Ttr_p.T @ yc
                    XtX = np.block([[A, D], [D.T, Cmat]])
                    Xty = np.vstack([u, v])
                    wP = np.linalg.solve(XtX + ridge_alpha * np.eye(XtX.shape[0]), Xty)
                    yP = (np.concatenate([Bte_s, Tte_p], axis=1) @ wP) + y_mean
                    mP = vec_metrics(yte, yP)
                    deltasP_null.append(float(mP["pearson_mean"] - mB["pearson_mean"]))
                    deltasR_null.append(float(mP["relRMSE_mean"] - mB["relRMSE_mean"]))

                nullP = np.asarray(deltasP_null, dtype=np.float64)
                nullR = np.asarray(deltasR_null, dtype=np.float64)
                nullP_mu = float(nullP.mean())
                nullP_sd = float(nullP.std(ddof=1)) if len(nullP) > 1 else 0.0
                nullR_mu = float(nullR.mean())
                nullR_sd = float(nullR.std(ddof=1)) if len(nullR) > 1 else 0.0

                zP = float((deltaP_real - nullP_mu) / nullP_sd) if nullP_sd > 0 else float("nan")
                zR = float((deltaR_real - nullR_mu) / nullR_sd) if nullR_sd > 0 else float("nan")

                # Empirical p-values
                pP = float(np.mean(nullP >= deltaP_real))
                pR = float(np.mean(nullR <= deltaR_real))  # smaller relRMSE delta is better
                pvals_pearson.append(pP)
                pvals_rel.append(pR)

                fold_rows.append(
                    {
                        "w": int(w),
                        "field_id": int(test_field),
                        "pearson_B": float(mB["pearson_mean"]),
                        "relRMSE_B": float(mB["relRMSE_mean"]),
                        "pearson_C": float(mC["pearson_mean"]),
                        "relRMSE_C": float(mC["relRMSE_mean"]),
                        "deltaP_real": float(deltaP_real),
                        "deltaR_real": float(deltaR_real),
                        "nullP_mean": float(nullP_mu),
                        "nullP_std": float(nullP_sd),
                        "nullR_mean": float(nullR_mu),
                        "nullR_std": float(nullR_sd),
                        "zP": float(zP),
                        "zR": float(zR),
                        "p_emp_P": float(pP),
                        "p_emp_R": float(pR),
                    }
                )

            # Aggregate per w
            dp = [float(r["deltaP_real"]) for r in fold_rows]
            dr = [float(r["deltaR_real"]) for r in fold_rows]
            zps = [float(r["zP"]) for r in fold_rows if np.isfinite(float(r["zP"]))]
            zrs = [float(r["zR"]) for r in fold_rows if np.isfinite(float(r["zR"]))]
            n_pos = int(np.sum(np.asarray(dp) > 0.0))

            fpP = fisher_p(pvals_pearson)
            fpR = fisher_p(pvals_rel)

            verdict = (
                (float(np.mean(dp)) > 0.0)
                and (float(np.mean(dr)) < 0.0)
                and (n_pos >= 7)
                and (fpP < 0.01)
            )

            residual_summary_by_w[w] = {
                "w": int(w),
                "deltaP_mean": float(np.mean(dp)),
                "deltaP_std": float(np.std(np.asarray(dp), ddof=1)) if len(dp) > 1 else 0.0,
                "deltaR_mean": float(np.mean(dr)),
                "deltaR_std": float(np.std(np.asarray(dr), ddof=1)) if len(dr) > 1 else 0.0,
                "zP_mean": float(np.mean(zps)) if zps else float("nan"),
                "zR_mean": float(np.mean(zrs)) if zrs else float("nan"),
                "fisher_p_P": float(fpP),
                "fisher_p_R": float(fpR),
                "n_pos_deltaP": int(n_pos),
                "n_fields": int(n_fields),
                "verdict_PASS": bool(verdict),
                "folds": fold_rows,
            }

            # Write per-w fold CSV
            header = list(fold_rows[0].keys()) if fold_rows else []
            write_csv(paths.run_dir / f"residual_lofo_by_field_w{w}.csv", header, [[r[h] for h in header] for r in fold_rows])

        # Baseline kernel table aggregated by w
        baseline_summary: dict[int, dict[str, float]] = {}
        for w in ws:
            rows_w = [r for r in baseline_rows if int(r["w"]) == int(w)]
            valsP = [float(r["pearson_mean"]) for r in rows_w]
            valsR = [float(r["relRMSE_mean"]) for r in rows_w]
            baseline_summary[w] = {
                "pearson_mean": float(np.mean(valsP)) if valsP else float("nan"),
                "pearson_std": float(np.std(np.asarray(valsP), ddof=1)) if len(valsP) > 1 else 0.0,
                "relRMSE_mean": float(np.mean(valsR)) if valsR else float("nan"),
                "relRMSE_std": float(np.std(np.asarray(valsR), ddof=1)) if len(valsR) > 1 else 0.0,
            }

        # Write baseline CSV
        header = list(baseline_rows[0].keys()) if baseline_rows else []
        write_csv(paths.run_dir / "baseline_kernel_by_field.csv", header, [[r[h] for h in header] for r in baseline_rows])

        # Summary markdown
        base_md_rows: list[dict[str, str]] = []
        for w in ws:
            b = baseline_summary[w]
            base_md_rows.append(
                {
                    "w": str(int(w)),
                    "pearson_mean": f"{b['pearson_mean']:.4f} ± {b['pearson_std']:.4f}",
                    "relRMSE_mean": f"{b['relRMSE_mean']:.4f} ± {b['relRMSE_std']:.4f}",
                }
            )

        resid_md_rows: list[dict[str, str]] = []
        for w in ws:
            s = residual_summary_by_w[w]
            resid_md_rows.append(
                {
                    "w": str(int(w)),
                    "ΔPearson(C-B) (resid)": f"{s['deltaP_mean']:.4f} ± {s['deltaP_std']:.4f}",
                    "ΔrelRMSE(C-B) (resid)": f"{s['deltaR_mean']:.4f} ± {s['deltaR_std']:.4f}",
                    "zP_mean": fmt(float(s["zP_mean"])),
                    "Fisher p (Pearson)": fmt(float(s["fisher_p_P"])),
                    "#folds ΔPearson>0": f"{int(s['n_pos_deltaP'])}/{int(s['n_fields'])}",
                    "verdict": "PASS" if bool(s["verdict_PASS"]) else "FAIL",
                }
            )

        summary_md = (
            "# E19 — Kernel sufficiency + topo-on-residual (LOFO)\n\n"
            f"- run: `{paths.run_dir}`\n"
            f"- grid_size={grid_size}, alpha={alpha}, k0_frac={k0_frac}, n_fields={n_fields}, patches_per_field={patches_per_field}\n"
            f"- impulse_run_dir: `{impulse_run_dir}`\n"
            f"- ws: {ws}\n"
            f"- topo_mode: `{topo_mode}` quantiles_b0={quantiles_b0}\n"
            f"- ridge_alpha: {ridge_alpha}  n_perms: {n_perms}  placebo_shuffle_test: {shuffle_test}\n\n"
            "## A) Baseline kernel (truncated impulse) performance on g_high\n"
            + md_table(base_md_rows, ["w", "pearson_mean", "relRMSE_mean"])
            + "\n\n"
            "## B) LOFO residual: Δ(C-B) on residual (gx_resid,gy_resid)\n"
            + md_table(
                resid_md_rows,
                ["w", "ΔPearson(C-B) (resid)", "ΔrelRMSE(C-B) (resid)", "zP_mean", "Fisher p (Pearson)", "#folds ΔPearson>0", "verdict"],
            )
            + "\n"
        )
        (paths.run_dir / "summary_e19_kernel_sufficiency.md").write_text(summary_md, encoding="utf-8")

        metrics = {
            "experiment": experiment,
            "exp_name": exp_name,
            "seed": seed,
            "grid_size": grid_size,
            "alpha": alpha,
            "k0_frac": k0_frac,
            "n_fields": n_fields,
            "patches_per_field": patches_per_field,
            "ws": ws,
            "impulse_run_dir": str(impulse_run_dir),
            "topo_mode": topo_mode,
            "quantiles_b0": quantiles_b0,
            "ridge_alpha": ridge_alpha,
            "n_perms": n_perms,
            "placebo_shuffle_test": shuffle_test,
            "baseline_by_field": baseline_rows,
            "baseline_summary": baseline_summary,
            "residual_summary_by_w": residual_summary_by_w,
        }
        write_json(paths.metrics_json, metrics)
        return paths

    if experiment == "e20":
        # Sufficiency test v2: baseline = best local linear predictor on pixels (P), then test topo on residual in LOFO.
        from numpy.lib.stride_tricks import sliding_window_view
        from scipy import stats
        from sklearn.linear_model import SGDRegressor

        grid_size = int(cfg.get("grid_size", 256))
        alpha = float(cfg.get("alpha", 2.0))
        k0_frac = float(cfg.get("k0_frac", 0.15))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 20_000))
        ws = [int(x) for x in cfg.get("ws", [17, 33])]
        ridge_alpha = float(cfg.get("ridge_alpha", 1.0))
        quantiles_b0 = [float(x) for x in cfg.get("quantiles_b0", [0.6, 0.7, 0.8, 0.9])]
        n_perms = int(cfg.get("n_perms", 200))

        pixel_epochs = int(cfg.get("pixel_epochs", 3))
        pixel_batch_size = int(cfg.get("pixel_batch_size", 4096))
        pixel_alpha = float(cfg.get("pixel_alpha", 1e-4))
        pixel_eta0 = float(cfg.get("pixel_eta0", 0.01))
        pixel_power_t = float(cfg.get("pixel_power_t", 0.25))

        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if not (0.0 < k0_frac < 0.5):
            raise ValueError("k0_frac must be in (0,0.5)")
        if n_fields < 2:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        if any(w <= 0 or (w % 2) == 0 for w in ws):
            raise ValueError("ws must be odd positive window sizes")
        if any((q <= 0.0) or (q >= 1.0) for q in quantiles_b0):
            raise ValueError("quantiles_b0 must be in (0,1)")
        if n_perms <= 0:
            raise ValueError("n_perms must be > 0")

        def fisher_p(ps: list[float]) -> float:
            if not ps:
                return float("nan")
            eps = 1.0 / float(n_perms + 1)
            ps2 = [min(1.0, max(float(p), eps)) for p in ps]
            stat = -2.0 * float(np.sum(np.log(ps2)))
            return float(stats.chi2.sf(stat, 2 * len(ps2)))

        def fmt(x: float) -> str:
            if not np.isfinite(x):
                return "nan"
            ax = abs(float(x))
            if ax != 0.0 and (ax < 1e-3 or ax >= 1e3):
                return f"{x:.3e}"
            return f"{x:.4f}"

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r in rows:
                out.append("| " + " | ".join(r.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        def vec_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
            from .models import rmse, safe_pearson

            out: dict[str, float] = {}
            for i, name in enumerate(["gx", "gy"]):
                yt = np.asarray(y_true[:, i], dtype=np.float64)
                yp = np.asarray(y_pred[:, i], dtype=np.float64)
                out[f"pearson_{name}"] = float(safe_pearson(yt, yp))
                r = float(rmse(yt, yp))
                s = float(np.std(yt))
                out[f"relRMSE_{name}"] = float(r / s) if s > 0 else float("nan")
            out["pearson_mean"] = float(np.mean([out["pearson_gx"], out["pearson_gy"]]))
            out["relRMSE_mean"] = float(np.mean([out["relRMSE_gx"], out["relRMSE_gy"]]))
            return out

        def fit_pixel_model(train_fields: list[int], Xpix_by_field: list[np.ndarray], y_by_field: list[np.ndarray]) -> tuple[SGDRegressor, SGDRegressor]:
            reg_gx = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(pixel_alpha),
                learning_rate="invscaling",
                eta0=float(pixel_eta0),
                power_t=float(pixel_power_t),
                max_iter=1,
                tol=None,
                fit_intercept=True,
                random_state=int(placebo_seed) + 0,
            )
            reg_gy = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(pixel_alpha),
                learning_rate="invscaling",
                eta0=float(pixel_eta0),
                power_t=float(pixel_power_t),
                max_iter=1,
                tol=None,
                fit_intercept=True,
                random_state=int(placebo_seed) + 1,
            )
            for _epoch in range(pixel_epochs):
                for fid in train_fields:
                    X = Xpix_by_field[fid]
                    y = y_by_field[fid]
                    perm = rng_placebo.permutation(X.shape[0])
                    for start in range(0, X.shape[0], pixel_batch_size):
                        idx = perm[start : start + pixel_batch_size]
                        Xb = X[idx]
                        reg_gx.partial_fit(Xb, y[idx, 0])
                        reg_gy.partial_fit(Xb, y[idx, 1])
            return reg_gx, reg_gy

        def predict_pixel(reg_gx: SGDRegressor, reg_gy: SGDRegressor, X: np.ndarray) -> np.ndarray:
            X = np.asarray(X)
            y_pred = np.empty((X.shape[0], 2), dtype=np.float64)
            for start in range(0, X.shape[0], pixel_batch_size):
                sl = slice(start, min(X.shape[0], start + pixel_batch_size))
                y_pred[sl, 0] = reg_gx.predict(X[sl])
                y_pred[sl, 1] = reg_gy.predict(X[sl])
            return y_pred

        # Shared per-field maps (rho_high z-scored for stability; target is g_high on the original split).
        rho_high_by_field: list[np.ndarray] = []
        gx_by_field: list[np.ndarray] = []
        gy_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
            split = band_split_poisson_2d(rho01, k0_frac=k0_frac)
            rho_high = _zscore_field(split.rho_high)
            rho_high_by_field.append(np.asarray(rho_high, dtype=np.float64))
            gx_by_field.append(np.asarray(split.high.gx, dtype=np.float64))
            gy_by_field.append(np.asarray(split.high.gy, dtype=np.float64))

        per_w_results: dict[int, dict[str, Any]] = {}

        for w in ws:
            ps = _require_odd("w", w)
            r = ps // 2

            # Build per-field datasets: pixels, B features, topo features, y (g_high) for patch centers.
            Xpix_by_field: list[np.ndarray] = []
            B_by_field: list[np.ndarray] = []
            T_by_field: list[np.ndarray] = []
            y_by_field: list[np.ndarray] = []
            topo_names: list[str] | None = None

            for field_id in range(n_fields):
                field = rho_high_by_field[field_id]
                gx = gx_by_field[field_id]
                gy = gy_by_field[field_id]

                # thresholds per field (quantiles on the z-scored rho_high map)
                thr = [float(np.quantile(field.reshape(-1), q)) for q in quantiles_b0]

                # sample centers + compute B/topo features on the same field
                rng_samp = np.random.default_rng(seed + 10_000 * field_id + 17 * ps)
                feats, y, centers = _sample_features_2d_fast_topo(
                    field=field,
                    gx=gx,
                    gy=gy,
                    n_patches=patches_per_field,
                    patch_size=ps,
                    topo_mode="quantile_per_field",
                    topo_thresholds=thr,
                    thresholds_pos_sigma=None,
                    thresholds_neg_sigma=None,
                    rng=rng_samp,
                )
                y = np.asarray(y, dtype=np.float64)
                assert y.shape == (patches_per_field, 2)

                # pixel patches at the same centers
                cx = centers[:, 0].astype(np.int64)
                cy = centers[:, 1].astype(np.int64)
                win = sliding_window_view(field, (ps, ps))
                Xpix = win[cx - r, cy - r].reshape(patches_per_field, -1).astype(np.float32, copy=False)

                B_names = ["mass", "mass2", "var", "max", "grad_energy"]
                B = np.column_stack([np.asarray(feats[n], dtype=np.float64) for n in B_names])

                tnames = sorted([k for k in feats.keys() if k.startswith("b0_")])
                if topo_names is None:
                    topo_names = tnames
                elif topo_names != tnames:
                    raise RuntimeError(f"Topo feature names mismatch across fields: {topo_names} vs {tnames}")
                T = np.column_stack([np.asarray(feats[n], dtype=np.float64) for n in topo_names])

                Xpix_by_field.append(Xpix)
                B_by_field.append(B)
                T_by_field.append(T)
                y_by_field.append(y)

            if topo_names is None:
                raise RuntimeError("No topo features found")

            # LOFO: baseline pixels-only P, residual models B_resid vs C_resid with topo placebo.
            fold_rows: list[dict[str, Any]] = []
            pvals_pearson: list[float] = []

            for test_field in range(n_fields):
                train_fields = [i for i in range(n_fields) if i != test_field]

                reg_gx, reg_gy = fit_pixel_model(train_fields, Xpix_by_field, y_by_field)

                # baseline predictions on test
                yte = y_by_field[test_field]
                yP_te = predict_pixel(reg_gx, reg_gy, Xpix_by_field[test_field])
                mP = vec_metrics(yte, yP_te)

                # build residual targets for train and test
                yres_tr_list: list[np.ndarray] = []
                Btr_list: list[np.ndarray] = []
                Ttr_list: list[np.ndarray] = []
                for fid in train_fields:
                    ytr = y_by_field[fid]
                    yP_tr = predict_pixel(reg_gx, reg_gy, Xpix_by_field[fid])
                    yres_tr_list.append(ytr - yP_tr)
                    Btr_list.append(B_by_field[fid])
                    Ttr_list.append(T_by_field[fid])

                ytr_res = np.concatenate(yres_tr_list, axis=0)
                Btr = np.concatenate(Btr_list, axis=0)
                Ttr = np.concatenate(Ttr_list, axis=0)

                yte_res = yte - yP_te
                Bte = B_by_field[test_field]
                Tte = T_by_field[test_field]

                # standardize residual-model features on train
                B_mu = Btr.mean(axis=0)
                B_sd = np.where(Btr.std(axis=0) > 0, Btr.std(axis=0), 1.0)
                T_mu = Ttr.mean(axis=0)
                T_sd = np.where(Ttr.std(axis=0) > 0, Ttr.std(axis=0), 1.0)
                Btr_s = (Btr - B_mu) / B_sd
                Bte_s = (Bte - B_mu) / B_sd
                Ttr_s = (Ttr - T_mu) / T_sd
                Tte_s = (Tte - T_mu) / T_sd

                y_mean = ytr_res.mean(axis=0)
                yc = ytr_res - y_mean

                # Fit B_resid
                XtX_B = Btr_s.T @ Btr_s
                Xty_B = Btr_s.T @ yc
                wB = np.linalg.solve(XtX_B + ridge_alpha * np.eye(XtX_B.shape[0]), Xty_B)
                yB_te = Bte_s @ wB + y_mean
                mB = vec_metrics(yte_res, yB_te)

                # Fit C_resid (real)
                XtrC = np.concatenate([Btr_s, Ttr_s], axis=1)
                XteC = np.concatenate([Bte_s, Tte_s], axis=1)
                XtX_C = XtrC.T @ XtrC
                Xty_C = XtrC.T @ yc
                wC = np.linalg.solve(XtX_C + ridge_alpha * np.eye(XtX_C.shape[0]), Xty_C)
                yC_te = XteC @ wC + y_mean
                mC = vec_metrics(yte_res, yC_te)

                deltaP_real = float(mC["pearson_mean"] - mB["pearson_mean"])
                deltaR_real = float(mC["relRMSE_mean"] - mB["relRMSE_mean"])

                # Placebo: shuffle topo rows in train only, refit C each time.
                A = XtX_B
                Cmat = Ttr_s.T @ Ttr_s
                u = Xty_B
                deltasP_null: list[float] = []
                deltasR_null: list[float] = []
                for _p in range(n_perms):
                    perm_tr = rng_placebo.permutation(Ttr_s.shape[0])
                    Ttr_p = Ttr_s[perm_tr]
                    D = Btr_s.T @ Ttr_p
                    v = Ttr_p.T @ yc
                    XtX = np.block([[A, D], [D.T, Cmat]])
                    Xty = np.vstack([u, v])
                    wP = np.linalg.solve(XtX + ridge_alpha * np.eye(XtX.shape[0]), Xty)
                    yP_res = (np.concatenate([Bte_s, Tte_s], axis=1) @ wP) + y_mean
                    mP_res = vec_metrics(yte_res, yP_res)
                    deltasP_null.append(float(mP_res["pearson_mean"] - mB["pearson_mean"]))
                    deltasR_null.append(float(mP_res["relRMSE_mean"] - mB["relRMSE_mean"]))

                nullP = np.asarray(deltasP_null, dtype=np.float64)
                nullR = np.asarray(deltasR_null, dtype=np.float64)
                nullP_mu = float(nullP.mean())
                nullP_sd = float(nullP.std(ddof=1)) if len(nullP) > 1 else 0.0
                nullR_mu = float(nullR.mean())
                nullR_sd = float(nullR.std(ddof=1)) if len(nullR) > 1 else 0.0
                zP = float((deltaP_real - nullP_mu) / nullP_sd) if nullP_sd > 0 else float("nan")
                zR = float((deltaR_real - nullR_mu) / nullR_sd) if nullR_sd > 0 else float("nan")
                pP = float(np.mean(nullP >= deltaP_real))
                pvals_pearson.append(pP)

                fold_rows.append(
                    {
                        "w": int(w),
                        "field_id": int(test_field),
                        "pearson_P": float(mP["pearson_mean"]),
                        "relRMSE_P": float(mP["relRMSE_mean"]),
                        "pearson_B_resid": float(mB["pearson_mean"]),
                        "relRMSE_B_resid": float(mB["relRMSE_mean"]),
                        "pearson_C_resid": float(mC["pearson_mean"]),
                        "relRMSE_C_resid": float(mC["relRMSE_mean"]),
                        "deltaP_resid_real": float(deltaP_real),
                        "deltaR_resid_real": float(deltaR_real),
                        "nullP_mean": float(nullP_mu),
                        "nullP_std": float(nullP_sd),
                        "zP": float(zP),
                        "p_emp_P": float(pP),
                        "nullR_mean": float(nullR_mu),
                        "nullR_std": float(nullR_sd),
                        "zR": float(zR),
                    }
                )

            # Aggregate per w
            p_perf = [float(r["pearson_P"]) for r in fold_rows]
            r_perf = [float(r["relRMSE_P"]) for r in fold_rows]
            dp = [float(r["deltaP_resid_real"]) for r in fold_rows]
            dr = [float(r["deltaR_resid_real"]) for r in fold_rows]
            zps = [float(r["zP"]) for r in fold_rows if np.isfinite(float(r["zP"]))]
            n_pos = int(np.sum(np.asarray(dp) > 0.0))
            fpP = fisher_p(pvals_pearson)

            verdict = (float(np.mean(dp)) > 0.0) and (float(np.mean(dr)) < 0.0) and (n_pos >= 7) and (fpP < 0.01)

            per_w_results[w] = {
                "w": int(w),
                "P_pearson_mean": float(np.mean(p_perf)),
                "P_pearson_std": float(np.std(np.asarray(p_perf), ddof=1)) if len(p_perf) > 1 else 0.0,
                "P_relRMSE_mean": float(np.mean(r_perf)),
                "P_relRMSE_std": float(np.std(np.asarray(r_perf), ddof=1)) if len(r_perf) > 1 else 0.0,
                "deltaP_resid_mean": float(np.mean(dp)),
                "deltaP_resid_std": float(np.std(np.asarray(dp), ddof=1)) if len(dp) > 1 else 0.0,
                "deltaR_resid_mean": float(np.mean(dr)),
                "deltaR_resid_std": float(np.std(np.asarray(dr), ddof=1)) if len(dr) > 1 else 0.0,
                "zP_mean": float(np.mean(zps)) if zps else float("nan"),
                "fisher_p_P": float(fpP),
                "n_pos_deltaP": int(n_pos),
                "n_fields": int(n_fields),
                "verdict_PASS": bool(verdict),
                "folds": fold_rows,
                "topo_names": topo_names,
            }

            # Per-w fold CSV
            header = list(fold_rows[0].keys()) if fold_rows else []
            write_csv(paths.run_dir / f"lofo_by_field_w{w}.csv", header, [[r[h] for h in header] for r in fold_rows])

        # Summary markdown
        perf_rows: list[dict[str, str]] = []
        resid_rows: list[dict[str, str]] = []
        for w in ws:
            s = per_w_results[w]
            perf_rows.append(
                {
                    "w": str(int(w)),
                    "pearson_P": f"{s['P_pearson_mean']:.4f} ± {s['P_pearson_std']:.4f}",
                    "relRMSE_P": f"{s['P_relRMSE_mean']:.4f} ± {s['P_relRMSE_std']:.4f}",
                }
            )
            resid_rows.append(
                {
                    "w": str(int(w)),
                    "ΔPearson(C-B) (resid)": f"{s['deltaP_resid_mean']:.4f} ± {s['deltaP_resid_std']:.4f}",
                    "ΔrelRMSE(C-B) (resid)": f"{s['deltaR_resid_mean']:.4f} ± {s['deltaR_resid_std']:.4f}",
                    "zP_mean": fmt(float(s["zP_mean"])),
                    "Fisher p (Pearson)": fmt(float(s["fisher_p_P"])),
                    "#folds ΔPearson>0": f"{int(s['n_pos_deltaP'])}/{int(s['n_fields'])}",
                    "verdict": "PASS" if bool(s["verdict_PASS"]) else "FAIL",
                }
            )

        summary_md = (
            "# E20 — Pixels-only baseline sufficiency + topo-on-residual (LOFO)\n\n"
            f"- run: `{paths.run_dir}`\n"
            f"- grid_size={grid_size}, alpha={alpha}, k0_frac={k0_frac}, n_fields={n_fields}, patches_per_field={patches_per_field}\n"
            f"- ws: {ws}\n"
            f"- pixel: SGD L2 (ridge-like) alpha={pixel_alpha}, epochs={pixel_epochs}, batch={pixel_batch_size}\n"
            f"- residual ridge_alpha={ridge_alpha}, topo quantiles_b0={quantiles_b0}, perms={n_perms}\n\n"
            "## A) Baseline P (pixels-only) LOFO performance on g_high\n"
            + md_table(perf_rows, ["w", "pearson_P", "relRMSE_P"])
            + "\n\n"
            "## B) LOFO residual: Δ(C-B) on residual (gx_resid,gy_resid)\n"
            + md_table(
                resid_rows,
                ["w", "ΔPearson(C-B) (resid)", "ΔrelRMSE(C-B) (resid)", "zP_mean", "Fisher p (Pearson)", "#folds ΔPearson>0", "verdict"],
            )
            + "\n"
        )
        (paths.run_dir / "summary_e20_pixels_sufficiency.md").write_text(summary_md, encoding="utf-8")

        metrics = {
            "experiment": experiment,
            "exp_name": exp_name,
            "seed": seed,
            "grid_size": grid_size,
            "alpha": alpha,
            "k0_frac": k0_frac,
            "n_fields": n_fields,
            "patches_per_field": patches_per_field,
            "ws": ws,
            "pixel": {
                "solver": "sgd_l2",
                "alpha": pixel_alpha,
                "epochs": pixel_epochs,
                "batch_size": pixel_batch_size,
                "eta0": pixel_eta0,
                "power_t": pixel_power_t,
            },
            "residual": {"ridge_alpha": ridge_alpha, "quantiles_b0": quantiles_b0, "n_perms": n_perms},
            "per_w": per_w_results,
        }
        write_json(paths.metrics_json, metrics)
        return paths

    if experiment == "e21":
        # Injection study: detectability limit of an injected nonlinear topo term beyond pixels-only baseline (LOFO).
        from numpy.lib.stride_tricks import sliding_window_view
        from scipy import stats
        from sklearn.linear_model import SGDRegressor

        grid_size = int(cfg.get("grid_size", 256))
        alpha = float(cfg.get("alpha", 2.0))
        k0_frac = float(cfg.get("k0_frac", 0.15))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 10_000))
        ws = [int(x) for x in cfg.get("ws", [17, 33])]
        eps_list = [float(x) for x in cfg.get("eps_list", [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3])]

        ridge_alpha = float(cfg.get("ridge_alpha", 1.0))
        quantiles_b0 = [float(x) for x in cfg.get("quantiles_b0", [0.6, 0.7, 0.8, 0.9])]
        n_perms = int(cfg.get("n_perms", 200))

        pixel_epochs = int(cfg.get("pixel_epochs", 2))
        pixel_batch_size = int(cfg.get("pixel_batch_size", 4096))
        pixel_alpha = float(cfg.get("pixel_alpha", 1e-4))
        pixel_eta0 = float(cfg.get("pixel_eta0", 0.01))
        pixel_power_t = float(cfg.get("pixel_power_t", 0.25))

        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if not (0.0 < k0_frac < 0.5):
            raise ValueError("k0_frac must be in (0,0.5)")
        if n_fields < 2:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        if any(w <= 0 or (w % 2) == 0 for w in ws):
            raise ValueError("ws must be odd positive window sizes")
        if any((q <= 0.0) or (q >= 1.0) for q in quantiles_b0):
            raise ValueError("quantiles_b0 must be in (0,1)")
        if n_perms <= 0:
            raise ValueError("n_perms must be > 0")
        if not eps_list:
            raise ValueError("eps_list must be non-empty")

        def fisher_p(ps: list[float]) -> float:
            if not ps:
                return float("nan")
            eps = 1.0 / float(n_perms + 1)
            ps2 = [min(1.0, max(float(p), eps)) for p in ps]
            stat = -2.0 * float(np.sum(np.log(ps2)))
            return float(stats.chi2.sf(stat, 2 * len(ps2)))

        def fmt(x: float) -> str:
            if not np.isfinite(x):
                return "nan"
            ax = abs(float(x))
            if ax != 0.0 and (ax < 1e-3 or ax >= 1e3):
                return f"{x:.3e}"
            return f"{x:.4f}"

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r in rows:
                out.append("| " + " | ".join(r.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        def pearson_fast(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            out = []
            for j in range(2):
                a = y_true[:, j]
                b = y_pred[:, j]
                am = a - float(a.mean())
                bm = b - float(b.mean())
                denom = float(np.linalg.norm(am) * np.linalg.norm(bm)) + 1e-12
                out.append(float((am @ bm) / denom))
            return float(np.mean(out))

        def relrmse_fast(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            vals = []
            for j in range(2):
                a = y_true[:, j]
                b = y_pred[:, j]
                rm = float(np.sqrt(np.mean((a - b) ** 2)))
                sd = float(np.std(a))
                vals.append(rm / sd if sd > 0 else float("nan"))
            return float(np.mean(vals))

        def fit_pixel_model(train_fields: list[int], Xpix_by_field: list[np.ndarray], y_by_field: list[np.ndarray], *, rng_perm: np.random.Generator) -> tuple[SGDRegressor, SGDRegressor]:
            reg_gx = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(pixel_alpha),
                learning_rate="invscaling",
                eta0=float(pixel_eta0),
                power_t=float(pixel_power_t),
                max_iter=1,
                tol=None,
                fit_intercept=True,
                random_state=int(placebo_seed) + 0,
            )
            reg_gy = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(pixel_alpha),
                learning_rate="invscaling",
                eta0=float(pixel_eta0),
                power_t=float(pixel_power_t),
                max_iter=1,
                tol=None,
                fit_intercept=True,
                random_state=int(placebo_seed) + 1,
            )
            for _epoch in range(pixel_epochs):
                for fid in train_fields:
                    X = Xpix_by_field[fid]
                    y = y_by_field[fid]
                    perm = rng_perm.permutation(X.shape[0])
                    for start in range(0, X.shape[0], pixel_batch_size):
                        idx = perm[start : start + pixel_batch_size]
                        Xb = X[idx]
                        reg_gx.partial_fit(Xb, y[idx, 0])
                        reg_gy.partial_fit(Xb, y[idx, 1])
            return reg_gx, reg_gy

        def predict_pixel(reg_gx: SGDRegressor, reg_gy: SGDRegressor, X: np.ndarray) -> np.ndarray:
            X = np.asarray(X)
            y_pred = np.empty((X.shape[0], 2), dtype=np.float64)
            for start in range(0, X.shape[0], pixel_batch_size):
                sl = slice(start, min(X.shape[0], start + pixel_batch_size))
                y_pred[sl, 0] = reg_gx.predict(X[sl])
                y_pred[sl, 1] = reg_gy.predict(X[sl])
            return y_pred

        # Generate per-field maps once (rho_high z-scored per field).
        rho_high_by_field: list[np.ndarray] = []
        gx_by_field: list[np.ndarray] = []
        gy_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
            split = band_split_poisson_2d(rho01, k0_frac=k0_frac)
            rho_high = _zscore_field(split.rho_high)
            rho_high_by_field.append(np.asarray(rho_high, dtype=np.float64))
            gx_by_field.append(np.asarray(split.high.gx, dtype=np.float64))
            gy_by_field.append(np.asarray(split.high.gy, dtype=np.float64))

        results_by_w: dict[int, Any] = {}

        for w in ws:
            ps = _require_odd("w", w)
            r = ps // 2

            Xpix_by_field: list[np.ndarray] = []
            B_by_field: list[np.ndarray] = []
            Tmulti_by_field: list[np.ndarray] = []
            t_scalar_by_field: list[np.ndarray] = []
            y_by_field: list[np.ndarray] = []
            u_by_field: list[np.ndarray] = []
            tz_by_field: list[np.ndarray] = []
            topo_names: list[str] | None = None

            # Build patch datasets for this w.
            for field_id in range(n_fields):
                field = rho_high_by_field[field_id]
                gx = gx_by_field[field_id]
                gy = gy_by_field[field_id]

                thr = [float(np.quantile(field.reshape(-1), q)) for q in quantiles_b0]
                rng_samp = np.random.default_rng(seed + 10_000 * field_id + 17 * ps)
                feats, y, centers = _sample_features_2d_fast_topo(
                    field=field,
                    gx=gx,
                    gy=gy,
                    n_patches=patches_per_field,
                    patch_size=ps,
                    topo_mode="quantile_per_field",
                    topo_thresholds=thr,
                    thresholds_pos_sigma=None,
                    thresholds_neg_sigma=None,
                    rng=rng_samp,
                )
                y = np.asarray(y, dtype=np.float64)
                assert y.shape == (patches_per_field, 2)

                cx = centers[:, 0].astype(np.int64)
                cy = centers[:, 1].astype(np.int64)
                win = sliding_window_view(field, (ps, ps))
                Xpix = win[cx - r, cy - r].reshape(patches_per_field, -1).astype(np.float32, copy=False)

                B_names = ["mass", "mass2", "var", "max", "grad_energy"]
                B = np.column_stack([np.asarray(feats[n], dtype=np.float64) for n in B_names])

                tnames = sorted([k for k in feats.keys() if k.startswith("b0_")])
                if topo_names is None:
                    topo_names = tnames
                elif topo_names != tnames:
                    raise RuntimeError(f"Topo feature names mismatch across fields: {topo_names} vs {tnames}")
                Tmulti = np.column_stack([np.asarray(feats[n], dtype=np.float64) for n in topo_names])
                t_scalar = np.mean(Tmulti, axis=1).astype(np.float64, copy=False)

                Xpix_by_field.append(Xpix)
                B_by_field.append(B)
                Tmulti_by_field.append(Tmulti)
                t_scalar_by_field.append(t_scalar)
                y_by_field.append(y)

            if topo_names is None:
                raise RuntimeError("No topo features found")

            # Global scaling for injected term: s = std(|g_high|) over all patches.
            mags = np.concatenate([np.sqrt(np.sum(y * y, axis=1)) for y in y_by_field], axis=0)
            s_scale = float(np.std(mags))

            # Global z-score for scalar topo t.
            t_all = np.concatenate(t_scalar_by_field, axis=0)
            t_mu = float(np.mean(t_all))
            t_sd = float(np.std(t_all))
            if t_sd <= 0:
                t_sd = 1.0
            tz_by_field = [(t - t_mu) / t_sd for t in t_scalar_by_field]

            # Compute LOFO out-of-fold pixel predictions on the original target (g_high) to define u.
            pred_oof_by_field: list[np.ndarray] = [np.zeros_like(y_by_field[i]) for i in range(n_fields)]
            for test_field in range(n_fields):
                train_fields = [i for i in range(n_fields) if i != test_field]
                reg_gx, reg_gy = fit_pixel_model(train_fields, Xpix_by_field, y_by_field, rng_perm=rng_placebo)
                pred_oof_by_field[test_field] = predict_pixel(reg_gx, reg_gy, Xpix_by_field[test_field])

            u_by_field = []
            for fid in range(n_fields):
                yp = np.asarray(pred_oof_by_field[fid], dtype=np.float64)
                nrm = np.linalg.norm(yp, axis=1)
                u = np.zeros_like(yp, dtype=np.float64)
                good = nrm > 1e-12
                u[good] = yp[good] / nrm[good, None]
                u[~good, 0] = 1.0
                u_by_field.append(u)

            # Precompute fold-invariant standardized feature matrices and permutation cross-terms for the residual model.
            fold_cache: dict[int, dict[str, Any]] = {}
            for test_field in range(n_fields):
                train_fields = [i for i in range(n_fields) if i != test_field]
                Btr = np.concatenate([B_by_field[i] for i in train_fields], axis=0)
                Ttr = np.concatenate([Tmulti_by_field[i] for i in train_fields], axis=0)
                Bte = B_by_field[test_field]
                Tte = Tmulti_by_field[test_field]

                B_mu = Btr.mean(axis=0)
                B_sd = np.where(Btr.std(axis=0) > 0, Btr.std(axis=0), 1.0)
                T_mu = Ttr.mean(axis=0)
                T_sd = np.where(Ttr.std(axis=0) > 0, Ttr.std(axis=0), 1.0)
                Btr_s = (Btr - B_mu) / B_sd
                Bte_s = (Bte - B_mu) / B_sd
                Ttr_s = (Ttr - T_mu) / T_sd
                Tte_s = (Tte - T_mu) / T_sd

                A = Btr_s.T @ Btr_s
                Cmat = Ttr_s.T @ Ttr_s
                D_real = Btr_s.T @ Ttr_s
                XtX_C = np.block([[A, D_real], [D_real.T, Cmat]])

                # Store permutation indices and the corresponding D matrices (B^T T_perm), reused across eps.
                perms = [rng_placebo.permutation(Ttr_s.shape[0]) for _ in range(n_perms)]
                D_perms = np.stack([Btr_s.T @ Ttr_s[p] for p in perms], axis=0)  # (P, pB, pT)

                fold_cache[test_field] = {
                    "train_fields": train_fields,
                    "Btr_s": Btr_s,
                    "Bte_s": Bte_s,
                    "Ttr_s": Ttr_s,
                    "Tte_s": Tte_s,
                    "A": A,
                    "Cmat": Cmat,
                    "XtX_C": XtX_C,
                    "perms": perms,
                    "D_perms": D_perms,
                }

            eps_rows: list[dict[str, Any]] = []

            for eps in eps_list:
                # Build injected targets (per field)
                y_total_by_field = []
                for fid in range(n_fields):
                    y = y_by_field[fid]
                    tz = tz_by_field[fid].reshape(-1, 1)
                    u = u_by_field[fid]
                    y_total = y + float(eps) * float(s_scale) * (tz * u)
                    y_total_by_field.append(y_total.astype(np.float64, copy=False))

                fold_rows: list[dict[str, Any]] = []
                pvals_pearson: list[float] = []

                for test_field in range(n_fields):
                    cache = fold_cache[test_field]
                    train_fields = cache["train_fields"]

                    # Train pixels-only baseline P on y_total.
                    reg_gx, reg_gy = fit_pixel_model(train_fields, Xpix_by_field, y_total_by_field, rng_perm=rng_placebo)

                    # Predict on all fields to get residual targets.
                    ypred_by_field: list[np.ndarray] = []
                    for fid in range(n_fields):
                        ypred_by_field.append(predict_pixel(reg_gx, reg_gy, Xpix_by_field[fid]))

                    # Residual targets for this fold
                    ytr_res = np.concatenate([y_total_by_field[i] - ypred_by_field[i] for i in train_fields], axis=0)
                    yte_res = y_total_by_field[test_field] - ypred_by_field[test_field]

                    Btr_s = cache["Btr_s"]
                    Bte_s = cache["Bte_s"]
                    Ttr_s = cache["Ttr_s"]
                    Tte_s = cache["Tte_s"]

                    y_mean = ytr_res.mean(axis=0)
                    yc = ytr_res - y_mean

                    # Fit B_resid
                    A = cache["A"]
                    uB = Btr_s.T @ yc
                    wB = np.linalg.solve(A + ridge_alpha * np.eye(A.shape[0]), uB)
                    yB_te = Bte_s @ wB + y_mean
                    pB = pearson_fast(yte_res, yB_te)
                    rB = relrmse_fast(yte_res, yB_te)

                    # Fit C_resid (real)
                    XtX_C = cache["XtX_C"]
                    v_real = Ttr_s.T @ yc
                    Xty_C = np.vstack([uB, v_real])
                    wC = np.linalg.solve(XtX_C + ridge_alpha * np.eye(XtX_C.shape[0]), Xty_C)
                    XteC = np.concatenate([Bte_s, Tte_s], axis=1)
                    yC_te = XteC @ wC + y_mean
                    pC = pearson_fast(yte_res, yC_te)
                    rC = relrmse_fast(yte_res, yC_te)

                    deltaP_real = float(pC - pB)
                    deltaR_real = float(rC - rB)

                    # Permutation null for ΔPearson: shuffle topo rows in train only.
                    perms = cache["perms"]
                    D_perms = cache["D_perms"]
                    Cmat = cache["Cmat"]
                    null_dP: list[float] = []
                    for pi in range(n_perms):
                        pidx = perms[pi]
                        D = D_perms[pi]
                        v = Ttr_s[pidx].T @ yc
                        XtX = np.block([[A, D], [D.T, Cmat]])
                        Xty = np.vstack([uB, v])
                        wP = np.linalg.solve(XtX + ridge_alpha * np.eye(XtX.shape[0]), Xty)
                        yP_te = XteC @ wP + y_mean
                        pP = pearson_fast(yte_res, yP_te)
                        null_dP.append(float(pP - pB))

                    null = np.asarray(null_dP, dtype=np.float64)
                    null_mu = float(null.mean())
                    null_sd = float(null.std(ddof=1)) if len(null) > 1 else 0.0
                    zP = float((deltaP_real - null_mu) / null_sd) if null_sd > 0 else float("nan")
                    p_emp = float(np.mean(null >= deltaP_real))
                    pvals_pearson.append(p_emp)

                    fold_rows.append(
                        {
                            "w": int(w),
                            "eps": float(eps),
                            "field_id": int(test_field),
                            "pearson_P": float(pearson_fast(y_total_by_field[test_field], ypred_by_field[test_field])),
                            "relRMSE_P": float(relrmse_fast(y_total_by_field[test_field], ypred_by_field[test_field])),
                            "deltaP_resid_real": float(deltaP_real),
                            "deltaR_resid_real": float(deltaR_real),
                            "zP": float(zP),
                            "p_emp_P": float(p_emp),
                        }
                    )

                # Aggregate for this eps
                perfP = np.asarray([r["pearson_P"] for r in fold_rows], dtype=np.float64)
                perfR = np.asarray([r["relRMSE_P"] for r in fold_rows], dtype=np.float64)
                dp = np.asarray([r["deltaP_resid_real"] for r in fold_rows], dtype=np.float64)
                dr = np.asarray([r["deltaR_resid_real"] for r in fold_rows], dtype=np.float64)
                zps = np.asarray([r["zP"] for r in fold_rows], dtype=np.float64)
                n_pos = int(np.sum(dp > 0.0))
                fp = fisher_p(pvals_pearson)

                verdict = (float(dp.mean()) > 0.0) and (float(dr.mean()) < 0.0) and (n_pos >= 7) and (fp < 0.05)

                eps_rows.append(
                    {
                        "w": int(w),
                        "eps": float(eps),
                        "P_pearson_mean": float(perfP.mean()),
                        "P_pearson_std": float(perfP.std(ddof=1)) if len(perfP) > 1 else 0.0,
                        "P_relRMSE_mean": float(perfR.mean()),
                        "P_relRMSE_std": float(perfR.std(ddof=1)) if len(perfR) > 1 else 0.0,
                        "deltaP_mean": float(dp.mean()),
                        "deltaP_std": float(dp.std(ddof=1)) if len(dp) > 1 else 0.0,
                        "deltaR_mean": float(dr.mean()),
                        "deltaR_std": float(dr.std(ddof=1)) if len(dr) > 1 else 0.0,
                        "zP_mean": float(np.nanmean(zps)),
                        "fisher_p_P": float(fp),
                        "n_pos_deltaP": int(n_pos),
                        "verdict_PASS": bool(verdict),
                        "folds": fold_rows,
                    }
                )

                # Write per-eps fold CSV
                header = list(fold_rows[0].keys()) if fold_rows else []
                tag = f"w{w}_eps{str(eps).replace('.','p')}"
                write_csv(paths.run_dir / f"lofo_by_field_{tag}.csv", header, [[r[h] for h in header] for r in fold_rows])

            # Detection threshold: smallest eps passing.
            det = [r for r in eps_rows if bool(r["verdict_PASS"])]
            eps_det = float(min((r["eps"] for r in det), default=float("nan")))

            results_by_w[w] = {
                "w": int(w),
                "grid_size": grid_size,
                "alpha": alpha,
                "k0_frac": k0_frac,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "pixel": {
                    "solver": "sgd_l2",
                    "alpha": pixel_alpha,
                    "epochs": pixel_epochs,
                    "batch_size": pixel_batch_size,
                    "eta0": pixel_eta0,
                    "power_t": pixel_power_t,
                },
                "residual": {"ridge_alpha": ridge_alpha, "quantiles_b0": quantiles_b0, "n_perms": n_perms},
                "topo_names": topo_names,
                "t_scalar": {"mean": t_mu, "std": t_sd},
                "s_scale_std_gmag": s_scale,
                "eps_rows": eps_rows,
                "eps_detection_threshold": eps_det,
            }

            # Free big arrays before next w (helps memory).
            del Xpix_by_field, B_by_field, Tmulti_by_field, t_scalar_by_field, y_by_field, u_by_field, tz_by_field, fold_cache

        # Summary markdown
        md_parts: list[str] = []
        md_parts.append("# E21 — Injection power study (LOFO)\n")
        md_parts.append(f"- run: `{paths.run_dir}`")
        md_parts.append(f"- grid_size={grid_size}, alpha={alpha}, k0_frac={k0_frac}, n_fields={n_fields}, patches_per_field={patches_per_field}")
        md_parts.append(f"- eps_list={eps_list}")
        md_parts.append("")

        for w in ws:
            res = results_by_w[w]
            md_parts.append(f"## w={w}")
            md_parts.append(f"- detection threshold (smallest ε that PASS): `{fmt(float(res['eps_detection_threshold']))}`")
            md_parts.append("")

            rows = []
            for r0 in res["eps_rows"]:
                rows.append(
                    {
                        "ε": fmt(float(r0["eps"])),
                        "ΔPearson (mean±std)": f"{float(r0['deltaP_mean']):.4f} ± {float(r0['deltaP_std']):.4f}",
                        "ΔrelRMSE (mean±std)": f"{float(r0['deltaR_mean']):.4f} ± {float(r0['deltaR_std']):.4f}",
                        "Fisher p": fmt(float(r0["fisher_p_P"])),
                        "#folds ΔP>0": f"{int(r0['n_pos_deltaP'])}/{n_fields}",
                        "PASS": "PASS" if bool(r0["verdict_PASS"]) else "FAIL",
                    }
                )
            md_parts.append(
                md_table(rows, ["ε", "ΔPearson (mean±std)", "ΔrelRMSE (mean±std)", "Fisher p", "#folds ΔP>0", "PASS"])
            )
            md_parts.append("")

        summary_md = "\n".join(md_parts) + "\n"
        (paths.run_dir / "summary_e21_injection_power.md").write_text(summary_md, encoding="utf-8")

        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "grid_size": grid_size,
                "alpha": alpha,
                "k0_frac": k0_frac,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "ws": ws,
                "eps_list": eps_list,
                "pixel": {
                    "solver": "sgd_l2",
                    "alpha": pixel_alpha,
                    "epochs": pixel_epochs,
                    "batch_size": pixel_batch_size,
                    "eta0": pixel_eta0,
                    "power_t": pixel_power_t,
                },
                "residual": {"ridge_alpha": ridge_alpha, "quantiles_b0": quantiles_b0, "n_perms": n_perms},
                "results_by_w": results_by_w,
            },
        )
        return paths

    if experiment == "e21b":
        # Injection power study (scalar detection): inject topo signal into gx only, test detectability after pixels baseline.
        from numpy.lib.stride_tricks import sliding_window_view
        from scipy import stats
        from sklearn.linear_model import SGDRegressor

        grid_size = int(cfg.get("grid_size", 256))
        alpha = float(cfg.get("alpha", 2.0))
        k0_frac = float(cfg.get("k0_frac", 0.15))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 10_000))
        ws = [int(x) for x in cfg.get("ws", [17, 33])]
        eps_list = [float(x) for x in cfg.get("eps_list", [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])]

        ridge_alpha = float(cfg.get("ridge_alpha", 1.0))
        quantiles_b0 = [float(x) for x in cfg.get("quantiles_b0", [0.6, 0.7, 0.8, 0.9])]
        n_perms = int(cfg.get("n_perms", 200))

        pixel_epochs = int(cfg.get("pixel_epochs", 2))
        pixel_batch_size = int(cfg.get("pixel_batch_size", 4096))
        pixel_alpha = float(cfg.get("pixel_alpha", 1e-4))
        pixel_eta0 = float(cfg.get("pixel_eta0", 0.01))
        pixel_power_t = float(cfg.get("pixel_power_t", 0.25))

        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if not (0.0 < k0_frac < 0.5):
            raise ValueError("k0_frac must be in (0,0.5)")
        if n_fields < 2:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        if any(w <= 0 or (w % 2) == 0 for w in ws):
            raise ValueError("ws must be odd positive window sizes")
        if any((q <= 0.0) or (q >= 1.0) for q in quantiles_b0):
            raise ValueError("quantiles_b0 must be in (0,1)")
        if n_perms <= 0:
            raise ValueError("n_perms must be > 0")
        if not eps_list:
            raise ValueError("eps_list must be non-empty")

        def fisher_p(ps: list[float]) -> float:
            if not ps:
                return float("nan")
            eps = 1.0 / float(n_perms + 1)
            ps2 = [min(1.0, max(float(p), eps)) for p in ps]
            stat = -2.0 * float(np.sum(np.log(ps2)))
            return float(stats.chi2.sf(stat, 2 * len(ps2)))

        def fmt(x: float) -> str:
            if not np.isfinite(x):
                return "nan"
            ax = abs(float(x))
            if ax != 0.0 and (ax < 1e-3 or ax >= 1e3):
                return f"{x:.3e}"
            return f"{x:.4f}"

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r in rows:
                out.append("| " + " | ".join(r.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        def safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            am = a - float(a.mean())
            bm = b - float(b.mean())
            denom = float(np.linalg.norm(am) * np.linalg.norm(bm)) + 1e-12
            return float((am @ bm) / denom)

        def rmse_1d(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            return float(np.sqrt(np.mean((a - b) ** 2)))

        def relrmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
            sd = float(np.std(y_true))
            return float(rmse_1d(y_true, y_pred) / sd) if sd > 0 else float("nan")

        def fit_pixel_model(train_fields: list[int], Xpix_by_field: list[np.ndarray], y_by_field: list[np.ndarray], *, rng_perm: np.random.Generator) -> tuple[SGDRegressor, SGDRegressor]:
            reg_gx = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(pixel_alpha),
                learning_rate="invscaling",
                eta0=float(pixel_eta0),
                power_t=float(pixel_power_t),
                max_iter=1,
                tol=None,
                fit_intercept=True,
                random_state=int(placebo_seed) + 0,
            )
            reg_gy = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(pixel_alpha),
                learning_rate="invscaling",
                eta0=float(pixel_eta0),
                power_t=float(pixel_power_t),
                max_iter=1,
                tol=None,
                fit_intercept=True,
                random_state=int(placebo_seed) + 1,
            )
            for _epoch in range(pixel_epochs):
                for fid in train_fields:
                    X = Xpix_by_field[fid]
                    y = y_by_field[fid]
                    perm = rng_perm.permutation(X.shape[0])
                    for start in range(0, X.shape[0], pixel_batch_size):
                        idx = perm[start : start + pixel_batch_size]
                        Xb = X[idx]
                        reg_gx.partial_fit(Xb, y[idx, 0])
                        reg_gy.partial_fit(Xb, y[idx, 1])
            return reg_gx, reg_gy

        def predict_pixel(reg_gx: SGDRegressor, reg_gy: SGDRegressor, X: np.ndarray) -> np.ndarray:
            X = np.asarray(X)
            y_pred = np.empty((X.shape[0], 2), dtype=np.float64)
            for start in range(0, X.shape[0], pixel_batch_size):
                sl = slice(start, min(X.shape[0], start + pixel_batch_size))
                y_pred[sl, 0] = reg_gx.predict(X[sl])
                y_pred[sl, 1] = reg_gy.predict(X[sl])
            return y_pred

        # Field generation (rho_high z-scored per field; target g_high from same split).
        rho_high_by_field: list[np.ndarray] = []
        gx_by_field: list[np.ndarray] = []
        gy_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
            split = band_split_poisson_2d(rho01, k0_frac=k0_frac)
            rho_high = _zscore_field(split.rho_high)
            rho_high_by_field.append(np.asarray(rho_high, dtype=np.float64))
            gx_by_field.append(np.asarray(split.high.gx, dtype=np.float64))
            gy_by_field.append(np.asarray(split.high.gy, dtype=np.float64))

        results_by_w: dict[int, Any] = {}

        for w in ws:
            ps = _require_odd("w", w)
            r = ps // 2

            Xpix_by_field: list[np.ndarray] = []
            B_by_field: list[np.ndarray] = []
            Tmulti_by_field: list[np.ndarray] = []
            t_scalar_by_field: list[np.ndarray] = []
            y_by_field: list[np.ndarray] = []
            tz_by_field: list[np.ndarray] = []
            topo_names: list[str] | None = None

            for field_id in range(n_fields):
                field = rho_high_by_field[field_id]
                gx = gx_by_field[field_id]
                gy = gy_by_field[field_id]

                thr = [float(np.quantile(field.reshape(-1), q)) for q in quantiles_b0]
                rng_samp = np.random.default_rng(seed + 10_000 * field_id + 17 * ps)
                feats, y, centers = _sample_features_2d_fast_topo(
                    field=field,
                    gx=gx,
                    gy=gy,
                    n_patches=patches_per_field,
                    patch_size=ps,
                    topo_mode="quantile_per_field",
                    topo_thresholds=thr,
                    thresholds_pos_sigma=None,
                    thresholds_neg_sigma=None,
                    rng=rng_samp,
                )
                y = np.asarray(y, dtype=np.float64)
                assert y.shape == (patches_per_field, 2)

                cx = centers[:, 0].astype(np.int64)
                cy = centers[:, 1].astype(np.int64)
                win = sliding_window_view(field, (ps, ps))
                Xpix = win[cx - r, cy - r].reshape(patches_per_field, -1).astype(np.float32, copy=False)

                B_names = ["mass", "mass2", "var", "max", "grad_energy"]
                B = np.column_stack([np.asarray(feats[n], dtype=np.float64) for n in B_names])

                tnames = sorted([k for k in feats.keys() if k.startswith("b0_")])
                if topo_names is None:
                    topo_names = tnames
                elif topo_names != tnames:
                    raise RuntimeError(f"Topo feature names mismatch across fields: {topo_names} vs {tnames}")
                Tmulti = np.column_stack([np.asarray(feats[n], dtype=np.float64) for n in topo_names])
                t_scalar = np.mean(Tmulti, axis=1).astype(np.float64, copy=False)

                Xpix_by_field.append(Xpix)
                B_by_field.append(B)
                Tmulti_by_field.append(Tmulti)
                t_scalar_by_field.append(t_scalar)
                y_by_field.append(y)

            if topo_names is None:
                raise RuntimeError("No topo features found")

            # Global scaling for injection: s = std(gx_high) over all samples.
            gx_all = np.concatenate([y[:, 0] for y in y_by_field], axis=0)
            s_scale = float(np.std(gx_all))

            # Global tz for scalar topo t.
            t_all = np.concatenate(t_scalar_by_field, axis=0)
            t_mu = float(np.mean(t_all))
            t_sd = float(np.std(t_all))
            if t_sd <= 0:
                t_sd = 1.0
            tz_by_field = [(t - t_mu) / t_sd for t in t_scalar_by_field]

            # Fold cache for residual models (features only; reused across eps).
            fold_cache: dict[int, dict[str, Any]] = {}
            for test_field in range(n_fields):
                train_fields = [i for i in range(n_fields) if i != test_field]
                Btr = np.concatenate([B_by_field[i] for i in train_fields], axis=0)
                Ttr = np.concatenate([Tmulti_by_field[i] for i in train_fields], axis=0)
                Bte = B_by_field[test_field]
                Tte = Tmulti_by_field[test_field]

                B_mu = Btr.mean(axis=0)
                B_sd = np.where(Btr.std(axis=0) > 0, Btr.std(axis=0), 1.0)
                T_mu = Ttr.mean(axis=0)
                T_sd = np.where(Ttr.std(axis=0) > 0, Ttr.std(axis=0), 1.0)
                Btr_s = (Btr - B_mu) / B_sd
                Bte_s = (Bte - B_mu) / B_sd
                Ttr_s = (Ttr - T_mu) / T_sd
                Tte_s = (Tte - T_mu) / T_sd

                A = Btr_s.T @ Btr_s
                Cmat = Ttr_s.T @ Ttr_s
                D_real = Btr_s.T @ Ttr_s
                XtX_C = np.block([[A, D_real], [D_real.T, Cmat]])

                perms = [rng_placebo.permutation(Ttr_s.shape[0]) for _ in range(n_perms)]
                D_perms = np.stack([Btr_s.T @ Ttr_s[p] for p in perms], axis=0)

                fold_cache[test_field] = {
                    "train_fields": train_fields,
                    "Btr_s": Btr_s,
                    "Bte_s": Bte_s,
                    "Ttr_s": Ttr_s,
                    "Tte_s": Tte_s,
                    "A": A,
                    "Cmat": Cmat,
                    "XtX_C": XtX_C,
                    "perms": perms,
                    "D_perms": D_perms,
                }

            eps_rows: list[dict[str, Any]] = []

            for eps in eps_list:
                # Build injected targets per field (gx only).
                y_total_by_field: list[np.ndarray] = []
                for fid in range(n_fields):
                    y = y_by_field[fid]
                    tz = tz_by_field[fid]
                    y_total = y.copy()
                    y_total[:, 0] = y_total[:, 0] + float(eps) * float(s_scale) * tz
                    y_total_by_field.append(y_total)

                fold_rows: list[dict[str, Any]] = []
                pvals_pearson: list[float] = []
                corr_tz_r_test: list[float] = []

                for test_field in range(n_fields):
                    cache = fold_cache[test_field]
                    train_fields = cache["train_fields"]

                    reg_gx, reg_gy = fit_pixel_model(train_fields, Xpix_by_field, y_total_by_field, rng_perm=rng_placebo)
                    ypred_te = predict_pixel(reg_gx, reg_gy, Xpix_by_field[test_field])

                    # scalar residual on test (gx channel)
                    gx_te = y_total_by_field[test_field][:, 0]
                    gx_pred_te = ypred_te[:, 0]
                    r_te = gx_te - gx_pred_te
                    corr_tz_r_test.append(safe_corr_1d(tz_by_field[test_field], r_te))

                    # build train residual targets (gx only)
                    r_tr_list: list[np.ndarray] = []
                    for fid in train_fields:
                        ypred_tr = predict_pixel(reg_gx, reg_gy, Xpix_by_field[fid])
                        r_tr_list.append(y_total_by_field[fid][:, 0] - ypred_tr[:, 0])
                    r_tr = np.concatenate(r_tr_list, axis=0)

                    Btr_s = cache["Btr_s"]
                    Bte_s = cache["Bte_s"]
                    Ttr_s = cache["Ttr_s"]
                    Tte_s = cache["Tte_s"]

                    r_mu = float(np.mean(r_tr))
                    rc = r_tr - r_mu

                    # Fit B_resid (scalar)
                    A = cache["A"]
                    uB = Btr_s.T @ rc.reshape(-1, 1)
                    wB = np.linalg.solve(A + ridge_alpha * np.eye(A.shape[0]), uB).reshape(-1)
                    rB_te = Bte_s @ wB + r_mu
                    pB = safe_corr_1d(r_te, rB_te)
                    rrB = relrmse_1d(r_te, rB_te)

                    # Fit C_resid (real)
                    XtX_C = cache["XtX_C"]
                    v_real = Ttr_s.T @ rc.reshape(-1, 1)
                    Xty_C = np.vstack([uB, v_real])
                    wC = np.linalg.solve(XtX_C + ridge_alpha * np.eye(XtX_C.shape[0]), Xty_C).reshape(-1)
                    XteC = np.concatenate([Bte_s, Tte_s], axis=1)
                    rC_te = XteC @ wC + r_mu
                    pC = safe_corr_1d(r_te, rC_te)
                    rrC = relrmse_1d(r_te, rC_te)

                    deltaP_real = float(pC - pB)
                    deltaR_real = float(rrC - rrB)

                    # Permutation null on ΔPearson
                    perms = cache["perms"]
                    D_perms = cache["D_perms"]
                    Cmat = cache["Cmat"]
                    null_dP: list[float] = []
                    for pi in range(n_perms):
                        pidx = perms[pi]
                        D = D_perms[pi]
                        v = Ttr_s[pidx].T @ rc.reshape(-1, 1)
                        XtX = np.block([[A, D], [D.T, Cmat]])
                        Xty = np.vstack([uB, v])
                        wP = np.linalg.solve(XtX + ridge_alpha * np.eye(XtX.shape[0]), Xty).reshape(-1)
                        rP_te = XteC @ wP + r_mu
                        pP = safe_corr_1d(r_te, rP_te)
                        null_dP.append(float(pP - pB))

                    null = np.asarray(null_dP, dtype=np.float64)
                    null_mu = float(null.mean())
                    null_sd = float(null.std(ddof=1)) if len(null) > 1 else 0.0
                    zP = float((deltaP_real - null_mu) / null_sd) if null_sd > 0 else float("nan")
                    p_emp = float(np.mean(null >= deltaP_real))
                    pvals_pearson.append(p_emp)

                    fold_rows.append(
                        {
                            "w": int(w),
                            "eps": float(eps),
                            "field_id": int(test_field),
                            "deltaP_real": float(deltaP_real),
                            "deltaR_real": float(deltaR_real),
                            "zP": float(zP),
                            "p_emp_P": float(p_emp),
                            "corr_tz_r_test": float(corr_tz_r_test[-1]),
                        }
                    )

                dp = np.asarray([r["deltaP_real"] for r in fold_rows], dtype=np.float64)
                dr = np.asarray([r["deltaR_real"] for r in fold_rows], dtype=np.float64)
                n_pos = int(np.sum(dp > 0.0))
                fp = fisher_p(pvals_pearson)

                corr_tr = np.asarray(corr_tz_r_test, dtype=np.float64)
                verdict = (float(fp) < 0.05) and (float(dr.mean()) < 0.0) and (n_pos >= 7)

                eps_rows.append(
                    {
                        "w": int(w),
                        "eps": float(eps),
                        "deltaP_mean": float(dp.mean()),
                        "deltaP_std": float(dp.std(ddof=1)) if len(dp) > 1 else 0.0,
                        "deltaR_mean": float(dr.mean()),
                        "deltaR_std": float(dr.std(ddof=1)) if len(dr) > 1 else 0.0,
                        "fisher_p_P": float(fp),
                        "n_pos_deltaP": int(n_pos),
                        "corr_tz_r_test_mean": float(np.mean(corr_tr)),
                        "corr_tz_r_test_std": float(np.std(corr_tr, ddof=1)) if len(corr_tr) > 1 else 0.0,
                        "verdict_PASS": bool(verdict),
                        "folds": fold_rows,
                    }
                )

                header = list(fold_rows[0].keys()) if fold_rows else []
                tag = f"w{w}_eps{str(eps).replace('.','p')}"
                write_csv(paths.run_dir / f"lofo_by_field_{tag}.csv", header, [[r[h] for h in header] for r in fold_rows])

            det = [r for r in eps_rows if bool(r["verdict_PASS"])]
            eps_det = float(min((r["eps"] for r in det), default=float("nan")))

            results_by_w[w] = {
                "w": int(w),
                "grid_size": grid_size,
                "alpha": alpha,
                "k0_frac": k0_frac,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "pixel": {
                    "solver": "sgd_l2",
                    "alpha": pixel_alpha,
                    "epochs": pixel_epochs,
                    "batch_size": pixel_batch_size,
                    "eta0": pixel_eta0,
                    "power_t": pixel_power_t,
                },
                "residual": {"ridge_alpha": ridge_alpha, "quantiles_b0": quantiles_b0, "n_perms": n_perms},
                "topo_names": topo_names,
                "t_scalar": {"mean": t_mu, "std": t_sd},
                "s_scale_std_gx": s_scale,
                "eps_rows": eps_rows,
                "eps_detection_threshold": eps_det,
            }

            del Xpix_by_field, B_by_field, Tmulti_by_field, t_scalar_by_field, y_by_field, tz_by_field, fold_cache

        # Summary markdown
        md_parts: list[str] = []
        md_parts.append("# E21b — Injection power study (scalar gx) (LOFO)\n")
        md_parts.append(f"- run: `{paths.run_dir}`")
        md_parts.append(f"- grid_size={grid_size}, alpha={alpha}, k0_frac={k0_frac}, n_fields={n_fields}, patches_per_field={patches_per_field}")
        md_parts.append(f"- eps_list={eps_list}")
        md_parts.append("")

        for w in ws:
            res = results_by_w[w]
            md_parts.append(f"## w={w}")
            md_parts.append(f"- detection threshold (smallest ε that PASS): `{fmt(float(res['eps_detection_threshold']))}`")
            md_parts.append("")
            rows = []
            for r0 in res["eps_rows"]:
                rows.append(
                    {
                        "ε": fmt(float(r0["eps"])),
                        "ΔPearson mean±std": f"{float(r0['deltaP_mean']):.4f} ± {float(r0['deltaP_std']):.4f}",
                        "ΔrelRMSE mean±std": f"{float(r0['deltaR_mean']):.4f} ± {float(r0['deltaR_std']):.4f}",
                        "Fisher p": fmt(float(r0["fisher_p_P"])),
                        "#folds ΔP>0": f"{int(r0['n_pos_deltaP'])}/{n_fields}",
                        "corr(tz,r)_test": f"{float(r0['corr_tz_r_test_mean']):.4f} ± {float(r0['corr_tz_r_test_std']):.4f}",
                        "PASS": "PASS" if bool(r0["verdict_PASS"]) else "FAIL",
                    }
                )
            md_parts.append(
                md_table(
                    rows,
                    ["ε", "ΔPearson mean±std", "ΔrelRMSE mean±std", "Fisher p", "#folds ΔP>0", "corr(tz,r)_test", "PASS"],
                )
            )
            md_parts.append("")

        summary_md = "\n".join(md_parts) + "\n"
        (paths.run_dir / "summary_e21b_injection_scalar_power.md").write_text(summary_md, encoding="utf-8")

        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "grid_size": grid_size,
                "alpha": alpha,
                "k0_frac": k0_frac,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "ws": ws,
                "eps_list": eps_list,
                "pixel": {
                    "solver": "sgd_l2",
                    "alpha": pixel_alpha,
                    "epochs": pixel_epochs,
                    "batch_size": pixel_batch_size,
                    "eta0": pixel_eta0,
                    "power_t": pixel_power_t,
                },
                "residual": {"ridge_alpha": ridge_alpha, "quantiles_b0": quantiles_b0, "n_perms": n_perms},
                "results_by_w": results_by_w,
            },
        )
        return paths

    if experiment == "e22":
        # Nonlinear topo correction detectability (LOFO) after subtracting pixels-only baseline, for two injection modes.
        from numpy.lib.stride_tricks import sliding_window_view
        from scipy import stats
        from sklearn.linear_model import SGDRegressor

        grid_size = int(cfg.get("grid_size", 256))
        alpha = float(cfg.get("alpha", 2.0))
        k0_frac = float(cfg.get("k0_frac", 0.15))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 20_000))
        ws = [int(x) for x in cfg.get("ws", [17, 33])]
        eps_list = [float(x) for x in cfg.get("eps_list", [0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1])]

        ridge_alpha = float(cfg.get("ridge_alpha", 1.0))
        quantiles_b0 = [float(x) for x in cfg.get("quantiles_b0", [0.6, 0.7, 0.8, 0.9])]
        n_perms = int(cfg.get("n_perms", 200))

        pixel_epochs = int(cfg.get("pixel_epochs", 2))
        pixel_batch_size = int(cfg.get("pixel_batch_size", 4096))
        pixel_alpha = float(cfg.get("pixel_alpha", 1e-4))
        pixel_eta0 = float(cfg.get("pixel_eta0", 0.01))
        pixel_power_t = float(cfg.get("pixel_power_t", 0.25))

        modes = [str(x).lower() for x in cfg.get("modes", ["square", "sign"])]
        if not modes:
            raise ValueError("modes must be non-empty")
        for m in modes:
            if m not in {"square", "sign"}:
                raise ValueError("modes must be subset of: square, sign")

        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if not (0.0 < k0_frac < 0.5):
            raise ValueError("k0_frac must be in (0,0.5)")
        if n_fields < 2:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        if any(w <= 0 or (w % 2) == 0 for w in ws):
            raise ValueError("ws must be odd positive window sizes")
        if any((q <= 0.0) or (q >= 1.0) for q in quantiles_b0):
            raise ValueError("quantiles_b0 must be in (0,1)")
        if n_perms <= 0:
            raise ValueError("n_perms must be > 0")
        if not eps_list:
            raise ValueError("eps_list must be non-empty")

        rng_pixel = np.random.default_rng(int(placebo_seed) + 11_111)
        rng_null = np.random.default_rng(int(placebo_seed) + 22_222)

        def fisher_p(ps: list[float]) -> float:
            if not ps:
                return float("nan")
            eps = 1.0 / float(n_perms + 1)
            ps2 = [min(1.0, max(float(p), eps)) for p in ps]
            stat = -2.0 * float(np.sum(np.log(ps2)))
            return float(stats.chi2.sf(stat, 2 * len(ps2)))

        def fmt(x: float) -> str:
            if not np.isfinite(x):
                return "nan"
            ax = abs(float(x))
            if ax != 0.0 and (ax < 1e-3 or ax >= 1e3):
                return f"{x:.3e}"
            return f"{x:.4f}"

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r in rows:
                out.append("| " + " | ".join(r.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        def safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            am = a - float(a.mean())
            bm = b - float(b.mean())
            denom = float(np.linalg.norm(am) * np.linalg.norm(bm)) + 1e-12
            return float((am @ bm) / denom)

        def rmse_1d(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            return float(np.sqrt(np.mean((a - b) ** 2)))

        def relrmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
            sd = float(np.std(y_true))
            return float(rmse_1d(y_true, y_pred) / sd) if sd > 0 else float("nan")

        def fit_pixel_model(
            train_fields: list[int], Xpix_by_field: list[np.ndarray], y_by_field: list[np.ndarray]
        ) -> tuple[SGDRegressor, SGDRegressor]:
            reg_gx = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(pixel_alpha),
                learning_rate="invscaling",
                eta0=float(pixel_eta0),
                power_t=float(pixel_power_t),
                max_iter=1,
                tol=None,
                fit_intercept=True,
                random_state=int(placebo_seed) + 0,
            )
            reg_gy = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(pixel_alpha),
                learning_rate="invscaling",
                eta0=float(pixel_eta0),
                power_t=float(pixel_power_t),
                max_iter=1,
                tol=None,
                fit_intercept=True,
                random_state=int(placebo_seed) + 1,
            )
            for _epoch in range(pixel_epochs):
                for fid in train_fields:
                    X = Xpix_by_field[fid]
                    y = y_by_field[fid]
                    perm = rng_pixel.permutation(X.shape[0])
                    for start in range(0, X.shape[0], pixel_batch_size):
                        idx = perm[start : start + pixel_batch_size]
                        Xb = X[idx]
                        reg_gx.partial_fit(Xb, y[idx, 0])
                        reg_gy.partial_fit(Xb, y[idx, 1])
            return reg_gx, reg_gy

        def predict_pixel(reg_gx: SGDRegressor, reg_gy: SGDRegressor, X: np.ndarray) -> np.ndarray:
            X = np.asarray(X)
            y_pred = np.empty((X.shape[0], 2), dtype=np.float64)
            for start in range(0, X.shape[0], pixel_batch_size):
                sl = slice(start, min(X.shape[0], start + pixel_batch_size))
                y_pred[sl, 0] = reg_gx.predict(X[sl])
                y_pred[sl, 1] = reg_gy.predict(X[sl])
            return y_pred

        # Field generation (rho_high z-scored per field; target g_high from same split).
        rho_high_by_field: list[np.ndarray] = []
        gx_by_field: list[np.ndarray] = []
        gy_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
            split = band_split_poisson_2d(rho01, k0_frac=k0_frac)
            rho_high = _zscore_field(split.rho_high)
            rho_high_by_field.append(np.asarray(rho_high, dtype=np.float64))
            gx_by_field.append(np.asarray(split.high.gx, dtype=np.float64))
            gy_by_field.append(np.asarray(split.high.gy, dtype=np.float64))

        results: dict[str, Any] = {m: {} for m in modes}

        for w in ws:
            ps = _require_odd("w", w)
            r = ps // 2

            # Build patch datasets for this w.
            Xpix_by_field: list[np.ndarray] = []
            B_by_field: list[np.ndarray] = []
            Tmulti_by_field: list[np.ndarray] = []
            t_scalar_by_field: list[np.ndarray] = []
            y_by_field: list[np.ndarray] = []
            topo_names: list[str] | None = None

            for field_id in range(n_fields):
                field = rho_high_by_field[field_id]
                gx = gx_by_field[field_id]
                gy = gy_by_field[field_id]

                thr = [float(np.quantile(field.reshape(-1), q)) for q in quantiles_b0]
                rng_samp = np.random.default_rng(seed + 10_000 * field_id + 17 * ps)
                feats, y, centers = _sample_features_2d_fast_topo(
                    field=field,
                    gx=gx,
                    gy=gy,
                    n_patches=patches_per_field,
                    patch_size=ps,
                    topo_mode="quantile_per_field",
                    topo_thresholds=thr,
                    thresholds_pos_sigma=None,
                    thresholds_neg_sigma=None,
                    rng=rng_samp,
                )
                y = np.asarray(y, dtype=np.float64)
                assert y.shape == (patches_per_field, 2)

                cx = centers[:, 0].astype(np.int64)
                cy = centers[:, 1].astype(np.int64)
                win = sliding_window_view(field, (ps, ps))
                Xpix = win[cx - r, cy - r].reshape(patches_per_field, -1).astype(np.float32, copy=False)

                B_names = ["mass", "mass2", "var", "max", "grad_energy"]
                B = np.column_stack([np.asarray(feats[n], dtype=np.float64) for n in B_names])

                tnames = sorted([k for k in feats.keys() if k.startswith("b0_")])
                if topo_names is None:
                    topo_names = tnames
                elif topo_names != tnames:
                    raise RuntimeError(f"Topo feature names mismatch across fields: {topo_names} vs {tnames}")
                Tmulti = np.column_stack([np.asarray(feats[n], dtype=np.float64) for n in topo_names])
                t_scalar = np.mean(Tmulti, axis=1).astype(np.float64, copy=False)

                Xpix_by_field.append(Xpix)
                B_by_field.append(B)
                Tmulti_by_field.append(Tmulti)
                t_scalar_by_field.append(t_scalar)
                y_by_field.append(y)

            if topo_names is None:
                raise RuntimeError("No topo features found")

            # Global tz for scalar topo t (over all samples for this w).
            t_all = np.concatenate(t_scalar_by_field, axis=0)
            t_mu = float(np.mean(t_all))
            t_sd = float(np.std(t_all))
            if t_sd <= 0:
                t_sd = 1.0
            tz_by_field = [(t - t_mu) / t_sd for t in t_scalar_by_field]

            # Centered square feature (global centering).
            tz2_all = np.concatenate([tz * tz for tz in tz_by_field], axis=0)
            tz2_mu = float(np.mean(tz2_all))
            tz2c_by_field = [(tz * tz) - tz2_mu for tz in tz_by_field]

            # Global scaling for injection: s = std(gx_high) over all samples.
            gx_all = np.concatenate([y[:, 0] for y in y_by_field], axis=0)
            s_scale = float(np.std(gx_all))

            # Fold cache for residual models (features only; reused across eps and modes).
            fold_cache: dict[int, dict[str, Any]] = {}
            for test_field in range(n_fields):
                train_fields = [i for i in range(n_fields) if i != test_field]
                Btr = np.concatenate([B_by_field[i] for i in train_fields], axis=0)
                Ttr = np.concatenate([Tmulti_by_field[i] for i in train_fields], axis=0)
                Bte = B_by_field[test_field]
                Tte = Tmulti_by_field[test_field]

                B_mu = Btr.mean(axis=0)
                B_sd = np.where(Btr.std(axis=0) > 0, Btr.std(axis=0), 1.0)
                T_mu = Ttr.mean(axis=0)
                T_sd = np.where(Ttr.std(axis=0) > 0, Ttr.std(axis=0), 1.0)
                Btr_s = (Btr - B_mu) / B_sd
                Bte_s = (Bte - B_mu) / B_sd
                Ttr_s = (Ttr - T_mu) / T_sd
                Tte_s = (Tte - T_mu) / T_sd

                A = Btr_s.T @ Btr_s
                Cmat = Ttr_s.T @ Ttr_s
                D_real = Btr_s.T @ Ttr_s
                XtX_C = np.block([[A, D_real], [D_real.T, Cmat]])

                perms = [rng_null.permutation(Ttr_s.shape[0]) for _ in range(n_perms)]
                D_perms = np.stack([Btr_s.T @ Ttr_s[p] for p in perms], axis=0)

                fold_cache[test_field] = {
                    "train_fields": train_fields,
                    "Btr_s": Btr_s,
                    "Bte_s": Bte_s,
                    "Ttr_s": Ttr_s,
                    "Tte_s": Tte_s,
                    "A": A,
                    "Cmat": Cmat,
                    "XtX_C": XtX_C,
                    "perms": perms,
                    "D_perms": D_perms,
                }

            for mode in modes:
                eps_rows: list[dict[str, Any]] = []

                for eps in eps_list:
                    # Build injected targets per field (gx only).
                    y_total_by_field: list[np.ndarray] = []
                    for fid in range(n_fields):
                        y = y_by_field[fid]
                        tz = tz_by_field[fid]
                        tz2c = tz2c_by_field[fid]
                        y_total = y.copy()
                        if mode == "square":
                            inj = tz2c
                        else:
                            inj = np.sign(tz)
                        y_total[:, 0] = y_total[:, 0] + float(eps) * float(s_scale) * inj
                        y_total_by_field.append(y_total)

                    fold_rows: list[dict[str, Any]] = []
                    pvals_pearson: list[float] = []
                    sanity_corrs: list[float] = []
                    leak_corrs: list[float] = []

                    for test_field in range(n_fields):
                        cache = fold_cache[test_field]
                        train_fields = cache["train_fields"]

                        reg_gx, reg_gy = fit_pixel_model(train_fields, Xpix_by_field, y_total_by_field)
                        ypred_te = predict_pixel(reg_gx, reg_gy, Xpix_by_field[test_field])

                        gx_te = y_total_by_field[test_field][:, 0]
                        gx_pred_te = ypred_te[:, 0]
                        r_te = gx_te - gx_pred_te

                        if mode == "square":
                            sanity_feat = tz2c_by_field[test_field]
                        else:
                            sanity_feat = tz_by_field[test_field]
                        sanity = safe_corr_1d(sanity_feat, r_te)
                        leak = safe_corr_1d(sanity_feat, gx_pred_te)
                        sanity_corrs.append(sanity)
                        leak_corrs.append(leak)

                        # build train residual targets (gx only)
                        r_tr_list: list[np.ndarray] = []
                        for fid in train_fields:
                            ypred_tr = predict_pixel(reg_gx, reg_gy, Xpix_by_field[fid])
                            r_tr_list.append(y_total_by_field[fid][:, 0] - ypred_tr[:, 0])
                        r_tr = np.concatenate(r_tr_list, axis=0)

                        Btr_s = cache["Btr_s"]
                        Bte_s = cache["Bte_s"]
                        Ttr_s = cache["Ttr_s"]
                        Tte_s = cache["Tte_s"]

                        r_mu = float(np.mean(r_tr))
                        rc = r_tr - r_mu

                        # Fit B_resid (scalar)
                        A = cache["A"]
                        uB = Btr_s.T @ rc.reshape(-1, 1)
                        wB = np.linalg.solve(A + ridge_alpha * np.eye(A.shape[0]), uB).reshape(-1)
                        rB_te = Bte_s @ wB + r_mu
                        pB = safe_corr_1d(r_te, rB_te)
                        rrB = relrmse_1d(r_te, rB_te)

                        # Fit C_resid (real)
                        XtX_C = cache["XtX_C"]
                        v_real = Ttr_s.T @ rc.reshape(-1, 1)
                        Xty_C = np.vstack([uB, v_real])
                        wC = np.linalg.solve(XtX_C + ridge_alpha * np.eye(XtX_C.shape[0]), Xty_C).reshape(-1)
                        XteC = np.concatenate([Bte_s, Tte_s], axis=1)
                        rC_te = XteC @ wC + r_mu
                        pC = safe_corr_1d(r_te, rC_te)
                        rrC = relrmse_1d(r_te, rC_te)

                        deltaP_real = float(pC - pB)
                        deltaR_real = float(rrC - rrB)

                        # Permutation null on ΔPearson
                        perms = cache["perms"]
                        D_perms = cache["D_perms"]
                        Cmat = cache["Cmat"]
                        null_dP: list[float] = []
                        for pi in range(n_perms):
                            pidx = perms[pi]
                            D = D_perms[pi]
                            v = Ttr_s[pidx].T @ rc.reshape(-1, 1)
                            XtX = np.block([[A, D], [D.T, Cmat]])
                            Xty = np.vstack([uB, v])
                            wP = np.linalg.solve(XtX + ridge_alpha * np.eye(XtX.shape[0]), Xty).reshape(-1)
                            rP_te = XteC @ wP + r_mu
                            pP = safe_corr_1d(r_te, rP_te)
                            null_dP.append(float(pP - pB))

                        null = np.asarray(null_dP, dtype=np.float64)
                        null_mu = float(null.mean())
                        null_sd = float(null.std(ddof=1)) if len(null) > 1 else 0.0
                        zP = float((deltaP_real - null_mu) / null_sd) if null_sd > 0 else float("nan")
                        p_emp = float(np.mean(null >= deltaP_real))
                        pvals_pearson.append(p_emp)

                        fold_rows.append(
                            {
                                "mode": mode,
                                "w": int(w),
                                "eps": float(eps),
                                "field_id": int(test_field),
                                "deltaP_real": float(deltaP_real),
                                "deltaR_real": float(deltaR_real),
                                "zP": float(zP),
                                "p_emp_P": float(p_emp),
                                "corr_sanity_test": float(sanity),
                                "corr_sanity_pred": float(leak),
                            }
                        )

                    dp = np.asarray([r["deltaP_real"] for r in fold_rows], dtype=np.float64)
                    dr = np.asarray([r["deltaR_real"] for r in fold_rows], dtype=np.float64)
                    n_pos = int(np.sum(dp > 0.0))
                    fp = fisher_p(pvals_pearson)
                    verdict = (float(fp) < 0.05) and (float(dr.mean()) < 0.0) and (n_pos >= 7)

                    eps_rows.append(
                        {
                            "mode": mode,
                            "w": int(w),
                            "eps": float(eps),
                            "deltaP_mean": float(dp.mean()),
                            "deltaP_std": float(dp.std(ddof=1)) if len(dp) > 1 else 0.0,
                            "deltaR_mean": float(dr.mean()),
                            "deltaR_std": float(dr.std(ddof=1)) if len(dr) > 1 else 0.0,
                            "fisher_p_P": float(fp),
                            "n_pos_deltaP": int(n_pos),
                            "corr_sanity_test_mean": float(np.mean(np.asarray(sanity_corrs, dtype=np.float64))),
                            "corr_sanity_test_std": float(np.std(np.asarray(sanity_corrs, dtype=np.float64), ddof=1))
                            if len(sanity_corrs) > 1
                            else 0.0,
                            "corr_sanity_pred_mean": float(np.mean(np.asarray(leak_corrs, dtype=np.float64))),
                            "corr_sanity_pred_std": float(np.std(np.asarray(leak_corrs, dtype=np.float64), ddof=1))
                            if len(leak_corrs) > 1
                            else 0.0,
                            "verdict_PASS": bool(verdict),
                            "folds": fold_rows,
                        }
                    )

                    header = list(fold_rows[0].keys()) if fold_rows else []
                    tag = f"{mode}_w{w}_eps{str(eps).replace('.','p')}"
                    write_csv(paths.run_dir / f"lofo_by_field_{tag}.csv", header, [[r[h] for h in header] for r in fold_rows])

                det = [r for r in eps_rows if bool(r["verdict_PASS"])]
                eps_det = float(min((r["eps"] for r in det), default=float("nan")))

                results[mode][w] = {
                    "mode": mode,
                    "w": int(w),
                    "grid_size": grid_size,
                    "alpha": alpha,
                    "k0_frac": k0_frac,
                    "n_fields": n_fields,
                    "patches_per_field": patches_per_field,
                    "pixel": {
                        "solver": "sgd_l2",
                        "alpha": pixel_alpha,
                        "epochs": pixel_epochs,
                        "batch_size": pixel_batch_size,
                        "eta0": pixel_eta0,
                        "power_t": pixel_power_t,
                    },
                    "residual": {"ridge_alpha": ridge_alpha, "quantiles_b0": quantiles_b0, "n_perms": n_perms},
                    "topo_names": topo_names,
                    "tz_stats": {"mean": t_mu, "std": t_sd},
                    "tz2_mean": tz2_mu,
                    "s_scale_std_gx": s_scale,
                    "eps_rows": eps_rows,
                    "eps_detection_threshold": eps_det,
                }

            del Xpix_by_field, B_by_field, Tmulti_by_field, t_scalar_by_field, y_by_field, tz_by_field, tz2c_by_field, fold_cache

        # Summary markdown
        md_parts: list[str] = []
        md_parts.append("# E22 — Nonlinear injection detectability (LOFO)\n")
        md_parts.append(f"- run: `{paths.run_dir}`")
        md_parts.append(f"- grid_size={grid_size}, alpha={alpha}, k0_frac={k0_frac}, n_fields={n_fields}, patches_per_field={patches_per_field}")
        md_parts.append(f"- ws={ws} eps_list={eps_list} modes={modes}")
        md_parts.append("")

        for mode in modes:
            md_parts.append(f"## Mode: `{mode}`")
            md_parts.append("")
            for w in ws:
                res = results[mode][w]
                det = fmt(float(res["eps_detection_threshold"]))
                md_parts.append(f"### w={w}  detection threshold ε: `{det}`")
                rows = []
                for r0 in res["eps_rows"]:
                    corr_label = "corr(tz2c,r)_test" if mode == "square" else "corr(tz,r)_test"
                    rows.append(
                        {
                            "ε": fmt(float(r0["eps"])),
                            "ΔPearson mean±std": f"{float(r0['deltaP_mean']):.4f} ± {float(r0['deltaP_std']):.4f}",
                            "ΔrelRMSE mean±std": f"{float(r0['deltaR_mean']):.4f} ± {float(r0['deltaR_std']):.4f}",
                            "Fisher p": fmt(float(r0["fisher_p_P"])),
                            "#folds ΔP>0": f"{int(r0['n_pos_deltaP'])}/{n_fields}",
                            corr_label: f"{float(r0['corr_sanity_test_mean']):.4f} ± {float(r0['corr_sanity_test_std']):.4f}",
                            "corr(sanity,gx_pred)": f"{float(r0['corr_sanity_pred_mean']):.4f} ± {float(r0['corr_sanity_pred_std']):.4f}",
                            "PASS": "PASS" if bool(r0["verdict_PASS"]) else "FAIL",
                        }
                    )
                cols = ["ε", "ΔPearson mean±std", "ΔrelRMSE mean±std", "Fisher p", "#folds ΔP>0"]
                cols.append("corr(tz2c,r)_test" if mode == "square" else "corr(tz,r)_test")
                cols.append("corr(sanity,gx_pred)")
                cols.append("PASS")
                md_parts.append(md_table(rows, cols))
                md_parts.append("")

        summary_md = "\n".join(md_parts) + "\n"
        (paths.run_dir / "summary_e22_nonlinear_injection_power.md").write_text(summary_md, encoding="utf-8")

        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "grid_size": grid_size,
                "alpha": alpha,
                "k0_frac": k0_frac,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "ws": ws,
                "eps_list": eps_list,
                "modes": modes,
                "pixel": {
                    "solver": "sgd_l2",
                    "alpha": pixel_alpha,
                    "epochs": pixel_epochs,
                    "batch_size": pixel_batch_size,
                    "eta0": pixel_eta0,
                    "power_t": pixel_power_t,
                },
                "residual": {"ridge_alpha": ridge_alpha, "quantiles_b0": quantiles_b0, "n_perms": n_perms},
                "results": results,
            },
        )
        return paths

    if experiment == "e23":
        # Fix Mode A failure (tz^2 centered) by adding even/nonlinear topo transforms in the residual model (LOFO).
        from numpy.lib.stride_tricks import sliding_window_view
        from scipy import stats
        from sklearn.linear_model import SGDRegressor

        grid_size = int(cfg.get("grid_size", 256))
        alpha = float(cfg.get("alpha", 2.0))
        k0_frac = float(cfg.get("k0_frac", 0.15))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 20_000))
        w = _require_odd("window_size", int(cfg.get("window_size", 33)))
        eps_list = [float(x) for x in cfg.get("eps_list", [0.0, 0.0025, 0.005, 0.01, 0.02, 0.05])]

        ridge_alpha = float(cfg.get("ridge_alpha", 1.0))
        quantiles_b0 = [float(x) for x in cfg.get("quantiles_b0", [0.6, 0.7, 0.8, 0.9])]
        n_perms = int(cfg.get("n_perms", 200))

        pixel_epochs = int(cfg.get("pixel_epochs", 2))
        pixel_batch_size = int(cfg.get("pixel_batch_size", 4096))
        pixel_alpha = float(cfg.get("pixel_alpha", 1e-4))
        pixel_eta0 = float(cfg.get("pixel_eta0", 0.01))
        pixel_power_t = float(cfg.get("pixel_power_t", 0.25))

        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if not (0.0 < k0_frac < 0.5):
            raise ValueError("k0_frac must be in (0,0.5)")
        if n_fields < 2:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        if any((q <= 0.0) or (q >= 1.0) for q in quantiles_b0):
            raise ValueError("quantiles_b0 must be in (0,1)")
        if n_perms <= 0:
            raise ValueError("n_perms must be > 0")
        if not eps_list:
            raise ValueError("eps_list must be non-empty")

        rng_pixel = np.random.default_rng(int(placebo_seed) + 33_333)
        rng_null = np.random.default_rng(int(placebo_seed) + 44_444)

        def fisher_p(ps: list[float]) -> float:
            if not ps:
                return float("nan")
            eps = 1.0 / float(n_perms + 1)
            ps2 = [min(1.0, max(float(p), eps)) for p in ps]
            stat = -2.0 * float(np.sum(np.log(ps2)))
            return float(stats.chi2.sf(stat, 2 * len(ps2)))

        def fmt(x: float) -> str:
            if not np.isfinite(x):
                return "nan"
            ax = abs(float(x))
            if ax != 0.0 and (ax < 1e-3 or ax >= 1e3):
                return f"{x:.3e}"
            return f"{x:.4f}"

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r0 in rows:
                out.append("| " + " | ".join(r0.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        def safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            am = a - float(a.mean())
            bm = b - float(b.mean())
            denom = float(np.linalg.norm(am) * np.linalg.norm(bm)) + 1e-12
            return float((am @ bm) / denom)

        def relrmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
            rm = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            sd = float(np.std(y_true))
            return float(rm / sd) if sd > 0 else float("nan")

        def fit_pixel_model(train_fields: list[int], Xpix_by_field: list[np.ndarray], y_by_field: list[np.ndarray]) -> tuple[SGDRegressor, SGDRegressor]:
            reg_gx = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(pixel_alpha),
                learning_rate="invscaling",
                eta0=float(pixel_eta0),
                power_t=float(pixel_power_t),
                max_iter=1,
                tol=None,
                fit_intercept=True,
                random_state=int(placebo_seed) + 0,
            )
            reg_gy = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(pixel_alpha),
                learning_rate="invscaling",
                eta0=float(pixel_eta0),
                power_t=float(pixel_power_t),
                max_iter=1,
                tol=None,
                fit_intercept=True,
                random_state=int(placebo_seed) + 1,
            )
            for _epoch in range(pixel_epochs):
                for fid in train_fields:
                    X = Xpix_by_field[fid]
                    y = y_by_field[fid]
                    perm = rng_pixel.permutation(X.shape[0])
                    for start in range(0, X.shape[0], pixel_batch_size):
                        idx = perm[start : start + pixel_batch_size]
                        Xb = X[idx]
                        reg_gx.partial_fit(Xb, y[idx, 0])
                        reg_gy.partial_fit(Xb, y[idx, 1])
            return reg_gx, reg_gy

        def predict_pixel(reg_gx: SGDRegressor, reg_gy: SGDRegressor, X: np.ndarray) -> np.ndarray:
            X = np.asarray(X)
            y_pred = np.empty((X.shape[0], 2), dtype=np.float64)
            for start in range(0, X.shape[0], pixel_batch_size):
                sl = slice(start, min(X.shape[0], start + pixel_batch_size))
                y_pred[sl, 0] = reg_gx.predict(X[sl])
                y_pred[sl, 1] = reg_gy.predict(X[sl])
            return y_pred

        # Field generation (rho_high z-scored per field).
        rho_high_by_field: list[np.ndarray] = []
        gx_by_field: list[np.ndarray] = []
        gy_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
            split = band_split_poisson_2d(rho01, k0_frac=k0_frac)
            rho_high_by_field.append(np.asarray(_zscore_field(split.rho_high), dtype=np.float64))
            gx_by_field.append(np.asarray(split.high.gx, dtype=np.float64))
            gy_by_field.append(np.asarray(split.high.gy, dtype=np.float64))

        ps = w
        r = ps // 2

        # Build patch datasets for w=33.
        Xpix_by_field: list[np.ndarray] = []
        B_by_field: list[np.ndarray] = []
        T_by_field: list[np.ndarray] = []
        t_scalar_by_field: list[np.ndarray] = []
        y_by_field: list[np.ndarray] = []
        topo_names: list[str] | None = None

        for field_id in range(n_fields):
            field = rho_high_by_field[field_id]
            gx = gx_by_field[field_id]
            gy = gy_by_field[field_id]

            thr = [float(np.quantile(field.reshape(-1), q)) for q in quantiles_b0]
            rng_samp = np.random.default_rng(seed + 10_000 * field_id + 17 * ps)
            feats, y, centers = _sample_features_2d_fast_topo(
                field=field,
                gx=gx,
                gy=gy,
                n_patches=patches_per_field,
                patch_size=ps,
                topo_mode="quantile_per_field",
                topo_thresholds=thr,
                thresholds_pos_sigma=None,
                thresholds_neg_sigma=None,
                rng=rng_samp,
            )
            y = np.asarray(y, dtype=np.float64)
            assert y.shape == (patches_per_field, 2)

            cx = centers[:, 0].astype(np.int64)
            cy = centers[:, 1].astype(np.int64)
            win = sliding_window_view(field, (ps, ps))
            Xpix = win[cx - r, cy - r].reshape(patches_per_field, -1).astype(np.float32, copy=False)

            B_names = ["mass", "mass2", "var", "max", "grad_energy"]
            B = np.column_stack([np.asarray(feats[n], dtype=np.float64) for n in B_names])

            tnames = sorted([k for k in feats.keys() if k.startswith("b0_")])
            if topo_names is None:
                topo_names = tnames
            elif topo_names != tnames:
                raise RuntimeError(f"Topo feature names mismatch across fields: {topo_names} vs {tnames}")
            T = np.column_stack([np.asarray(feats[n], dtype=np.float64) for n in topo_names])
            t_scalar = np.mean(T, axis=1).astype(np.float64, copy=False)

            Xpix_by_field.append(Xpix)
            B_by_field.append(B)
            T_by_field.append(T)
            t_scalar_by_field.append(t_scalar)
            y_by_field.append(y)

        if topo_names is None:
            raise RuntimeError("No topo features found")

        # Injection feature tz and tz2c are computed globally over ALL samples (matches E22).
        t_all = np.concatenate(t_scalar_by_field, axis=0)
        t_mu_global = float(np.mean(t_all))
        t_sd_global = float(np.std(t_all))
        if t_sd_global <= 0:
            t_sd_global = 1.0
        tz_by_field = [(t - t_mu_global) / t_sd_global for t in t_scalar_by_field]
        tz2_all = np.concatenate([tz * tz for tz in tz_by_field], axis=0)
        tz2_mu_global = float(np.mean(tz2_all))
        tz2c_by_field = [(tz * tz) - tz2_mu_global for tz in tz_by_field]

        gx_all = np.concatenate([y[:, 0] for y in y_by_field], axis=0)
        s_scale = float(np.std(gx_all))

        # Define residual feature builders (fold-safe).
        def build_F(train_fields: list[int], test_field: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
            Ttr_raw = np.concatenate([T_by_field[i] for i in train_fields], axis=0)
            Tte_raw = T_by_field[test_field]
            if kind == "C1":
                return Ttr_raw, Tte_raw
            if kind == "C2":
                return np.concatenate([Ttr_raw, Ttr_raw * Ttr_raw], axis=1), np.concatenate([Tte_raw, Tte_raw * Tte_raw], axis=1)
            if kind == "C3":
                mu = Ttr_raw.mean(axis=0, keepdims=True)
                return np.concatenate([Ttr_raw, np.abs(Ttr_raw - mu)], axis=1), np.concatenate([Tte_raw, np.abs(Tte_raw - mu)], axis=1)
            if kind == "C4":
                # Fold-safe tz and tz^2 centered from scalar t computed on train only.
                ttr = np.mean(Ttr_raw, axis=1)
                tte = np.mean(Tte_raw, axis=1)
                mu = float(np.mean(ttr))
                sd = float(np.std(ttr))
                if sd <= 0:
                    sd = 1.0
                tz_tr = (ttr - mu) / sd
                tz_te = (tte - mu) / sd
                tz2_mu = float(np.mean(tz_tr * tz_tr))
                tz2c_tr = (tz_tr * tz_tr) - tz2_mu
                tz2c_te = (tz_te * tz_te) - tz2_mu
                return np.column_stack([tz_tr, tz2c_tr]), np.column_stack([tz_te, tz2c_te])
            raise ValueError(f"Unknown kind: {kind}")

        # Main loop over eps
        ckinds = ["C1", "C2", "C3", "C4"]
        rows_by_ck: dict[str, list[dict[str, Any]]] = {k: [] for k in ckinds}

        for eps in eps_list:
            # Build injected targets per field (Mode A only: tz^2 centered global).
            y_total_by_field: list[np.ndarray] = []
            for fid in range(n_fields):
                y = y_by_field[fid]
                tz2c = tz2c_by_field[fid]
                y_total = y.copy()
                y_total[:, 0] = y_total[:, 0] + float(eps) * float(s_scale) * tz2c
                y_total_by_field.append(y_total)

            # Per fold: train pixels baseline, compute residual r on gx, and evaluate models.
            fold_cache: dict[int, dict[str, Any]] = {}
            for test_field in range(n_fields):
                train_fields = [i for i in range(n_fields) if i != test_field]

                reg_gx, reg_gy = fit_pixel_model(train_fields, Xpix_by_field, y_total_by_field)
                ypred_te = predict_pixel(reg_gx, reg_gy, Xpix_by_field[test_field])

                gx_te = y_total_by_field[test_field][:, 0]
                gx_pred_te = ypred_te[:, 0]
                r_te = gx_te - gx_pred_te

                # Train residual targets
                r_tr_list: list[np.ndarray] = []
                for fid in train_fields:
                    ypred_tr = predict_pixel(reg_gx, reg_gy, Xpix_by_field[fid])
                    r_tr_list.append(y_total_by_field[fid][:, 0] - ypred_tr[:, 0])
                r_tr = np.concatenate(r_tr_list, axis=0)

                # Standardize B on train
                Btr = np.concatenate([B_by_field[i] for i in train_fields], axis=0)
                Bte = B_by_field[test_field]
                B_mu = Btr.mean(axis=0)
                B_sd = np.where(Btr.std(axis=0) > 0, Btr.std(axis=0), 1.0)
                Btr_s = (Btr - B_mu) / B_sd
                Bte_s = (Bte - B_mu) / B_sd

                r_mu = float(np.mean(r_tr))
                rc = (r_tr - r_mu).reshape(-1, 1)

                # Fit B_resid once.
                A = Btr_s.T @ Btr_s
                uB = Btr_s.T @ rc
                wB = np.linalg.solve(A + ridge_alpha * np.eye(A.shape[0]), uB).reshape(-1)
                rB_te = Bte_s @ wB + r_mu
                pB = safe_corr_1d(r_te, rB_te)
                rrB = relrmse_1d(r_te, rB_te)

                # Sanity: corr(tz2c,r)_test uses the injection tz2c (global) on the test fold.
                sanity_corr = safe_corr_1d(tz2c_by_field[test_field], r_te)

                fold_cache[test_field] = {
                    "train_fields": train_fields,
                    "r_te": r_te,
                    "pB": pB,
                    "rrB": rrB,
                    "Btr_s": Btr_s,
                    "Bte_s": Bte_s,
                    "rc": rc,
                    "r_mu": r_mu,
                    "A": A,
                    "uB": uB,
                    "sanity_corr": sanity_corr,
                }

            # For each Ck, evaluate real deltas and permutation p-values.
            for kind in ckinds:
                deltasP: list[float] = []
                deltasR: list[float] = []
                sanity_fold: list[float] = []
                pvals_fold: list[float] = []

                for test_field in range(n_fields):
                    fc = fold_cache[test_field]
                    train_fields = fc["train_fields"]

                    Btr_s = fc["Btr_s"]
                    Bte_s = fc["Bte_s"]
                    rc = fc["rc"]
                    r_mu = fc["r_mu"]
                    r_te = fc["r_te"]
                    pB = float(fc["pB"])
                    rrB = float(fc["rrB"])
                    sanity_fold.append(float(fc["sanity_corr"]))

                    Ftr_raw, Fte_raw = build_F(train_fields, test_field, kind)
                    Ftr_raw = np.asarray(Ftr_raw, dtype=np.float64)
                    Fte_raw = np.asarray(Fte_raw, dtype=np.float64)

                    # Standardize F on train (perm invariant).
                    F_mu = Ftr_raw.mean(axis=0)
                    F_sd = np.where(Ftr_raw.std(axis=0) > 0, Ftr_raw.std(axis=0), 1.0)
                    Ftr_s = (Ftr_raw - F_mu) / F_sd
                    Fte_s = (Fte_raw - F_mu) / F_sd

                    # Real C fit
                    XtrC = np.concatenate([Btr_s, Ftr_s], axis=1)
                    XteC = np.concatenate([Bte_s, Fte_s], axis=1)
                    XtX = XtrC.T @ XtrC
                    Xty = XtrC.T @ rc
                    wC = np.linalg.solve(XtX + ridge_alpha * np.eye(XtX.shape[0]), Xty).reshape(-1)
                    rC_te = XteC @ wC + r_mu
                    pC = safe_corr_1d(r_te, rC_te)
                    rrC = relrmse_1d(r_te, rC_te)
                    deltaP_real = float(pC - pB)
                    deltaR_real = float(rrC - rrB)

                    deltasP.append(deltaP_real)
                    deltasR.append(deltaR_real)

                    # Permutation null for ΔPearson: shuffle topo-derived block within train.
                    # Use standardized blocks and block-matrix solves to avoid re-standardization.
                    A = fc["A"]
                    uB = fc["uB"]
                    Cmat = Ftr_s.T @ Ftr_s
                    Z = np.concatenate([Btr_s, rc], axis=1)  # (n, pB+1)

                    null_dP: list[float] = []
                    # Pre-generate permutations for this fold/kind to keep reproducible.
                    perms = [rng_null.permutation(Ftr_s.shape[0]) for _ in range(n_perms)]
                    for pidx in perms:
                        Fp = Ftr_s[pidx]
                        M = Z.T @ Fp  # (pB+1, pF)
                        D = M[:-1, :]  # (pB,pF)
                        v = M[-1, :].reshape(-1, 1)  # (pF,1) = F^T rc
                        XtX_p = np.block([[A, D], [D.T, Cmat]])
                        Xty_p = np.vstack([uB, v])
                        wP = np.linalg.solve(XtX_p + ridge_alpha * np.eye(XtX_p.shape[0]), Xty_p).reshape(-1)
                        rP_te = XteC @ wP + r_mu
                        pP = safe_corr_1d(r_te, rP_te)
                        null_dP.append(float(pP - pB))

                    null = np.asarray(null_dP, dtype=np.float64)
                    p_emp = float(np.mean(null >= deltaP_real))
                    pvals_fold.append(p_emp)

                dp = np.asarray(deltasP, dtype=np.float64)
                dr = np.asarray(deltasR, dtype=np.float64)
                n_pos = int(np.sum(dp > 0.0))
                fp = fisher_p(pvals_fold)
                verdict = (float(fp) < 0.05) and (float(dr.mean()) < 0.0) and (n_pos >= 7)

                rows_by_ck[kind].append(
                    {
                        "eps": float(eps),
                        "deltaP_mean": float(dp.mean()),
                        "deltaP_std": float(dp.std(ddof=1)) if len(dp) > 1 else 0.0,
                        "deltaR_mean": float(dr.mean()),
                        "deltaR_std": float(dr.std(ddof=1)) if len(dr) > 1 else 0.0,
                        "fisher_p_P": float(fp),
                        "n_pos_deltaP": int(n_pos),
                        "corr_tz2c_r_test_mean": float(np.mean(np.asarray(sanity_fold, dtype=np.float64))),
                        "corr_tz2c_r_test_std": float(np.std(np.asarray(sanity_fold, dtype=np.float64), ddof=1))
                        if len(sanity_fold) > 1
                        else 0.0,
                        "verdict_PASS": bool(verdict),
                    }
                )

        # Detection thresholds per Ck
        det_by_ck: dict[str, float] = {}
        for kind in ckinds:
            passed = [r0 for r0 in rows_by_ck[kind] if bool(r0["verdict_PASS"])]
            det_by_ck[kind] = float(min((r0["eps"] for r0 in passed), default=float("nan")))

        best_kind = min(ckinds, key=lambda k: (np.inf if not np.isfinite(det_by_ck[k]) else det_by_ck[k]))

        # Summary markdown
        md_parts: list[str] = []
        md_parts.append("# E23 — Mode A (tz^2 centered) with nonlinear topo residual features (LOFO)\n")
        md_parts.append(f"- run: `{paths.run_dir}`")
        md_parts.append(
            f"- grid_size={grid_size}, alpha={alpha}, k0_frac={k0_frac}, n_fields={n_fields}, patches_per_field={patches_per_field}, w={w}"
        )
        md_parts.append(f"- eps_list={eps_list}")
        md_parts.append("- detection thresholds: " + ", ".join(f"{k}={fmt(v)}" for k, v in det_by_ck.items()))
        md_parts.append(f"- best (smallest ε): `{best_kind}`")
        md_parts.append("")

        for kind in ckinds:
            md_parts.append(f"## {kind}  (threshold ε: `{fmt(det_by_ck[kind])}`)")
            rows = []
            for r0 in rows_by_ck[kind]:
                rows.append(
                    {
                        "ε": fmt(float(r0["eps"])),
                        "ΔPearson mean±std": f"{float(r0['deltaP_mean']):.4f} ± {float(r0['deltaP_std']):.4f}",
                        "ΔrelRMSE mean±std": f"{float(r0['deltaR_mean']):.4f} ± {float(r0['deltaR_std']):.4f}",
                        "Fisher p": fmt(float(r0["fisher_p_P"])),
                        "#folds ΔP>0": f"{int(r0['n_pos_deltaP'])}/{n_fields}",
                        "corr(tz2c,r)_test": f"{float(r0['corr_tz2c_r_test_mean']):.4f} ± {float(r0['corr_tz2c_r_test_std']):.4f}",
                        "PASS": "PASS" if bool(r0["verdict_PASS"]) else "FAIL",
                    }
                )
            md_parts.append(
                md_table(
                    rows,
                    ["ε", "ΔPearson mean±std", "ΔrelRMSE mean±std", "Fisher p", "#folds ΔP>0", "corr(tz2c,r)_test", "PASS"],
                )
            )
            md_parts.append("")

        summary_md = "\n".join(md_parts) + "\n"
        (paths.run_dir / "summary_e23_modeA_nonlinear_topo_w33.md").write_text(summary_md, encoding="utf-8")

        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "grid_size": grid_size,
                "alpha": alpha,
                "k0_frac": k0_frac,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "window_size": w,
                "eps_list": eps_list,
                "pixel": {
                    "solver": "sgd_l2",
                    "alpha": pixel_alpha,
                    "epochs": pixel_epochs,
                    "batch_size": pixel_batch_size,
                    "eta0": pixel_eta0,
                    "power_t": pixel_power_t,
                },
                "residual": {"ridge_alpha": ridge_alpha, "quantiles_b0": quantiles_b0, "n_perms": n_perms},
                "tz_global": {"mean": t_mu_global, "std": t_sd_global, "tz2_mean": tz2_mu_global, "s_scale_std_gx": s_scale},
                "topo_names": topo_names,
                "results_by_ck": rows_by_ck,
                "detection_threshold_by_ck": det_by_ck,
                "best_kind": best_kind,
            },
        )
        return paths

    if experiment == "e24":
        # Natural residual (no injection): test nonlinear topo features beyond pixels-only baseline (LOFO).
        from numpy.lib.stride_tricks import sliding_window_view
        from scipy import stats
        from sklearn.linear_model import SGDRegressor

        grid_size = int(cfg.get("grid_size", 256))
        alpha = float(cfg.get("alpha", 2.0))
        k0_frac = float(cfg.get("k0_frac", 0.15))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 20_000))
        ws = [int(x) for x in cfg.get("ws", [17, 33])]

        ridge_alpha = float(cfg.get("ridge_alpha", 1.0))
        quantiles_b0 = [float(x) for x in cfg.get("quantiles_b0", [0.6, 0.7, 0.8, 0.9])]
        n_perms = int(cfg.get("n_perms", 200))

        pixel_epochs = int(cfg.get("pixel_epochs", 3))
        pixel_batch_size = int(cfg.get("pixel_batch_size", 4096))
        pixel_alpha = float(cfg.get("pixel_alpha", 1e-4))
        pixel_eta0 = float(cfg.get("pixel_eta0", 0.01))
        pixel_power_t = float(cfg.get("pixel_power_t", 0.25))

        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if not (0.0 < k0_frac < 0.5):
            raise ValueError("k0_frac must be in (0,0.5)")
        if n_fields < 2:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        if any(w <= 0 or (w % 2) == 0 for w in ws):
            raise ValueError("ws must be odd positive window sizes")
        if any((q <= 0.0) or (q >= 1.0) for q in quantiles_b0):
            raise ValueError("quantiles_b0 must be in (0,1)")
        if n_perms <= 0:
            raise ValueError("n_perms must be > 0")

        rng_pixel = np.random.default_rng(int(placebo_seed) + 55_555)
        rng_null = np.random.default_rng(int(placebo_seed) + 66_666)

        def fisher_p(ps: list[float]) -> float:
            if not ps:
                return float("nan")
            eps = 1.0 / float(n_perms + 1)
            ps2 = [min(1.0, max(float(p), eps)) for p in ps]
            stat = -2.0 * float(np.sum(np.log(ps2)))
            return float(stats.chi2.sf(stat, 2 * len(ps2)))

        def fmt(x: float) -> str:
            if not np.isfinite(x):
                return "nan"
            ax = abs(float(x))
            if ax != 0.0 and (ax < 1e-3 or ax >= 1e3):
                return f"{x:.3e}"
            return f"{x:.4f}"

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r0 in rows:
                out.append("| " + " | ".join(r0.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        def safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            am = a - float(a.mean())
            bm = b - float(b.mean())
            denom = float(np.linalg.norm(am) * np.linalg.norm(bm)) + 1e-12
            return float((am @ bm) / denom)

        def relrmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
            rm = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            sd = float(np.std(y_true))
            return float(rm / sd) if sd > 0 else float("nan")

        def fit_pixel_model(train_fields: list[int], Xpix_by_field: list[np.ndarray], y_by_field: list[np.ndarray]) -> tuple[SGDRegressor, SGDRegressor]:
            reg_gx = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(pixel_alpha),
                learning_rate="invscaling",
                eta0=float(pixel_eta0),
                power_t=float(pixel_power_t),
                max_iter=1,
                tol=None,
                fit_intercept=True,
                random_state=int(placebo_seed) + 0,
            )
            reg_gy = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(pixel_alpha),
                learning_rate="invscaling",
                eta0=float(pixel_eta0),
                power_t=float(pixel_power_t),
                max_iter=1,
                tol=None,
                fit_intercept=True,
                random_state=int(placebo_seed) + 1,
            )
            for _epoch in range(pixel_epochs):
                for fid in train_fields:
                    X = Xpix_by_field[fid]
                    y = y_by_field[fid]
                    perm = rng_pixel.permutation(X.shape[0])
                    for start in range(0, X.shape[0], pixel_batch_size):
                        idx = perm[start : start + pixel_batch_size]
                        Xb = X[idx]
                        reg_gx.partial_fit(Xb, y[idx, 0])
                        reg_gy.partial_fit(Xb, y[idx, 1])
            return reg_gx, reg_gy

        def predict_pixel(reg_gx: SGDRegressor, reg_gy: SGDRegressor, X: np.ndarray) -> np.ndarray:
            X = np.asarray(X)
            y_pred = np.empty((X.shape[0], 2), dtype=np.float64)
            for start in range(0, X.shape[0], pixel_batch_size):
                sl = slice(start, min(X.shape[0], start + pixel_batch_size))
                y_pred[sl, 0] = reg_gx.predict(X[sl])
                y_pred[sl, 1] = reg_gy.predict(X[sl])
            return y_pred

        # Generate per-field maps once (rho_high z-scored per field).
        rho_high_by_field: list[np.ndarray] = []
        gx_by_field: list[np.ndarray] = []
        gy_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
            split = band_split_poisson_2d(rho01, k0_frac=k0_frac)
            rho_high_by_field.append(np.asarray(_zscore_field(split.rho_high), dtype=np.float64))
            gx_by_field.append(np.asarray(split.high.gx, dtype=np.float64))
            gy_by_field.append(np.asarray(split.high.gy, dtype=np.float64))

        ckinds = ["C1", "C2", "C3", "C4"]
        per_w_summary: dict[int, dict[str, Any]] = {}

        for w in ws:
            ps = _require_odd("w", w)
            r = ps // 2

            Xpix_by_field: list[np.ndarray] = []
            B_by_field: list[np.ndarray] = []
            T_by_field: list[np.ndarray] = []
            y_by_field: list[np.ndarray] = []
            topo_names: list[str] | None = None

            # Build patch datasets
            for field_id in range(n_fields):
                field = rho_high_by_field[field_id]
                gx = gx_by_field[field_id]
                gy = gy_by_field[field_id]

                thr = [float(np.quantile(field.reshape(-1), q)) for q in quantiles_b0]
                rng_samp = np.random.default_rng(seed + 10_000 * field_id + 17 * ps)
                feats, y, centers = _sample_features_2d_fast_topo(
                    field=field,
                    gx=gx,
                    gy=gy,
                    n_patches=patches_per_field,
                    patch_size=ps,
                    topo_mode="quantile_per_field",
                    topo_thresholds=thr,
                    thresholds_pos_sigma=None,
                    thresholds_neg_sigma=None,
                    rng=rng_samp,
                )
                y = np.asarray(y, dtype=np.float64)
                assert y.shape == (patches_per_field, 2)

                cx = centers[:, 0].astype(np.int64)
                cy = centers[:, 1].astype(np.int64)
                win = sliding_window_view(field, (ps, ps))
                Xpix = win[cx - r, cy - r].reshape(patches_per_field, -1).astype(np.float32, copy=False)

                B_names = ["mass", "mass2", "var", "max", "grad_energy"]
                B = np.column_stack([np.asarray(feats[n], dtype=np.float64) for n in B_names])

                tnames = sorted([k for k in feats.keys() if k.startswith("b0_")])
                if topo_names is None:
                    topo_names = tnames
                elif topo_names != tnames:
                    raise RuntimeError(f"Topo feature names mismatch across fields: {topo_names} vs {tnames}")
                T = np.column_stack([np.asarray(feats[n], dtype=np.float64) for n in topo_names])

                Xpix_by_field.append(Xpix)
                B_by_field.append(B)
                T_by_field.append(T)
                y_by_field.append(y)

            if topo_names is None:
                raise RuntimeError("No topo features found")

            # Fold cache: pixel baseline residual targets and standardized B
            fold_cache: dict[int, dict[str, Any]] = {}
            for test_field in range(n_fields):
                train_fields = [i for i in range(n_fields) if i != test_field]
                reg_gx, reg_gy = fit_pixel_model(train_fields, Xpix_by_field, y_by_field)

                # Predictions for train and test to build residual targets (gx channel only).
                r_tr_list: list[np.ndarray] = []
                for fid in train_fields:
                    ypred_tr = predict_pixel(reg_gx, reg_gy, Xpix_by_field[fid])
                    r_tr_list.append(y_by_field[fid][:, 0] - ypred_tr[:, 0])
                r_tr = np.concatenate(r_tr_list, axis=0)

                ypred_te = predict_pixel(reg_gx, reg_gy, Xpix_by_field[test_field])
                r_te = y_by_field[test_field][:, 0] - ypred_te[:, 0]

                # Standardize B on train
                Btr = np.concatenate([B_by_field[i] for i in train_fields], axis=0)
                Bte = B_by_field[test_field]
                B_mu = Btr.mean(axis=0)
                B_sd = np.where(Btr.std(axis=0) > 0, Btr.std(axis=0), 1.0)
                Btr_s = (Btr - B_mu) / B_sd
                Bte_s = (Bte - B_mu) / B_sd

                r_mu = float(np.mean(r_tr))
                rc = (r_tr - r_mu).reshape(-1, 1)

                # Fit B_resid once per fold
                A = Btr_s.T @ Btr_s
                uB = Btr_s.T @ rc
                wB = np.linalg.solve(A + ridge_alpha * np.eye(A.shape[0]), uB).reshape(-1)
                rB_te = Bte_s @ wB + r_mu
                pB = safe_corr_1d(r_te, rB_te)
                rrB = relrmse_1d(r_te, rB_te)

                fold_cache[test_field] = {
                    "train_fields": train_fields,
                    "r_te": r_te,
                    "r_mu": r_mu,
                    "rc": rc,
                    "Btr_s": Btr_s,
                    "Bte_s": Bte_s,
                    "A": A,
                    "uB": uB,
                    "pB": pB,
                    "rrB": rrB,
                }

            # Residual feature builders for topo-derived blocks (fold-safe).
            def build_F(train_fields: list[int], test_field: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
                Ttr_raw = np.concatenate([T_by_field[i] for i in train_fields], axis=0)
                Tte_raw = T_by_field[test_field]
                if kind == "C1":
                    return Ttr_raw, Tte_raw
                if kind == "C2":
                    return np.concatenate([Ttr_raw, Ttr_raw * Ttr_raw], axis=1), np.concatenate([Tte_raw, Tte_raw * Tte_raw], axis=1)
                if kind == "C3":
                    mu = Ttr_raw.mean(axis=0, keepdims=True)
                    return np.concatenate([Ttr_raw, np.abs(Ttr_raw - mu)], axis=1), np.concatenate([Tte_raw, np.abs(Tte_raw - mu)], axis=1)
                if kind == "C4":
                    # b0, b0^2, and adjacent products
                    parts_tr = [Ttr_raw, Ttr_raw * Ttr_raw]
                    parts_te = [Tte_raw, Tte_raw * Tte_raw]
                    if Ttr_raw.shape[1] >= 2:
                        adj_tr = [Ttr_raw[:, i] * Ttr_raw[:, i + 1] for i in range(Ttr_raw.shape[1] - 1)]
                        adj_te = [Tte_raw[:, i] * Tte_raw[:, i + 1] for i in range(Tte_raw.shape[1] - 1)]
                        parts_tr.append(np.column_stack(adj_tr))
                        parts_te.append(np.column_stack(adj_te))
                    return np.concatenate(parts_tr, axis=1), np.concatenate(parts_te, axis=1)
                raise ValueError(f"Unknown kind: {kind}")

            # Evaluate each Ck vs B across folds with permutation placebo
            ck_rows: list[dict[str, Any]] = []
            for kind in ckinds:
                deltasP: list[float] = []
                deltasR: list[float] = []
                pvals_fold: list[float] = []

                for test_field in range(n_fields):
                    fc = fold_cache[test_field]
                    train_fields = fc["train_fields"]
                    Btr_s = fc["Btr_s"]
                    Bte_s = fc["Bte_s"]
                    rc = fc["rc"]
                    r_mu = fc["r_mu"]
                    r_te = fc["r_te"]
                    pB = float(fc["pB"])
                    rrB = float(fc["rrB"])

                    Ftr_raw, Fte_raw = build_F(train_fields, test_field, kind)
                    Ftr_raw = np.asarray(Ftr_raw, dtype=np.float64)
                    Fte_raw = np.asarray(Fte_raw, dtype=np.float64)

                    F_mu = Ftr_raw.mean(axis=0)
                    F_sd = np.where(Ftr_raw.std(axis=0) > 0, Ftr_raw.std(axis=0), 1.0)
                    Ftr_s = (Ftr_raw - F_mu) / F_sd
                    Fte_s = (Fte_raw - F_mu) / F_sd

                    XtrC = np.concatenate([Btr_s, Ftr_s], axis=1)
                    XteC = np.concatenate([Bte_s, Fte_s], axis=1)
                    XtX = XtrC.T @ XtrC
                    Xty = XtrC.T @ rc
                    wC = np.linalg.solve(XtX + ridge_alpha * np.eye(XtX.shape[0]), Xty).reshape(-1)
                    rC_te = XteC @ wC + r_mu
                    pC = safe_corr_1d(r_te, rC_te)
                    rrC = relrmse_1d(r_te, rC_te)
                    deltaP_real = float(pC - pB)
                    deltaR_real = float(rrC - rrB)

                    deltasP.append(deltaP_real)
                    deltasR.append(deltaR_real)

                    # Permutation null on ΔPearson: shuffle topo-derived rows within train.
                    A = fc["A"]
                    uB = fc["uB"]
                    Cmat = Ftr_s.T @ Ftr_s
                    Z = np.concatenate([Btr_s, rc], axis=1)

                    null_dP: list[float] = []
                    perms = [rng_null.permutation(Ftr_s.shape[0]) for _ in range(n_perms)]
                    for pidx in perms:
                        Fp = Ftr_s[pidx]
                        M = Z.T @ Fp
                        D = M[:-1, :]
                        v = M[-1, :].reshape(-1, 1)
                        XtX_p = np.block([[A, D], [D.T, Cmat]])
                        Xty_p = np.vstack([uB, v])
                        wP = np.linalg.solve(XtX_p + ridge_alpha * np.eye(XtX_p.shape[0]), Xty_p).reshape(-1)
                        rP_te = XteC @ wP + r_mu
                        pP = safe_corr_1d(r_te, rP_te)
                        null_dP.append(float(pP - pB))

                    null = np.asarray(null_dP, dtype=np.float64)
                    p_emp = float(np.mean(null >= deltaP_real))
                    pvals_fold.append(p_emp)

                dp = np.asarray(deltasP, dtype=np.float64)
                dr = np.asarray(deltasR, dtype=np.float64)
                n_pos = int(np.sum(dp > 0.0))
                fp = fisher_p(pvals_fold)
                verdict = (float(fp) < 0.05) and (float(dr.mean()) < 0.0) and (n_pos >= 7)

                ck_rows.append(
                    {
                        "w": int(w),
                        "Ck": kind,
                        "deltaP_mean": float(dp.mean()),
                        "deltaP_std": float(dp.std(ddof=1)) if len(dp) > 1 else 0.0,
                        "deltaR_mean": float(dr.mean()),
                        "deltaR_std": float(dr.std(ddof=1)) if len(dr) > 1 else 0.0,
                        "fisher_p_P": float(fp),
                        "n_pos_deltaP": int(n_pos),
                        "n_fields": int(n_fields),
                        "verdict_PASS": bool(verdict),
                    }
                )

            # best Ck by ΔPearson mean (tie-break by fisher p)
            best = sorted(
                ck_rows,
                key=lambda r0: (-float(r0["deltaP_mean"]), float(r0["fisher_p_P"]), -int(r0["n_pos_deltaP"])),
            )[0]

            per_w_summary[w] = {
                "w": int(w),
                "rows": ck_rows,
                "best_Ck": str(best["Ck"]),
                "best_deltaP_mean": float(best["deltaP_mean"]),
                "best_fisher_p": float(best["fisher_p_P"]),
            }

            # write per-w CSV
            header = list(ck_rows[0].keys()) if ck_rows else []
            write_csv(paths.run_dir / f"summary_by_ck_w{w}.csv", header, [[r0[h] for h in header] for r0 in ck_rows])

            del Xpix_by_field, B_by_field, T_by_field, y_by_field, fold_cache

        # Summary markdown
        md_parts: list[str] = []
        md_parts.append("# E24 — Natural residual nonlinear topo (LOFO)\n")
        md_parts.append(f"- run: `{paths.run_dir}`")
        md_parts.append(f"- grid_size={grid_size}, alpha={alpha}, k0_frac={k0_frac}, n_fields={n_fields}, patches_per_field={patches_per_field}")
        md_parts.append(f"- ws={ws}  perms={n_perms}")
        md_parts.append("")

        table_rows: list[dict[str, str]] = []
        for w in ws:
            for r0 in per_w_summary[w]["rows"]:
                table_rows.append(
                    {
                        "w": str(int(r0["w"])),
                        "Ck": str(r0["Ck"]),
                        "ΔPearson mean±std": f"{float(r0['deltaP_mean']):.4f} ± {float(r0['deltaP_std']):.4f}",
                        "ΔrelRMSE mean±std": f"{float(r0['deltaR_mean']):.4f} ± {float(r0['deltaR_std']):.4f}",
                        "Fisher p": fmt(float(r0["fisher_p_P"])),
                        "#folds ΔP>0": f"{int(r0['n_pos_deltaP'])}/{n_fields}",
                        "verdict": "PASS" if bool(r0["verdict_PASS"]) else "FAIL",
                    }
                )
        md_parts.append(
            md_table(
                table_rows,
                ["w", "Ck", "ΔPearson mean±std", "ΔrelRMSE mean±std", "Fisher p", "#folds ΔP>0", "verdict"],
            )
        )
        md_parts.append("")
        md_parts.append("## Best per w")
        for w in ws:
            md_parts.append(f"- w={w}: best `{per_w_summary[w]['best_Ck']}` (ΔPearson_mean={fmt(per_w_summary[w]['best_deltaP_mean'])}, Fisher p={fmt(per_w_summary[w]['best_fisher_p'])})")
        md_parts.append("")

        summary_md = "\n".join(md_parts) + "\n"
        (paths.run_dir / "summary_e24_natural_residual_nonlinear_topo.md").write_text(summary_md, encoding="utf-8")

        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "grid_size": grid_size,
                "alpha": alpha,
                "k0_frac": k0_frac,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "ws": ws,
                "pixel": {
                    "solver": "sgd_l2",
                    "alpha": pixel_alpha,
                    "epochs": pixel_epochs,
                    "batch_size": pixel_batch_size,
                    "eta0": pixel_eta0,
                    "power_t": pixel_power_t,
                },
                "residual": {"ridge_alpha": ridge_alpha, "quantiles_b0": quantiles_b0, "n_perms": n_perms},
                "per_w_summary": per_w_summary,
            },
        )
        return paths

    if experiment == "e25":
        # Multi-scale nonlinear topo features to explain nonlocal residual in full g (LOFO).
        #
        # Baseline: local pixels-only ridge-like SGD on rho patches (w_local).
        # Residual target: r_parallel = rx = gx_full_center - gx_pred_center (scalar).
        # Residual models: B_resid (mass/var/max/grad_energy on w_local) vs C_resid (B + multi-scale topo).
        #
        # Topo features: for each scale, compute b0(superlevel) at sigma thresholds t in {0.0,0.5,1.0}
        # on a z-scored rho field (per-field mean/std), after optional avg-pooling.
        # Add nonlinear expansions: b0, b0^2, |b0-mean_train|.
        from numpy.lib.stride_tricks import sliding_window_view
        from scipy import ndimage, stats
        from sklearn.linear_model import SGDRegressor

        grid_size = int(cfg.get("grid_size", 256))
        alpha = float(cfg.get("alpha", 2.0))
        n_fields = int(cfg.get("n_fields", 10))
        k0_frac = float(cfg.get("k0_frac", 0.15))  # used only as a seed-compatible knob; full-g does not band-split.
        patches_per_field = int(cfg.get("patches_per_field", 10_000))
        w_locals = [int(x) for x in cfg.get("w_locals", [33, 65])]
        topo_scales = [int(x) for x in cfg.get("topo_scales", [33, 65, 129])]
        topo_pool_factors = [int(x) for x in cfg.get("topo_pool_factors", [1, 2, 4])]
        sigma_thresholds = [float(x) for x in cfg.get("sigma_thresholds", [0.0, 0.5, 1.0])]
        n_perms = int(cfg.get("n_perms", 200))
        ridge_alpha = float(cfg.get("ridge_alpha", 1.0))

        pixel_epochs = int(cfg.get("pixel_epochs", 2))
        pixel_batch_size = int(cfg.get("pixel_batch_size", 1024))
        pixel_alpha = float(cfg.get("pixel_alpha", 1e-4))
        pixel_eta0 = float(cfg.get("pixel_eta0", 0.01))
        pixel_power_t = float(cfg.get("pixel_power_t", 0.25))

        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if n_fields < 2:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        if any((w <= 0) or (w % 2 == 0) for w in w_locals):
            raise ValueError("w_locals must be odd positive ints")
        if any((w <= 0) or (w % 2 == 0) for w in topo_scales):
            raise ValueError("topo_scales must be odd positive ints")
        if len(topo_pool_factors) < len(topo_scales):
            raise ValueError("topo_pool_factors must have at least as many entries as topo_scales")
        if any(f <= 0 for f in topo_pool_factors):
            raise ValueError("topo_pool_factors must be positive ints")
        if any(t < 0.0 for t in sigma_thresholds):
            raise ValueError("sigma_thresholds must be >= 0")
        if n_perms <= 0:
            raise ValueError("n_perms must be > 0")

        rng_pixel = np.random.default_rng(int(placebo_seed) + 77_777)
        rng_null = np.random.default_rng(int(placebo_seed) + 88_888)

        def fisher_p(ps: list[float]) -> float:
            if not ps:
                return float("nan")
            eps = 1.0 / float(n_perms + 1)
            ps2 = [min(1.0, max(float(p), eps)) for p in ps]
            stat = -2.0 * float(np.sum(np.log(ps2)))
            return float(stats.chi2.sf(stat, 2 * len(ps2)))

        def fmt(x: float) -> str:
            if not np.isfinite(x):
                return "nan"
            ax = abs(float(x))
            if ax != 0.0 and (ax < 1e-3 or ax >= 1e3):
                return f"{x:.3e}"
            return f"{x:.4f}"

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r0 in rows:
                out.append("| " + " | ".join(r0.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        def safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            am = a - float(a.mean())
            bm = b - float(b.mean())
            denom = float(np.linalg.norm(am) * np.linalg.norm(bm)) + 1e-12
            return float((am @ bm) / denom)

        def relrmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
            rm = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            sd = float(np.std(y_true))
            return float(rm / sd) if sd > 0 else float("nan")

        def avg_pool(patch: np.ndarray, factor: int) -> np.ndarray:
            if factor == 1:
                return patch
            h, w_ = patch.shape
            hh = (h // factor) * factor
            ww = (w_ // factor) * factor
            x = patch[:hh, :ww]
            x = x.reshape(hh // factor, factor, ww // factor, factor).mean(axis=(1, 3))
            return x

        # Connected components structure (4-neigh)
        cc_structure = ndimage.generate_binary_structure(2, 1)

        def b0_counts_sigma(patch_z: np.ndarray) -> np.ndarray:
            out = np.zeros((len(sigma_thresholds),), dtype=np.float64)
            for j, t in enumerate(sigma_thresholds):
                binary = patch_z > float(t)
                if not binary.any():
                    out[j] = 0.0
                else:
                    _, num = ndimage.label(binary, structure=cc_structure)
                    out[j] = float(num)
            return out

        def fit_pixel_model(train_fields: list[int], fields_z: list[np.ndarray], centers_by_field: list[np.ndarray], w_local: int, y_by_field: list[np.ndarray]) -> tuple[SGDRegressor, SGDRegressor]:
            reg_gx = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(pixel_alpha),
                learning_rate="invscaling",
                eta0=float(pixel_eta0),
                power_t=float(pixel_power_t),
                max_iter=1,
                tol=None,
                fit_intercept=True,
                random_state=int(placebo_seed) + 0,
            )
            reg_gy = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(pixel_alpha),
                learning_rate="invscaling",
                eta0=float(pixel_eta0),
                power_t=float(pixel_power_t),
                max_iter=1,
                tol=None,
                fit_intercept=True,
                random_state=int(placebo_seed) + 1,
            )
            rloc = w_local // 2
            for _epoch in range(pixel_epochs):
                for fid in train_fields:
                    field = fields_z[fid]
                    centers = centers_by_field[fid]
                    y = y_by_field[fid]
                    win = sliding_window_view(field, (w_local, w_local))
                    perm = rng_pixel.permutation(centers.shape[0])
                    for start in range(0, centers.shape[0], pixel_batch_size):
                        idx = perm[start : start + pixel_batch_size]
                        cx = centers[idx, 0].astype(np.int64)
                        cy = centers[idx, 1].astype(np.int64)
                        Xb = win[cx - rloc, cy - rloc].reshape(len(idx), -1).astype(np.float32, copy=False)
                        reg_gx.partial_fit(Xb, y[idx, 0])
                        reg_gy.partial_fit(Xb, y[idx, 1])
            return reg_gx, reg_gy

        def predict_pixel(reg_gx: SGDRegressor, reg_gy: SGDRegressor, field_z: np.ndarray, centers: np.ndarray, w_local: int) -> np.ndarray:
            rloc = w_local // 2
            win = sliding_window_view(field_z, (w_local, w_local))
            out = np.empty((centers.shape[0], 2), dtype=np.float64)
            for start in range(0, centers.shape[0], pixel_batch_size):
                idx = slice(start, min(centers.shape[0], start + pixel_batch_size))
                cx = centers[idx, 0].astype(np.int64)
                cy = centers[idx, 1].astype(np.int64)
                Xb = win[cx - rloc, cy - rloc].reshape(len(cx), -1).astype(np.float32, copy=False)
                out[idx, 0] = reg_gx.predict(Xb)
                out[idx, 1] = reg_gy.predict(Xb)
            return out

        # Generate fields (rho in [0,1]) and full g, plus z-scored rho for pixels/topo.
        from .data_synth import solve_poisson_periodic_fft_2d

        rho01_by_field: list[np.ndarray] = []
        rhoz_by_field: list[np.ndarray] = []
        gx_full_by_field: list[np.ndarray] = []
        gy_full_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id + int(1e6 * k0_frac))
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
            sol = solve_poisson_periodic_fft_2d(rho01)
            rho01_by_field.append(rho01.astype(np.float64, copy=False))
            rhoz_by_field.append(_zscore_field(rho01).astype(np.float64, copy=False))
            gx_full_by_field.append(sol.gx.astype(np.float64, copy=False))
            gy_full_by_field.append(sol.gy.astype(np.float64, copy=False))

        # Choose centers once per field, valid for all w_local and topo_scales (avoid borders).
        r_max = max(max(w_locals), max(topo_scales)) // 2
        centers_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_cent = np.random.default_rng(seed + 123_456 + 1_000 * field_id)
            cx = rng_cent.integers(r_max, grid_size - r_max, size=patches_per_field, dtype=np.int64)
            cy = rng_cent.integers(r_max, grid_size - r_max, size=patches_per_field, dtype=np.int64)
            centers_by_field.append(np.column_stack([cx, cy]).astype(np.int64, copy=False))

        # Precompute B features on rho01 for each w_local and each field (stored per w_local).
        B_by_w_field: dict[int, list[np.ndarray]] = {w: [] for w in w_locals}
        for w_local in w_locals:
            rloc = w_local // 2
            for field_id in range(n_fields):
                field = rho01_by_field[field_id]
                centers = centers_by_field[field_id]
                cx = centers[:, 0]
                cy = centers[:, 1]
                pref1 = _prefix_sum_2d(field)
                pref2 = _prefix_sum_2d(field * field)
                x0 = (cx - rloc).astype(np.int64)
                x1 = (cx + rloc + 1).astype(np.int64)
                y0 = (cy - rloc).astype(np.int64)
                y1 = (cy + rloc + 1).astype(np.int64)
                mass = _box_sum_2d(pref1, x0, x1, y0, y1)
                nvox = float(w_local * w_local)
                mean = mass / nvox
                sumsq = _box_sum_2d(pref2, x0, x1, y0, y1)
                var = np.maximum(0.0, sumsq / nvox - mean * mean)
                mass2 = mass * mass
                max_grid = ndimage.maximum_filter(field, size=w_local, mode="constant", cval=-np.inf)
                mx = max_grid[cx, cy]
                gx_f, gy_f = np.gradient(field)
                egrid = gx_f * gx_f + gy_f * gy_f
                eavg = ndimage.uniform_filter(egrid, size=w_local, mode="constant", cval=0.0)
                ge = eavg[cx, cy]
                B = np.column_stack([mass, mass2, var, mx, ge]).astype(np.float64, copy=False)
                B_by_w_field[w_local].append(B)

        # Precompute raw multi-scale b0 features per field (independent of w_local).
        # Shape per field: (n_patches, n_scales * n_thresholds)
        topo_scale_cfg: list[tuple[int, int]] = []
        for s, f in zip(topo_scales, topo_pool_factors):
            if s > grid_size:
                continue
            topo_scale_cfg.append((int(s), int(f)))
        if not topo_scale_cfg:
            raise ValueError("No valid topo_scales (all larger than grid_size?)")

        topo_raw_by_field: list[np.ndarray] = []
        topo_names: list[str] = []
        for s, f in topo_scale_cfg:
            for t in sigma_thresholds:
                topo_names.append(f"b0_s{s}_p{f}_t{str(t).replace('.','p')}")

        for field_id in range(n_fields):
            field_z = rhoz_by_field[field_id]
            centers = centers_by_field[field_id]
            vals = np.zeros((centers.shape[0], len(topo_names)), dtype=np.float64)
            col = 0
            for s, f in topo_scale_cfg:
                rs = s // 2
                for i in range(centers.shape[0]):
                    cx, cy = int(centers[i, 0]), int(centers[i, 1])
                    patch = field_z[cx - rs : cx + rs + 1, cy - rs : cy + rs + 1]
                    patch_p = avg_pool(patch, f)
                    b0 = b0_counts_sigma(patch_p)
                    vals[i, col : col + len(sigma_thresholds)] = b0
                col += len(sigma_thresholds)
            topo_raw_by_field.append(vals)

        # Run configs: for each w_local, fit LOFO pixel baseline and evaluate residual model B vs C.
        rows_summary: list[dict[str, Any]] = []
        for w_local in w_locals:
            # y_by_field at centers for this w_local
            y_cent_by_field: list[np.ndarray] = []
            for fid in range(n_fields):
                centers = centers_by_field[fid]
                cx = centers[:, 0].astype(np.int64)
                cy = centers[:, 1].astype(np.int64)
                y_cent_by_field.append(
                    np.column_stack([gx_full_by_field[fid][cx, cy], gy_full_by_field[fid][cx, cy]]).astype(np.float64, copy=False)
                )

            fold_rows: list[dict[str, Any]] = []
            pvals: list[float] = []

            for test_field in range(n_fields):
                train_fields = [i for i in range(n_fields) if i != test_field]
                reg_gx, reg_gy = fit_pixel_model(train_fields, rhoz_by_field, centers_by_field, w_local, y_cent_by_field)

                # residual targets
                r_tr_list: list[np.ndarray] = []
                Btr_list: list[np.ndarray] = []
                Ftr_list: list[np.ndarray] = []
                for fid in train_fields:
                    y = y_cent_by_field[fid]
                    ypred = predict_pixel(reg_gx, reg_gy, rhoz_by_field[fid], centers_by_field[fid], w_local)
                    r_tr_list.append((y[:, 0] - ypred[:, 0]).astype(np.float64, copy=False))
                    Btr_list.append(B_by_w_field[w_local][fid])
                    Ftr_list.append(topo_raw_by_field[fid])
                r_tr = np.concatenate(r_tr_list, axis=0)
                Btr = np.concatenate(Btr_list, axis=0)
                Ftr_raw = np.concatenate(Ftr_list, axis=0)

                y_te = y_cent_by_field[test_field]
                ypred_te = predict_pixel(reg_gx, reg_gy, rhoz_by_field[test_field], centers_by_field[test_field], w_local)
                r_te = (y_te[:, 0] - ypred_te[:, 0]).astype(np.float64, copy=False)
                Bte = B_by_w_field[w_local][test_field]
                Fte_raw = topo_raw_by_field[test_field]

                # Standardize B on train
                B_mu = Btr.mean(axis=0)
                B_sd = np.where(Btr.std(axis=0) > 0, Btr.std(axis=0), 1.0)
                Btr_s = (Btr - B_mu) / B_sd
                Bte_s = (Bte - B_mu) / B_sd

                # Base topo block = raw b0 counts; compute expansions fold-safely:
                # b0, b0^2, |b0-mean_train|
                F_mu_raw = Ftr_raw.mean(axis=0, keepdims=True)
                Ftr = np.concatenate([Ftr_raw, Ftr_raw * Ftr_raw, np.abs(Ftr_raw - F_mu_raw)], axis=1)
                Fte = np.concatenate([Fte_raw, Fte_raw * Fte_raw, np.abs(Fte_raw - F_mu_raw)], axis=1)

                # Standardize F on train
                F_mu = Ftr.mean(axis=0)
                F_sd = np.where(Ftr.std(axis=0) > 0, Ftr.std(axis=0), 1.0)
                Ftr_s = (Ftr - F_mu) / F_sd
                Fte_s = (Fte - F_mu) / F_sd

                # Center residual target
                r_mu = float(np.mean(r_tr))
                rc = (r_tr - r_mu).reshape(-1, 1)

                # Fit B_resid
                A = Btr_s.T @ Btr_s
                uB = Btr_s.T @ rc
                wB = np.linalg.solve(A + ridge_alpha * np.eye(A.shape[0]), uB).reshape(-1)
                rB_te = Bte_s @ wB + r_mu
                pB = safe_corr_1d(r_te, rB_te)
                rrB = relrmse_1d(r_te, rB_te)

                # Fit C_resid (real)
                XtrC = np.concatenate([Btr_s, Ftr_s], axis=1)
                XteC = np.concatenate([Bte_s, Fte_s], axis=1)
                XtX = XtrC.T @ XtrC
                Xty = XtrC.T @ rc
                wC = np.linalg.solve(XtX + ridge_alpha * np.eye(XtX.shape[0]), Xty).reshape(-1)
                rC_te = XteC @ wC + r_mu
                pC = safe_corr_1d(r_te, rC_te)
                rrC = relrmse_1d(r_te, rC_te)

                deltaP = float(pC - pB)
                deltaR = float(rrC - rrB)

                # Placebo permutations: shuffle topo rows in train.
                # Use block-matrix trick: Z = [B, rc], then M = Z^T @ Fperm gives D and v.
                Cmat = Ftr_s.T @ Ftr_s
                Z = np.concatenate([Btr_s, rc], axis=1)
                perms = [rng_null.permutation(Ftr_s.shape[0]) for _ in range(n_perms)]
                null_dP: list[float] = []
                for pidx in perms:
                    Fp = Ftr_s[pidx]
                    M = Z.T @ Fp
                    D = M[:-1, :]
                    v = M[-1, :].reshape(-1, 1)
                    XtX_p = np.block([[A, D], [D.T, Cmat]])
                    Xty_p = np.vstack([uB, v])
                    wP = np.linalg.solve(XtX_p + ridge_alpha * np.eye(XtX_p.shape[0]), Xty_p).reshape(-1)
                    rP_te = XteC @ wP + r_mu
                    pP = safe_corr_1d(r_te, rP_te)
                    null_dP.append(float(pP - pB))
                null = np.asarray(null_dP, dtype=np.float64)
                p_emp = float(np.mean(null >= deltaP))
                pvals.append(p_emp)

                fold_rows.append(
                    {
                        "field_id": int(test_field),
                        "pearson_B": float(pB),
                        "relRMSE_B": float(rrB),
                        "pearson_C": float(pC),
                        "relRMSE_C": float(rrC),
                        "deltaP": float(deltaP),
                        "deltaR": float(deltaR),
                        "p_emp": float(p_emp),
                    }
                )

            dp = np.asarray([r0["deltaP"] for r0 in fold_rows], dtype=np.float64)
            dr = np.asarray([r0["deltaR"] for r0 in fold_rows], dtype=np.float64)
            n_pos = int(np.sum(dp > 0.0))
            fp = fisher_p(pvals)
            verdict = (float(dp.mean()) > 0.0) and (float(dr.mean()) < 0.0) and (fp < 0.05) and (n_pos >= 7)

            cfg_str = f"topo_scales={[(s,f) for (s,f) in topo_scale_cfg]} sigma_thr={sigma_thresholds} exp=[b0,b0^2,abs]"
            rows_summary.append(
                {
                    "w_local": int(w_local),
                    "config": cfg_str,
                    "deltaP_mean": float(dp.mean()),
                    "deltaP_std": float(dp.std(ddof=1)) if len(dp) > 1 else 0.0,
                    "deltaR_mean": float(dr.mean()),
                    "deltaR_std": float(dr.std(ddof=1)) if len(dr) > 1 else 0.0,
                    "fisher_p": float(fp),
                    "n_pos": int(n_pos),
                    "verdict": bool(verdict),
                }
            )

            header = list(fold_rows[0].keys()) if fold_rows else []
            write_csv(paths.run_dir / f"lofo_by_field_wlocal{w_local}.csv", header, [[r0[h] for h in header] for r0 in fold_rows])

        # Summary markdown
        md_rows: list[dict[str, str]] = []
        for r0 in rows_summary:
            md_rows.append(
                {
                    "w_local": str(int(r0["w_local"])),
                    "config": str(r0["config"]),
                    "ΔPearson mean±std": f"{float(r0['deltaP_mean']):.4f} ± {float(r0['deltaP_std']):.4f}",
                    "ΔrelRMSE mean±std": f"{float(r0['deltaR_mean']):.4f} ± {float(r0['deltaR_std']):.4f}",
                    "Fisher p": fmt(float(r0["fisher_p"])),
                    "#folds ΔP>0": f"{int(r0['n_pos'])}/{n_fields}",
                    "verdict": "PASS" if bool(r0["verdict"]) else "FAIL",
                }
            )

        summary_md = (
            "# E25 — Multi-scale topo on full-g residual (LOFO)\n\n"
            f"- run: `{paths.run_dir}`\n"
            f"- grid_size={grid_size}, alpha={alpha}, n_fields={n_fields}, patches_per_field={patches_per_field}\n"
            f"- w_locals={w_locals}\n"
            f"- topo_scales_used={topo_scale_cfg}\n"
            f"- sigma_thresholds={sigma_thresholds}\n"
            f"- pixel: SGD L2 (ridge-like) alpha={pixel_alpha}, epochs={pixel_epochs}, batch={pixel_batch_size}\n"
            f"- residual ridge_alpha={ridge_alpha}, perms={n_perms}\n\n"
            + md_table(md_rows, ["w_local", "config", "ΔPearson mean±std", "ΔrelRMSE mean±std", "Fisher p", "#folds ΔP>0", "verdict"])
            + "\n"
        )
        (paths.run_dir / "summary_e25_multiscale_topo_fullg_residual.md").write_text(summary_md, encoding="utf-8")

        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "grid_size": grid_size,
                "alpha": alpha,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "w_locals": w_locals,
                "topo_scales_used": topo_scale_cfg,
                "sigma_thresholds": sigma_thresholds,
                "pixel": {
                    "solver": "sgd_l2",
                    "alpha": pixel_alpha,
                    "epochs": pixel_epochs,
                    "batch_size": pixel_batch_size,
                    "eta0": pixel_eta0,
                    "power_t": pixel_power_t,
                },
                "residual": {"ridge_alpha": ridge_alpha, "n_perms": n_perms},
                "topo_names_raw": topo_names,
                "rows_summary": rows_summary,
            },
        )
        return paths

    if experiment == "e26":
        # Non-local geometric moments vs topo for full-g residual (LOFO).
        #
        # Data: rho01 in [0,1] (2D 1/f), full gx_full from Poisson FFT.
        # Baseline: local pixels-only ridge-like SGD on z-scored rho patches (w_local).
        # Residual target: r = gx_full_center - gx_pred_center (scalar).
        # Residual models: B_resid (local B features on w_local) vs C_resid (B + nonlocal moments on w_big=129).
        from numpy.lib.stride_tricks import sliding_window_view
        from scipy import ndimage, stats
        from sklearn.linear_model import SGDRegressor

        from .data_synth import solve_poisson_periodic_fft_2d

        grid_size = int(cfg.get("grid_size", 256))
        alpha = float(cfg.get("alpha", 2.0))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 10_000))
        w_locals = [int(x) for x in cfg.get("w_locals", [33, 65])]
        w_big = int(cfg.get("w_big", 129))
        include_quadrupole = bool(cfg.get("include_quadrupole", True))
        b_resid_mode = str(cfg.get("b_resid_mode", "local_B")).lower()
        n_perms = int(cfg.get("n_perms", 200))
        ridge_alpha = float(cfg.get("ridge_alpha", 1.0))

        pixel_epochs = int(cfg.get("pixel_epochs", 2))
        pixel_batch_size = int(cfg.get("pixel_batch_size", 1024))
        pixel_alpha = float(cfg.get("pixel_alpha", 1e-4))
        pixel_eta0 = float(cfg.get("pixel_eta0", 0.01))
        pixel_power_t = float(cfg.get("pixel_power_t", 0.25))

        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if n_fields < 2:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        if any((w <= 0) or (w % 2 == 0) for w in w_locals):
            raise ValueError("w_locals must be odd positive ints")
        w_big = _require_odd("w_big", w_big)
        if w_big > grid_size:
            raise ValueError("w_big must be <= grid_size")
        if n_perms <= 0:
            raise ValueError("n_perms must be > 0")
        if b_resid_mode not in {"local_b", "intercept"}:
            raise ValueError("b_resid_mode must be 'local_B' or 'intercept'")

        rng_pixel = np.random.default_rng(int(placebo_seed) + 99_111)
        rng_null = np.random.default_rng(int(placebo_seed) + 99_222)

        def fisher_p(ps: list[float]) -> float:
            if not ps:
                return float("nan")
            eps = 1.0 / float(n_perms + 1)
            ps2 = [min(1.0, max(float(p), eps)) for p in ps]
            stat = -2.0 * float(np.sum(np.log(ps2)))
            return float(stats.chi2.sf(stat, 2 * len(ps2)))

        def fmt(x: float) -> str:
            if not np.isfinite(x):
                return "nan"
            ax = abs(float(x))
            if ax != 0.0 and (ax < 1e-3 or ax >= 1e3):
                return f"{x:.3e}"
            return f"{x:.4f}"

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r0 in rows:
                out.append("| " + " | ".join(r0.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        def safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            am = a - float(a.mean())
            bm = b - float(b.mean())
            denom = float(np.linalg.norm(am) * np.linalg.norm(bm)) + 1e-12
            return float((am @ bm) / denom)

        def relrmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
            e = y_pred - y_true
            rmse = float(np.sqrt(np.mean(e * e)))
            sd = float(np.std(y_true))
            return rmse / (sd + 1e-12)

        # Pixel baseline: ridge-like SGD (scalar target), with fold-specific y scaling.
        def fit_pixel_model(
            train_fields: list[int],
            fields_z: list[np.ndarray],
            centers_by_field: list[np.ndarray],
            w_local: int,
            y_by_field: list[np.ndarray],
        ) -> tuple[SGDRegressor, float, float]:
            y_all = np.concatenate([y_by_field[fid] for fid in train_fields], axis=0).astype(np.float64, copy=False)
            y_mu = float(np.mean(y_all))
            y_sd = float(np.std(y_all))
            if not np.isfinite(y_sd) or y_sd <= 0:
                y_sd = 1.0

            eta0_eff = float(pixel_eta0) / float(w_local)
            reg = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(pixel_alpha),
                learning_rate="invscaling",
                eta0=float(eta0_eff),
                power_t=float(pixel_power_t),
                max_iter=1,
                tol=None,
                fit_intercept=True,
                average=True,
                random_state=int(placebo_seed) + 0,
            )
            rloc = w_local // 2
            for _epoch in range(pixel_epochs):
                for fid in train_fields:
                    field = fields_z[fid]
                    centers = centers_by_field[fid]
                    y = y_by_field[fid]
                    win = sliding_window_view(field, (w_local, w_local))
                    perm = rng_pixel.permutation(centers.shape[0])
                    for start in range(0, centers.shape[0], pixel_batch_size):
                        idx = perm[start : start + pixel_batch_size]
                        cx = centers[idx, 0].astype(np.int64)
                        cy = centers[idx, 1].astype(np.int64)
                        Xb = win[cx - rloc, cy - rloc].reshape(len(idx), -1).astype(np.float32, copy=False)
                        reg.partial_fit(Xb, (y[idx] - y_mu) / y_sd)
            return reg, y_mu, y_sd

        def predict_pixel(reg: SGDRegressor, y_mu: float, y_sd: float, field_z: np.ndarray, centers: np.ndarray, w_local: int) -> np.ndarray:
            rloc = w_local // 2
            win = sliding_window_view(field_z, (w_local, w_local))
            out = np.empty((centers.shape[0],), dtype=np.float64)
            for start in range(0, centers.shape[0], pixel_batch_size):
                idx = slice(start, min(centers.shape[0], start + pixel_batch_size))
                cx = centers[idx, 0].astype(np.int64)
                cy = centers[idx, 1].astype(np.int64)
                Xb = win[cx - rloc, cy - rloc].reshape(len(cx), -1).astype(np.float32, copy=False)
                out[idx] = reg.predict(Xb) * y_sd + y_mu
            return out

        # Generate fields (rho in [0,1]) and full gx, plus z-scored rho for pixels.
        rho01_by_field: list[np.ndarray] = []
        rhoz_by_field: list[np.ndarray] = []
        gx_full_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
            sol = solve_poisson_periodic_fft_2d(rho01)
            rho01_by_field.append(rho01.astype(np.float64, copy=False))
            rhoz_by_field.append(_zscore_field(rho01).astype(np.float64, copy=False))
            gx_full_by_field.append(sol.gx.astype(np.float64, copy=False))

        # Choose centers once per field, valid for all windows (avoid borders).
        r_max = max(max(w_locals), w_big) // 2
        centers_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_cent = np.random.default_rng(seed + 333_333 + 1_000 * field_id)
            cx = rng_cent.integers(r_max, grid_size - r_max, size=patches_per_field, dtype=np.int64)
            cy = rng_cent.integers(r_max, grid_size - r_max, size=patches_per_field, dtype=np.int64)
            centers_by_field.append(np.column_stack([cx, cy]).astype(np.int64, copy=False))

        # y=gx_full at centers (independent of w_local).
        y_gx_by_field: list[np.ndarray] = []
        for fid in range(n_fields):
            centers = centers_by_field[fid]
            cx = centers[:, 0].astype(np.int64)
            cy = centers[:, 1].astype(np.int64)
            y_gx_by_field.append(gx_full_by_field[fid][cx, cy].astype(np.float64, copy=False))

        # Precompute local B features on rho01 for each w_local and each field (stored per w_local).
        B_by_w_field: dict[int, list[np.ndarray]] = {w: [] for w in w_locals}
        for w_local in w_locals:
            rloc = w_local // 2
            for field_id in range(n_fields):
                field = rho01_by_field[field_id]
                centers = centers_by_field[field_id]
                cx = centers[:, 0]
                cy = centers[:, 1]
                pref1 = _prefix_sum_2d(field)
                pref2 = _prefix_sum_2d(field * field)
                x0 = (cx - rloc).astype(np.int64)
                x1 = (cx + rloc + 1).astype(np.int64)
                y0 = (cy - rloc).astype(np.int64)
                y1 = (cy + rloc + 1).astype(np.int64)
                mass = _box_sum_2d(pref1, x0, x1, y0, y1)
                nvox = float(w_local * w_local)
                mean = mass / nvox
                sumsq = _box_sum_2d(pref2, x0, x1, y0, y1)
                var = np.maximum(0.0, sumsq / nvox - mean * mean)
                mass2 = mass * mass
                max_grid = ndimage.maximum_filter(field, size=w_local, mode="constant", cval=-np.inf)
                mx = max_grid[cx, cy]
                gx_f, gy_f = np.gradient(field)
                egrid = gx_f * gx_f + gy_f * gy_f
                eavg = ndimage.uniform_filter(egrid, size=w_local, mode="constant", cval=0.0)
                ge = eavg[cx, cy]
                B = np.column_stack([mass, mass2, var, mx, ge]).astype(np.float64, copy=False)
                B_by_w_field[w_local].append(B)

        # Precompute nonlocal geometric moments from a larger context w_big=129 (same for all w_local).
        xs = np.arange(grid_size, dtype=np.float64)[:, None]
        ys = np.arange(grid_size, dtype=np.float64)[None, :]
        X = np.broadcast_to(xs, (grid_size, grid_size))
        Y = np.broadcast_to(ys, (grid_size, grid_size))

        def _window_sums_at_centers(arr: np.ndarray, centers: np.ndarray, w: int) -> np.ndarray:
            area = float(w * w)
            grid = ndimage.uniform_filter(arr, size=w, mode="constant", cval=0.0) * area
            cx = centers[:, 0].astype(np.int64)
            cy = centers[:, 1].astype(np.int64)
            return grid[cx, cy].astype(np.float64, copy=False)

        nonlocal_names: list[str] = [
            "M33",
            "M65",
            "M129",
            "Ring1_M65m33",
            "Ring2_M129m65",
            "Dx33",
            "Dy33",
            "DxRing1",
            "DyRing1",
            "DxRing2",
            "DyRing2",
        ]
        if include_quadrupole:
            nonlocal_names += [
                "Qxx33",
                "Qyy33",
                "Qxy33",
                "QxxRing1",
                "QyyRing1",
                "QxyRing1",
                "QxxRing2",
                "QyyRing2",
                "QxyRing2",
            ]

        nonlocal_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            field = rho01_by_field[field_id]
            centers = centers_by_field[field_id]
            cx = centers[:, 0].astype(np.float64)
            cy = centers[:, 1].astype(np.float64)

            rho = field
            rho_x = rho * X
            rho_y = rho * Y
            rho_x2 = rho * (X * X)
            rho_y2 = rho * (Y * Y)
            rho_xy = rho * (X * Y)

            # Window sums at sizes 33/65/129.
            M33 = _window_sums_at_centers(rho, centers, 33)
            M65 = _window_sums_at_centers(rho, centers, 65)
            M129 = _window_sums_at_centers(rho, centers, 129)

            Sx33 = _window_sums_at_centers(rho_x, centers, 33)
            Sy33 = _window_sums_at_centers(rho_y, centers, 33)
            Sx65 = _window_sums_at_centers(rho_x, centers, 65)
            Sy65 = _window_sums_at_centers(rho_y, centers, 65)
            Sx129 = _window_sums_at_centers(rho_x, centers, 129)
            Sy129 = _window_sums_at_centers(rho_y, centers, 129)

            Dx33 = (Sx33 - cx * M33) / 33.0
            Dy33 = (Sy33 - cy * M33) / 33.0
            Dx65 = (Sx65 - cx * M65) / 65.0
            Dy65 = (Sy65 - cy * M65) / 65.0
            Dx129 = (Sx129 - cx * M129) / 129.0
            Dy129 = (Sy129 - cy * M129) / 129.0
            DxR1 = Dx65 - Dx33
            DyR1 = Dy65 - Dy33
            DxR2 = Dx129 - Dx65
            DyR2 = Dy129 - Dy65

            Ring1 = M65 - M33
            Ring2 = M129 - M65

            cols: list[np.ndarray] = [M33, M65, M129, Ring1, Ring2, Dx33, Dy33, DxR1, DyR1, DxR2, DyR2]

            if include_quadrupole:
                Sxx33 = _window_sums_at_centers(rho_x2, centers, 33)
                Syy33 = _window_sums_at_centers(rho_y2, centers, 33)
                Sxy33 = _window_sums_at_centers(rho_xy, centers, 33)
                Sxx65 = _window_sums_at_centers(rho_x2, centers, 65)
                Syy65 = _window_sums_at_centers(rho_y2, centers, 65)
                Sxy65 = _window_sums_at_centers(rho_xy, centers, 65)
                Sxx129 = _window_sums_at_centers(rho_x2, centers, 129)
                Syy129 = _window_sums_at_centers(rho_y2, centers, 129)
                Sxy129 = _window_sums_at_centers(rho_xy, centers, 129)

                Qxx33 = (Sxx33 - 2.0 * cx * Sx33 + (cx * cx) * M33) / (33.0 * 33.0)
                Qyy33 = (Syy33 - 2.0 * cy * Sy33 + (cy * cy) * M33) / (33.0 * 33.0)
                Qxy33 = (Sxy33 - cx * Sy33 - cy * Sx33 + (cx * cy) * M33) / (33.0 * 33.0)
                Qxx65 = (Sxx65 - 2.0 * cx * Sx65 + (cx * cx) * M65) / (65.0 * 65.0)
                Qyy65 = (Syy65 - 2.0 * cy * Sy65 + (cy * cy) * M65) / (65.0 * 65.0)
                Qxy65 = (Sxy65 - cx * Sy65 - cy * Sx65 + (cx * cy) * M65) / (65.0 * 65.0)
                Qxx129 = (Sxx129 - 2.0 * cx * Sx129 + (cx * cx) * M129) / (129.0 * 129.0)
                Qyy129 = (Syy129 - 2.0 * cy * Sy129 + (cy * cy) * M129) / (129.0 * 129.0)
                Qxy129 = (Sxy129 - cx * Sy129 - cy * Sx129 + (cx * cy) * M129) / (129.0 * 129.0)

                cols += [
                    Qxx33,
                    Qyy33,
                    Qxy33,
                    (Qxx65 - Qxx33),
                    (Qyy65 - Qyy33),
                    (Qxy65 - Qxy33),
                    (Qxx129 - Qxx65),
                    (Qyy129 - Qyy65),
                    (Qxy129 - Qxy65),
                ]

            F = np.column_stack(cols).astype(np.float64, copy=False)
            assert_finite("nonlocal_F", F)
            nonlocal_by_field.append(F)

        # Run configs: for each w_local, fit LOFO pixel baseline and evaluate residual model B vs C.
        rows_summary: list[dict[str, Any]] = []
        for w_local in w_locals:
            fold_rows: list[dict[str, Any]] = []
            pvals: list[float] = []

            for test_field in range(n_fields):
                train_fields = [i for i in range(n_fields) if i != test_field]
                reg, y_mu, y_sd = fit_pixel_model(train_fields, rhoz_by_field, centers_by_field, w_local, y_gx_by_field)

                # Baseline predictions and residual targets
                r_tr_list: list[np.ndarray] = []
                Btr_list: list[np.ndarray] = []
                Ftr_list: list[np.ndarray] = []
                for fid in train_fields:
                    y = y_gx_by_field[fid]
                    ypred = predict_pixel(reg, y_mu, y_sd, rhoz_by_field[fid], centers_by_field[fid], w_local)
                    r_tr_list.append((y - ypred).astype(np.float64, copy=False))
                    if b_resid_mode == "local_b":
                        Btr_list.append(B_by_w_field[w_local][fid])
                    Ftr_list.append(nonlocal_by_field[fid])
                r_tr = np.concatenate(r_tr_list, axis=0)
                Ftr_raw = np.concatenate(Ftr_list, axis=0)
                if b_resid_mode == "local_b":
                    Btr = np.concatenate(Btr_list, axis=0)
                else:
                    Btr = np.zeros((r_tr.shape[0], 0), dtype=np.float64)

                y_te = y_gx_by_field[test_field]
                ypred_te = predict_pixel(reg, y_mu, y_sd, rhoz_by_field[test_field], centers_by_field[test_field], w_local)
                r_te = (y_te - ypred_te).astype(np.float64, copy=False)
                if b_resid_mode == "local_b":
                    Bte = B_by_w_field[w_local][test_field]
                else:
                    Bte = np.zeros((r_te.shape[0], 0), dtype=np.float64)
                Fte_raw = nonlocal_by_field[test_field]

                # Baseline metrics for gx_full
                pearson_P = safe_corr_1d(y_te, ypred_te)
                relrmse_P = relrmse_1d(y_te, ypred_te)

                # Standardize B on train
                if Btr.shape[1] > 0:
                    B_mu = Btr.mean(axis=0)
                    B_sd = np.where(Btr.std(axis=0) > 0, Btr.std(axis=0), 1.0)
                    Btr_s = (Btr - B_mu) / B_sd
                    Bte_s = (Bte - B_mu) / B_sd
                else:
                    Btr_s = Btr
                    Bte_s = Bte

                # Nonlocal block is used as-is (no nonlinear expansions by default).
                F_mu = Ftr_raw.mean(axis=0)
                F_sd = np.where(Ftr_raw.std(axis=0) > 0, Ftr_raw.std(axis=0), 1.0)
                Ftr_s = (Ftr_raw - F_mu) / F_sd
                Fte_s = (Fte_raw - F_mu) / F_sd

                # Center residual target
                r_mu = float(np.mean(r_tr))
                rc = (r_tr - r_mu).reshape(-1, 1)

                # Fit B_resid
                if Btr_s.shape[1] > 0:
                    A = Btr_s.T @ Btr_s
                    uB = Btr_s.T @ rc
                    wB = np.linalg.solve(A + ridge_alpha * np.eye(A.shape[0]), uB).reshape(-1)
                    rB_te = Bte_s @ wB + r_mu
                else:
                    A = np.zeros((0, 0), dtype=np.float64)
                    uB = np.zeros((0, 1), dtype=np.float64)
                    rB_te = np.full_like(r_te, r_mu, dtype=np.float64)
                pB = safe_corr_1d(r_te, rB_te)
                rrB = relrmse_1d(r_te, rB_te)

                # Fit C_resid (real)
                if Btr_s.shape[1] > 0:
                    XtrC = np.concatenate([Btr_s, Ftr_s], axis=1)
                    XteC = np.concatenate([Bte_s, Fte_s], axis=1)
                else:
                    XtrC = Ftr_s
                    XteC = Fte_s
                XtX = XtrC.T @ XtrC
                Xty = XtrC.T @ rc
                wC = np.linalg.solve(XtX + ridge_alpha * np.eye(XtX.shape[0]), Xty).reshape(-1)
                rC_te = XteC @ wC + r_mu
                pC = safe_corr_1d(r_te, rC_te)
                rrC = relrmse_1d(r_te, rC_te)

                deltaP = float(pC - pB)
                deltaR = float(rrC - rrB)

                # Placebo permutations: shuffle nonlocal rows in train.
                Cmat = Ftr_s.T @ Ftr_s
                Z = np.concatenate([Btr_s, rc], axis=1)  # (n, dB+1)
                null_dP: list[float] = []
                for _ in range(n_perms):
                    pidx = rng_null.permutation(Ftr_s.shape[0])
                    Fp = Ftr_s[pidx]
                    M = Z.T @ Fp
                    D = M[:-1, :]
                    v = M[-1, :].reshape(-1, 1)
                    XtX_p = np.block([[A, D], [D.T, Cmat]])
                    Xty_p = np.vstack([uB, v])
                    wP = np.linalg.solve(XtX_p + ridge_alpha * np.eye(XtX_p.shape[0]), Xty_p).reshape(-1)
                    rP_te = XteC @ wP + r_mu
                    pP = safe_corr_1d(r_te, rP_te)
                    null_dP.append(float(pP - pB))
                null = np.asarray(null_dP, dtype=np.float64)
                p_emp = float((np.sum(null >= deltaP) + 1.0) / (float(n_perms) + 1.0))
                pvals.append(p_emp)

                fold_rows.append(
                    {
                        "field_id": int(test_field),
                        "pearson_P": float(pearson_P),
                        "relRMSE_P": float(relrmse_P),
                        "pearson_B": float(pB),
                        "relRMSE_B": float(rrB),
                        "pearson_C": float(pC),
                        "relRMSE_C": float(rrC),
                        "deltaP": float(deltaP),
                        "deltaR": float(deltaR),
                        "p_emp": float(p_emp),
                    }
                )

            dp = np.asarray([r0["deltaP"] for r0 in fold_rows], dtype=np.float64)
            dr = np.asarray([r0["deltaR"] for r0 in fold_rows], dtype=np.float64)
            pearP = np.asarray([r0["pearson_P"] for r0 in fold_rows], dtype=np.float64)
            rrP = np.asarray([r0["relRMSE_P"] for r0 in fold_rows], dtype=np.float64)
            n_pos = int(np.sum(dp > 0.0))
            fp = fisher_p(pvals)
            verdict = (float(dp.mean()) > 0.0) and (float(dr.mean()) < 0.0) and (fp < 0.05) and (n_pos >= 7)

            if b_resid_mode == "local_b":
                base = "B_resid=local_B(mass,mass2,var,max,grad_energy)"
            else:
                base = "B_resid=intercept"
            cfg_str = f"{base} + nonlocal_moments(w_big={w_big}, include_quadrupole={include_quadrupole})"
            rows_summary.append(
                {
                    "w_local": int(w_local),
                    "features_used": cfg_str,
                    "deltaP_mean": float(dp.mean()),
                    "deltaP_std": float(dp.std(ddof=1)) if len(dp) > 1 else 0.0,
                    "deltaR_mean": float(dr.mean()),
                    "deltaR_std": float(dr.std(ddof=1)) if len(dr) > 1 else 0.0,
                    "fisher_p": float(fp),
                    "n_pos": int(n_pos),
                    "verdict": bool(verdict),
                    "baseline_pearson_mean": float(pearP.mean()),
                    "baseline_pearson_std": float(pearP.std(ddof=1)) if len(pearP) > 1 else 0.0,
                    "baseline_relrmse_mean": float(rrP.mean()),
                    "baseline_relrmse_std": float(rrP.std(ddof=1)) if len(rrP) > 1 else 0.0,
                }
            )

            header = list(fold_rows[0].keys()) if fold_rows else []
            write_csv(paths.run_dir / f"lofo_by_field_wlocal{w_local}.csv", header, [[r0[h] for h in header] for r0 in fold_rows])

        # Summary markdown (baseline + deltas)
        md_base_rows: list[dict[str, str]] = []
        for r0 in rows_summary:
            md_base_rows.append(
                {
                    "w_local": str(int(r0["w_local"])),
                    "Pearson_P mean±std": f"{float(r0['baseline_pearson_mean']):.4f} ± {float(r0['baseline_pearson_std']):.4f}",
                    "relRMSE_P mean±std": f"{float(r0['baseline_relrmse_mean']):.4f} ± {float(r0['baseline_relrmse_std']):.4f}",
                }
            )
        md_rows: list[dict[str, str]] = []
        for r0 in rows_summary:
            md_rows.append(
                {
                    "w_local": str(int(r0["w_local"])),
                    "features_used": str(r0["features_used"]),
                    "ΔPearson mean±std": f"{float(r0['deltaP_mean']):.4f} ± {float(r0['deltaP_std']):.4f}",
                    "ΔrelRMSE mean±std": f"{float(r0['deltaR_mean']):.4f} ± {float(r0['deltaR_std']):.4f}",
                    "Fisher p": fmt(float(r0["fisher_p"])),
                    "#folds ΔP>0": f"{int(r0['n_pos'])}/{n_fields}",
                    "verdict": "PASS" if bool(r0["verdict"]) else "FAIL",
                }
            )

        summary_md = (
            "# E26 — Nonlocal geometric moments on full-g residual (LOFO)\n\n"
            f"- run: `{paths.run_dir}`\n"
            f"- grid_size={grid_size}, alpha={alpha}, n_fields={n_fields}, patches_per_field={patches_per_field}\n"
            f"- w_locals={w_locals}, w_big={w_big}, include_quadrupole={include_quadrupole}\n"
            f"- pixel: SGD L2 (ridge-like) alpha={pixel_alpha}, epochs={pixel_epochs}, batch={pixel_batch_size}\n"
            f"- residual ridge_alpha={ridge_alpha}, perms={n_perms}, B_resid_mode={b_resid_mode}\n"
            f"- nonlocal feature names: {nonlocal_names}\n\n"
            "## Baseline (pixels-only) on gx_full\n\n"
            + md_table(md_base_rows, ["w_local", "Pearson_P mean±std", "relRMSE_P mean±std"])
            + "\n\n## Residual model gain: C_resid(B + nonlocal) vs B_resid\n\n"
            + md_table(md_rows, ["w_local", "features_used", "ΔPearson mean±std", "ΔrelRMSE mean±std", "Fisher p", "#folds ΔP>0", "verdict"])
            + "\n"
        )
        (paths.run_dir / "summary_e26_nonlocal_moments_fullg_residual.md").write_text(summary_md, encoding="utf-8")

        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "grid_size": grid_size,
                "alpha": alpha,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "w_locals": w_locals,
                "w_big": w_big,
                "include_quadrupole": include_quadrupole,
                "nonlocal_names": nonlocal_names,
                "pixel": {
                    "solver": "sgd_l2",
                    "alpha": pixel_alpha,
                    "epochs": pixel_epochs,
                    "batch_size": pixel_batch_size,
                    "eta0": pixel_eta0,
                    "power_t": pixel_power_t,
                },
                "residual": {"ridge_alpha": ridge_alpha, "n_perms": n_perms, "b_resid_mode": b_resid_mode},
                "rows_summary": rows_summary,
            },
        )
        return paths

    if experiment == "e27":
        # Nonlocal moments ablation study on full-g residual (LOFO).
        #
        # Pipeline:
        # - Generate n_fields independent rho01 in [0,1] (2D 1/f) and compute full gx via Poisson FFT.
        # - Sample patches_per_field centers per field (same centers for all w_local/w_big).
        # - Baseline local predictor P_local: pixels-only ridge-like SGD on z-scored rho patches (w_local).
        # - Residual target: r = gx_true - gx_pred_P_local (scalar).
        # - Residual baseline B_resid: local_B (mass,mass2,var,max,grad_energy) on w_local patch.
        # - Residual augmented models C_resid: B_resid + nonlocal moments computed on w_big:
        #     C1 rings-only, C2 +dipoles, C3 +quadrupoles.
        #
        # For each (w_local, w_big, variant): report ΔPearson(C-B), ΔrelRMSE(C-B) on residual r (LOFO folds),
        # and a permutation placebo (shuffle nonlocal rows in TRAIN only, P perms/fold) with Fisher p-value.
        from numpy.lib.stride_tricks import sliding_window_view
        from scipy import ndimage, stats
        from sklearn.linear_model import SGDRegressor

        from .data_synth import solve_poisson_periodic_fft_2d

        grid_size = int(cfg.get("grid_size", 256))
        alpha = float(cfg.get("alpha", 2.0))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 10_000))
        w_locals = [int(x) for x in cfg.get("w_locals", [33, 65])]
        w_bigs = [int(x) for x in cfg.get("w_bigs", [65, 129, 193])]

        n_perms = int(cfg.get("n_perms", 200))
        ridge_alpha = float(cfg.get("ridge_alpha", 1.0))

        pixel_epochs = int(cfg.get("pixel_epochs", 2))
        pixel_batch_size = int(cfg.get("pixel_batch_size", 1024))
        pixel_alpha = float(cfg.get("pixel_alpha", 1e-4))
        pixel_eta0 = float(cfg.get("pixel_eta0", 0.01))
        pixel_power_t = float(cfg.get("pixel_power_t", 0.25))

        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if n_fields < 2:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        if any((w <= 0) or (w % 2 == 0) for w in w_locals):
            raise ValueError("w_locals must be odd positive ints")
        if any((w <= 0) or (w % 2 == 0) for w in w_bigs):
            raise ValueError("w_bigs must be odd positive ints")
        if any(w < 65 for w in w_bigs):
            raise ValueError("w_bigs must be >= 65 (uses fixed inner windows 33/65)")
        if any(w > grid_size for w in w_bigs):
            raise ValueError("w_bigs must be <= grid_size")
        if n_perms <= 0:
            raise ValueError("n_perms must be > 0")

        rng_pixel = np.random.default_rng(int(placebo_seed) + 101_111)
        rng_null = np.random.default_rng(int(placebo_seed) + 101_222)

        def fisher_p(ps: list[float]) -> float:
            if not ps:
                return float("nan")
            eps = 1.0 / float(n_perms + 1)
            ps2 = [min(1.0, max(float(p), eps)) for p in ps]
            stat = -2.0 * float(np.sum(np.log(ps2)))
            return float(stats.chi2.sf(stat, 2 * len(ps2)))

        def fmt(x: float) -> str:
            if not np.isfinite(x):
                return "nan"
            ax = abs(float(x))
            if ax != 0.0 and (ax < 1e-3 or ax >= 1e3):
                return f"{x:.3e}"
            return f"{x:.4f}"

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r0 in rows:
                out.append("| " + " | ".join(r0.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        def safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            am = a - float(a.mean())
            bm = b - float(b.mean())
            denom = float(np.linalg.norm(am) * np.linalg.norm(bm)) + 1e-12
            return float((am @ bm) / denom)

        def relrmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
            e = y_pred - y_true
            rmse = float(np.sqrt(np.mean(e * e)))
            sd = float(np.std(y_true))
            return rmse / (sd + 1e-12)

        # Pixel baseline: ridge-like SGD (scalar target), with fold-specific y scaling.
        def fit_pixel_model(
            train_fields: list[int],
            fields_z: list[np.ndarray],
            centers_by_field: list[np.ndarray],
            w_local: int,
            y_by_field: list[np.ndarray],
        ) -> tuple[SGDRegressor, float, float]:
            y_all = np.concatenate([y_by_field[fid] for fid in train_fields], axis=0).astype(np.float64, copy=False)
            y_mu = float(np.mean(y_all))
            y_sd = float(np.std(y_all))
            if not np.isfinite(y_sd) or y_sd <= 0:
                y_sd = 1.0

            eta0_eff = float(pixel_eta0) / float(w_local)
            reg = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(pixel_alpha),
                learning_rate="invscaling",
                eta0=float(eta0_eff),
                power_t=float(pixel_power_t),
                max_iter=1,
                tol=None,
                fit_intercept=True,
                average=True,
                random_state=int(placebo_seed) + 0,
            )
            rloc = w_local // 2
            for _epoch in range(pixel_epochs):
                for fid in train_fields:
                    field = fields_z[fid]
                    centers = centers_by_field[fid]
                    y = y_by_field[fid]
                    win = sliding_window_view(field, (w_local, w_local))
                    perm = rng_pixel.permutation(centers.shape[0])
                    for start in range(0, centers.shape[0], pixel_batch_size):
                        idx = perm[start : start + pixel_batch_size]
                        cx = centers[idx, 0].astype(np.int64)
                        cy = centers[idx, 1].astype(np.int64)
                        Xb = win[cx - rloc, cy - rloc].reshape(len(idx), -1).astype(np.float32, copy=False)
                        reg.partial_fit(Xb, (y[idx] - y_mu) / y_sd)
            return reg, y_mu, y_sd

        def predict_pixel(reg: SGDRegressor, y_mu: float, y_sd: float, field_z: np.ndarray, centers: np.ndarray, w_local: int) -> np.ndarray:
            rloc = w_local // 2
            win = sliding_window_view(field_z, (w_local, w_local))
            out = np.empty((centers.shape[0],), dtype=np.float64)
            for start in range(0, centers.shape[0], pixel_batch_size):
                idx = slice(start, min(centers.shape[0], start + pixel_batch_size))
                cx = centers[idx, 0].astype(np.int64)
                cy = centers[idx, 1].astype(np.int64)
                Xb = win[cx - rloc, cy - rloc].reshape(len(cx), -1).astype(np.float32, copy=False)
                out[idx] = reg.predict(Xb) * y_sd + y_mu
            return out

        # Generate fields (rho in [0,1]) and full gx, plus z-scored rho for pixels.
        rho01_by_field: list[np.ndarray] = []
        rhoz_by_field: list[np.ndarray] = []
        gx_full_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
            sol = solve_poisson_periodic_fft_2d(rho01)
            rho01_by_field.append(rho01.astype(np.float64, copy=False))
            rhoz_by_field.append(_zscore_field(rho01).astype(np.float64, copy=False))
            gx_full_by_field.append(sol.gx.astype(np.float64, copy=False))

        # Choose centers once per field, valid for all windows (avoid borders).
        r_max = max(max(w_locals), max(w_bigs)) // 2
        centers_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_cent = np.random.default_rng(seed + 444_444 + 1_000 * field_id)
            cx = rng_cent.integers(r_max, grid_size - r_max, size=patches_per_field, dtype=np.int64)
            cy = rng_cent.integers(r_max, grid_size - r_max, size=patches_per_field, dtype=np.int64)
            centers_by_field.append(np.column_stack([cx, cy]).astype(np.int64, copy=False))

        # y=gx_full at centers (independent of w_local/w_big).
        y_gx_by_field: list[np.ndarray] = []
        for fid in range(n_fields):
            centers = centers_by_field[fid]
            cx = centers[:, 0].astype(np.int64)
            cy = centers[:, 1].astype(np.int64)
            y_gx_by_field.append(gx_full_by_field[fid][cx, cy].astype(np.float64, copy=False))

        # Precompute local B features on rho01 for each w_local and each field (stored per w_local).
        B_by_w_field: dict[int, list[np.ndarray]] = {w: [] for w in w_locals}
        for w_local in w_locals:
            rloc = w_local // 2
            for field_id in range(n_fields):
                field = rho01_by_field[field_id]
                centers = centers_by_field[field_id]
                cx = centers[:, 0]
                cy = centers[:, 1]
                pref1 = _prefix_sum_2d(field)
                pref2 = _prefix_sum_2d(field * field)
                x0 = (cx - rloc).astype(np.int64)
                x1 = (cx + rloc + 1).astype(np.int64)
                y0 = (cy - rloc).astype(np.int64)
                y1 = (cy + rloc + 1).astype(np.int64)
                mass = _box_sum_2d(pref1, x0, x1, y0, y1)
                nvox = float(w_local * w_local)
                mean = mass / nvox
                sumsq = _box_sum_2d(pref2, x0, x1, y0, y1)
                var = np.maximum(0.0, sumsq / nvox - mean * mean)
                mass2 = mass * mass
                max_grid = ndimage.maximum_filter(field, size=w_local, mode="constant", cval=-np.inf)
                mx = max_grid[cx, cy]
                gx_f, gy_f = np.gradient(field)
                egrid = gx_f * gx_f + gy_f * gy_f
                eavg = ndimage.uniform_filter(egrid, size=w_local, mode="constant", cval=0.0)
                ge = eavg[cx, cy]
                B = np.column_stack([mass, mass2, var, mx, ge]).astype(np.float64, copy=False)
                B_by_w_field[w_local].append(B)

        # Nonlocal moment features are computed from nested windows:
        # inner windows fixed at 33 and 65, outer window is w_big.
        w_sizes = sorted(set([33, 65] + w_bigs))
        xs = np.arange(grid_size, dtype=np.float64)[:, None]
        ys = np.arange(grid_size, dtype=np.float64)[None, :]
        X = np.broadcast_to(xs, (grid_size, grid_size))
        Y = np.broadcast_to(ys, (grid_size, grid_size))

        def _window_sums_at_centers(arr: np.ndarray, centers: np.ndarray, w: int) -> np.ndarray:
            area = float(w * w)
            grid = ndimage.uniform_filter(arr, size=w, mode="constant", cval=0.0) * area
            cx = centers[:, 0].astype(np.int64)
            cy = centers[:, 1].astype(np.int64)
            return grid[cx, cy].astype(np.float64, copy=False)

        # Precompute full (C3) feature blocks for each w_big, per field.
        # Column order is always: rings, dipoles, quadrupoles (nested variants are prefixes).
        nonlocal_by_wbig_field: dict[int, list[np.ndarray]] = {w: [] for w in w_bigs}
        nonlocal_names_by_wbig: dict[int, list[str]] = {}

        for w_big in w_bigs:
            if w_big == 65:
                names = [
                    "M33",
                    "M65",
                    "Ring1_M65m33",
                    "Dx33",
                    "Dy33",
                    "DxRing1",
                    "DyRing1",
                    "Qxx33",
                    "Qyy33",
                    "Qxy33",
                    "QxxRing1",
                    "QyyRing1",
                    "QxyRing1",
                ]
            else:
                names = [
                    "M33",
                    "M65",
                    f"M{w_big}",
                    "Ring1_M65m33",
                    f"Ring2_M{w_big}m65",
                    "Dx33",
                    "Dy33",
                    "DxRing1",
                    "DyRing1",
                    "DxRing2",
                    "DyRing2",
                    "Qxx33",
                    "Qyy33",
                    "Qxy33",
                    "QxxRing1",
                    "QyyRing1",
                    "QxyRing1",
                    "QxxRing2",
                    "QyyRing2",
                    "QxyRing2",
                ]
            nonlocal_names_by_wbig[w_big] = names

        for field_id in range(n_fields):
            rho = rho01_by_field[field_id]
            centers = centers_by_field[field_id]
            cx = centers[:, 0].astype(np.float64)
            cy = centers[:, 1].astype(np.float64)

            rho_x = rho * X
            rho_y = rho * Y
            rho_x2 = rho * (X * X)
            rho_y2 = rho * (Y * Y)
            rho_xy = rho * (X * Y)

            M: dict[int, np.ndarray] = {}
            Sx: dict[int, np.ndarray] = {}
            Sy: dict[int, np.ndarray] = {}
            Sxx: dict[int, np.ndarray] = {}
            Syy: dict[int, np.ndarray] = {}
            Sxy: dict[int, np.ndarray] = {}
            for w in w_sizes:
                M[w] = _window_sums_at_centers(rho, centers, w)
                Sx[w] = _window_sums_at_centers(rho_x, centers, w)
                Sy[w] = _window_sums_at_centers(rho_y, centers, w)
                Sxx[w] = _window_sums_at_centers(rho_x2, centers, w)
                Syy[w] = _window_sums_at_centers(rho_y2, centers, w)
                Sxy[w] = _window_sums_at_centers(rho_xy, centers, w)

            Dx: dict[int, np.ndarray] = {}
            Dy: dict[int, np.ndarray] = {}
            Qxx: dict[int, np.ndarray] = {}
            Qyy: dict[int, np.ndarray] = {}
            Qxy: dict[int, np.ndarray] = {}
            for w in w_sizes:
                Dx[w] = (Sx[w] - cx * M[w]) / float(w)
                Dy[w] = (Sy[w] - cy * M[w]) / float(w)
                ww2 = float(w * w)
                Qxx[w] = (Sxx[w] - 2.0 * cx * Sx[w] + (cx * cx) * M[w]) / ww2
                Qyy[w] = (Syy[w] - 2.0 * cy * Sy[w] + (cy * cy) * M[w]) / ww2
                Qxy[w] = (Sxy[w] - cx * Sy[w] - cy * Sx[w] + (cx * cy) * M[w]) / ww2

            for w_big in w_bigs:
                if w_big == 65:
                    Ring1 = M[65] - M[33]
                    DxRing1 = Dx[65] - Dx[33]
                    DyRing1 = Dy[65] - Dy[33]
                    QxxRing1 = Qxx[65] - Qxx[33]
                    QyyRing1 = Qyy[65] - Qyy[33]
                    QxyRing1 = Qxy[65] - Qxy[33]
                    cols = [
                        M[33],
                        M[65],
                        Ring1,
                        Dx[33],
                        Dy[33],
                        DxRing1,
                        DyRing1,
                        Qxx[33],
                        Qyy[33],
                        Qxy[33],
                        QxxRing1,
                        QyyRing1,
                        QxyRing1,
                    ]
                else:
                    Ring1 = M[65] - M[33]
                    Ring2 = M[w_big] - M[65]
                    DxRing1 = Dx[65] - Dx[33]
                    DyRing1 = Dy[65] - Dy[33]
                    DxRing2 = Dx[w_big] - Dx[65]
                    DyRing2 = Dy[w_big] - Dy[65]
                    QxxRing1 = Qxx[65] - Qxx[33]
                    QyyRing1 = Qyy[65] - Qyy[33]
                    QxyRing1 = Qxy[65] - Qxy[33]
                    QxxRing2 = Qxx[w_big] - Qxx[65]
                    QyyRing2 = Qyy[w_big] - Qyy[65]
                    QxyRing2 = Qxy[w_big] - Qxy[65]
                    cols = [
                        M[33],
                        M[65],
                        M[w_big],
                        Ring1,
                        Ring2,
                        Dx[33],
                        Dy[33],
                        DxRing1,
                        DyRing1,
                        DxRing2,
                        DyRing2,
                        Qxx[33],
                        Qyy[33],
                        Qxy[33],
                        QxxRing1,
                        QyyRing1,
                        QxyRing1,
                        QxxRing2,
                        QyyRing2,
                        QxyRing2,
                    ]
                Ffull = np.column_stack(cols).astype(np.float64, copy=False)
                assert_finite(f"nonlocal_full_w{w_big}", Ffull)
                nonlocal_by_wbig_field[w_big].append(Ffull)

        # Variant definitions: nested prefixes of the full feature block.
        variant_order = ["C1_rings", "C2_rings+dipoles", "C3_rings+dipoles+quadrupoles"]
        ncols_by_variant_wbig: dict[tuple[int, str], int] = {}
        for w_big in w_bigs:
            if w_big == 65:
                n_ring, n_dip, n_quad = 3, 4, 6
            else:
                n_ring, n_dip, n_quad = 5, 6, 9
            ncols_by_variant_wbig[(w_big, "C1_rings")] = n_ring
            ncols_by_variant_wbig[(w_big, "C2_rings+dipoles")] = n_ring + n_dip
            ncols_by_variant_wbig[(w_big, "C3_rings+dipoles+quadrupoles")] = n_ring + n_dip + n_quad

        # Evaluate per w_local, per (w_big, variant) across LOFO folds.
        rows_summary: list[dict[str, Any]] = []
        for w_local in w_locals:
            # Aggregators across folds for each combination.
            combos = [(w_big, var) for w_big in w_bigs for var in variant_order]
            dp_by_combo: dict[tuple[int, str], list[float]] = {c: [] for c in combos}
            dr_by_combo: dict[tuple[int, str], list[float]] = {c: [] for c in combos}
            pvals_by_combo: dict[tuple[int, str], list[float]] = {c: [] for c in combos}

            for test_field in range(n_fields):
                train_fields = [i for i in range(n_fields) if i != test_field]
                reg, y_mu, y_sd = fit_pixel_model(train_fields, rhoz_by_field, centers_by_field, w_local, y_gx_by_field)

                # residual targets
                r_tr_list: list[np.ndarray] = []
                Btr_list: list[np.ndarray] = []
                for fid in train_fields:
                    y = y_gx_by_field[fid]
                    ypred = predict_pixel(reg, y_mu, y_sd, rhoz_by_field[fid], centers_by_field[fid], w_local)
                    r_tr_list.append((y - ypred).astype(np.float64, copy=False))
                    Btr_list.append(B_by_w_field[w_local][fid])
                r_tr = np.concatenate(r_tr_list, axis=0)
                Btr = np.concatenate(Btr_list, axis=0)

                y_te = y_gx_by_field[test_field]
                ypred_te = predict_pixel(reg, y_mu, y_sd, rhoz_by_field[test_field], centers_by_field[test_field], w_local)
                r_te = (y_te - ypred_te).astype(np.float64, copy=False)
                Bte = B_by_w_field[w_local][test_field]

                # Standardize B on train
                B_mu = Btr.mean(axis=0)
                B_sd = np.where(Btr.std(axis=0) > 0, Btr.std(axis=0), 1.0)
                Btr_s = (Btr - B_mu) / B_sd
                Bte_s = (Bte - B_mu) / B_sd

                # Center residual target
                r_mu = float(np.mean(r_tr))
                rc = (r_tr - r_mu).reshape(-1, 1)

                # Fit B_resid
                A = Btr_s.T @ Btr_s
                uB = Btr_s.T @ rc
                wB = np.linalg.solve(A + ridge_alpha * np.eye(A.shape[0]), uB).reshape(-1)
                rB_te = Bte_s @ wB + r_mu
                pB = safe_corr_1d(r_te, rB_te)
                rrB = relrmse_1d(r_te, rB_te)

                # Prepare Z (for fast permutation block construction)
                Z = np.concatenate([Btr_s, rc], axis=1)  # (n_train, dB+1)
                ZT = Z.T
                n_train = Btr_s.shape[0]

                # For each w_big, compute real deltas and run permutations (shared perms across variants).
                deltaP_real: dict[tuple[int, str], float] = {}

                # Precompute centered r_te for correlation computations (placebo Pearson only).
                rt = (r_te - float(r_te.mean())).astype(np.float64, copy=False)
                norm_rt = float(np.linalg.norm(rt)) + 1e-12

                # Cache per w_big to avoid recomputing in each perm loop.
                cache_by_wbig: dict[int, dict[str, Any]] = {}
                for w_big in w_bigs:
                    Ftr_raw_full = np.concatenate([nonlocal_by_wbig_field[w_big][fid] for fid in train_fields], axis=0)
                    Fte_raw_full = nonlocal_by_wbig_field[w_big][test_field]

                    F_mu = Ftr_raw_full.mean(axis=0)
                    F_sd = np.where(Ftr_raw_full.std(axis=0) > 0, Ftr_raw_full.std(axis=0), 1.0)
                    Ftr_s_full = (Ftr_raw_full - F_mu) / F_sd
                    Fte_s_full = (Fte_raw_full - F_mu) / F_sd
                    assert_finite(f"Ftr_s_full_w{w_big}", Ftr_s_full)
                    assert_finite(f"Fte_s_full_w{w_big}", Fte_s_full)

                    C_full = Ftr_s_full.T @ Ftr_s_full

                    # Precompute test centered cross-products for fast Pearson: Xc^T rt and Xc^T Xc.
                    Xte_full = np.concatenate([Bte_s, Fte_s_full], axis=1)  # (n_test, dB+dF)
                    mX = Xte_full.mean(axis=0)
                    Xc = Xte_full - mX
                    t_full = (Xc.T @ rt.reshape(-1, 1)).reshape(-1)
                    S_full = Xc.T @ Xc
                    cache_by_wbig[w_big] = {
                        "Ftr_s_full": Ftr_s_full,
                        "Fte_s_full": Fte_s_full,
                        "C_full": C_full,
                        "t_full": t_full,
                        "S_full": S_full,
                    }

                    # Real fits per variant (explicit predictions; cheap).
                    for var in variant_order:
                        k = ncols_by_variant_wbig[(w_big, var)]
                        Ftr_s = Ftr_s_full[:, :k]
                        Fte_s = Fte_s_full[:, :k]

                        XtrC = np.concatenate([Btr_s, Ftr_s], axis=1)
                        XteC = np.concatenate([Bte_s, Fte_s], axis=1)
                        XtX = XtrC.T @ XtrC
                        Xty = XtrC.T @ rc
                        wC = np.linalg.solve(XtX + ridge_alpha * np.eye(XtX.shape[0]), Xty).reshape(-1)
                        rC_te = XteC @ wC + r_mu
                        pC = safe_corr_1d(r_te, rC_te)
                        rrC = relrmse_1d(r_te, rC_te)

                        dp = float(pC - pB)
                        dr = float(rrC - rrB)
                        dp_by_combo[(w_big, var)].append(dp)
                        dr_by_combo[(w_big, var)].append(dr)
                        deltaP_real[(w_big, var)] = dp

                # Permutation placebo: generate perms once per fold and reuse for all variants.
                counts_ge: dict[tuple[int, str], int] = {(w_big, var): 0 for (w_big, var) in combos}
                for _ in range(n_perms):
                    pidx = rng_null.permutation(n_train)
                    for w_big in w_bigs:
                        c = cache_by_wbig[w_big]
                        Ftr_s_full = c["Ftr_s_full"]
                        C_full = c["C_full"]
                        t_full = c["t_full"]
                        S_full = c["S_full"]

                        Fp_full = Ftr_s_full[pidx]
                        M_full = ZT @ Fp_full  # (dB+1, dF_full)
                        D_full = M_full[:-1, :]
                        v_full = M_full[-1, :].reshape(-1, 1)

                        for var in variant_order:
                            k = ncols_by_variant_wbig[(w_big, var)]
                            D = D_full[:, :k]
                            v = v_full[:k, :]
                            Cmat = C_full[:k, :k]

                            XtX_p = np.block([[A, D], [D.T, Cmat]])
                            Xty_p = np.vstack([uB, v])
                            wP = np.linalg.solve(XtX_p + ridge_alpha * np.eye(XtX_p.shape[0]), Xty_p).reshape(-1)

                            # Pearson on test without materializing predictions: corr(rt, (Xc @ wP)).
                            # Full X columns are [B, F_full]; variant uses [B, F_prefix(k)].
                            dB = Bte_s.shape[1]
                            idx = np.r_[np.arange(dB), dB + np.arange(k)]
                            t_sub = t_full[idx]
                            S_sub = S_full[np.ix_(idx, idx)]
                            denom = norm_rt * (float(np.sqrt(max(0.0, wP @ (S_sub @ wP)))) + 1e-12)
                            pP = float((t_sub @ wP) / denom)

                            null_delta = pP - pB
                            if null_delta >= float(deltaP_real[(w_big, var)]):
                                counts_ge[(w_big, var)] += 1

                # Store fold p-values per combo (empirical, with +1 smoothing).
                for w_big, var in combos:
                    cge = counts_ge[(w_big, var)]
                    p_emp = float((cge + 1.0) / (float(n_perms) + 1.0))
                    pvals_by_combo[(w_big, var)].append(p_emp)

            # Aggregate across folds for this w_local and emit rows for all combos.
            for w_big, var in combos:
                dp = np.asarray(dp_by_combo[(w_big, var)], dtype=np.float64)
                dr = np.asarray(dr_by_combo[(w_big, var)], dtype=np.float64)
                n_pos = int(np.sum(dp > 0.0))
                fp = fisher_p(pvals_by_combo[(w_big, var)])
                verdict = (float(dp.mean()) > 0.0) and (float(dr.mean()) < 0.0) and (fp < 0.05) and (n_pos >= 7)
                rows_summary.append(
                    {
                        "w_local": int(w_local),
                        "w_big": int(w_big),
                        "variant": str(var),
                        "deltaP_mean": float(dp.mean()),
                        "deltaP_std": float(dp.std(ddof=1)) if len(dp) > 1 else 0.0,
                        "deltaR_mean": float(dr.mean()),
                        "deltaR_std": float(dr.std(ddof=1)) if len(dr) > 1 else 0.0,
                        "fisher_p": float(fp),
                        "n_pos": int(n_pos),
                        "verdict": bool(verdict),
                    }
                )

        # Winner lines: smallest (w_big, variant) that PASS, per w_local.
        winner_lines: list[str] = []
        for w_local in w_locals:
            passed = [r0 for r0 in rows_summary if int(r0["w_local"]) == int(w_local) and bool(r0["verdict"])]
            if not passed:
                winner_lines.append(f"- w_local={w_local}: no PASS")
                continue
            order_v = {v: i for i, v in enumerate(variant_order)}
            passed.sort(key=lambda r0: (int(r0["w_big"]), order_v.get(str(r0["variant"]), 999)))
            best = passed[0]
            winner_lines.append(
                f"- w_local={w_local}: best={best['variant']} at w_big={best['w_big']} (ΔPearson_mean={fmt(best['deltaP_mean'])}, Fisher p={fmt(best['fisher_p'])})"
            )

        # Summary markdown
        md_rows: list[dict[str, str]] = []
        for r0 in rows_summary:
            md_rows.append(
                {
                    "w_local": str(int(r0["w_local"])),
                    "w_big": str(int(r0["w_big"])),
                    "variant": str(r0["variant"]),
                    "ΔPearson mean±std": f"{float(r0['deltaP_mean']):.4f} ± {float(r0['deltaP_std']):.4f}",
                    "ΔrelRMSE mean±std": f"{float(r0['deltaR_mean']):.4f} ± {float(r0['deltaR_std']):.4f}",
                    "Fisher p": fmt(float(r0["fisher_p"])),
                    "#folds ΔP>0": f"{int(r0['n_pos'])}/{n_fields}",
                    "verdict": "PASS" if bool(r0["verdict"]) else "FAIL",
                }
            )

        summary_md = (
            "# E27 — Nonlocal moments ablation (full-g residual, LOFO)\n\n"
            f"- run: `{paths.run_dir}`\n"
            f"- grid_size={grid_size}, alpha={alpha}, n_fields={n_fields}, patches_per_field={patches_per_field}\n"
            f"- w_locals={w_locals}, w_bigs={w_bigs}\n"
            f"- pixel: SGD L2 (ridge-like) alpha={pixel_alpha}, epochs={pixel_epochs}, batch={pixel_batch_size}\n"
            f"- residual ridge_alpha={ridge_alpha}, perms={n_perms}\n\n"
            + md_table(
                md_rows,
                ["w_local", "w_big", "variant", "ΔPearson mean±std", "ΔrelRMSE mean±std", "Fisher p", "#folds ΔP>0", "verdict"],
            )
            + "\n\n## Winners\n\n"
            + "\n".join(winner_lines)
            + "\n"
        )
        (paths.run_dir / "summary_e27_nonlocal_ablation.md").write_text(summary_md, encoding="utf-8")

        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "grid_size": grid_size,
                "alpha": alpha,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "w_locals": w_locals,
                "w_bigs": w_bigs,
                "pixel": {
                    "solver": "sgd_l2",
                    "alpha": pixel_alpha,
                    "epochs": pixel_epochs,
                    "batch_size": pixel_batch_size,
                    "eta0": pixel_eta0,
                    "power_t": pixel_power_t,
                },
                "residual": {"ridge_alpha": ridge_alpha, "n_perms": n_perms},
                "rows_summary": rows_summary,
                "nonlocal_names_by_wbig": nonlocal_names_by_wbig,
            },
        )
        return paths

    if experiment == "e28":
        # Diagnose E27 anomaly: w_local=65 with w_big in {65,129}, ridge-alpha sweep, conditioning checks,
        # and "ring mask" visualization to verify the meaning of "rings".
        from numpy.lib.stride_tricks import sliding_window_view
        from scipy import ndimage, stats
        from sklearn.linear_model import SGDRegressor

        import matplotlib.pyplot as plt

        from .data_synth import solve_poisson_periodic_fft_2d
        # (no extra feature imports for E28)

        grid_size = int(cfg.get("grid_size", 256))
        alpha = float(cfg.get("alpha", 2.0))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 10_000))
        w_local = _require_odd("w_local", int(cfg.get("w_local", 65)))
        w_bigs = [int(x) for x in cfg.get("w_bigs", [65, 129])]
        ridge_alpha_grid = [float(x) for x in cfg.get("ridge_alpha_grid", [1e-6, 1e-4, 1e-2, 1e0, 1e2])]
        n_perms = int(cfg.get("n_perms", 100))
        diag_fold = int(cfg.get("diag_fold", 0))

        pixel_epochs = int(cfg.get("pixel_epochs", 2))
        pixel_batch_size = int(cfg.get("pixel_batch_size", 1024))
        pixel_alpha = float(cfg.get("pixel_alpha", 1e-4))
        pixel_eta0 = float(cfg.get("pixel_eta0", 0.01))
        pixel_power_t = float(cfg.get("pixel_power_t", 0.25))

        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if n_fields < 2:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        if w_local <= 0 or (w_local % 2) == 0:
            raise ValueError("w_local must be an odd positive int")
        w_bigs = [_require_odd("w_big", int(w)) for w in w_bigs]
        if any(w not in {65, 129} for w in w_bigs):
            raise ValueError("E28 focuses on w_bigs in {65,129}")
        if any(w > grid_size for w in w_bigs):
            raise ValueError("w_bigs must be <= grid_size")
        if n_perms <= 0:
            raise ValueError("n_perms must be > 0")
        if diag_fold < 0 or diag_fold >= n_fields:
            raise ValueError("diag_fold must be in [0,n_fields)")
        if any(a <= 0 for a in ridge_alpha_grid):
            raise ValueError("ridge_alpha_grid must be positive")

        # Match E27 RNG offsets so the residual baseline is comparable.
        rng_pixel = np.random.default_rng(int(placebo_seed) + 101_111)
        rng_null = np.random.default_rng(int(placebo_seed) + 101_222)

        variants = ["C1_rings", "C2_rings+dipoles", "C3_rings+dipoles+quadrupoles"]
        variant_order = {v: i for i, v in enumerate(variants)}
        # Column prefixes in our nonlocal block definition (same as E27).
        k_by_wbig_variant: dict[tuple[int, str], int] = {
            (65, "C1_rings"): 3,
            (65, "C2_rings+dipoles"): 7,
            (65, "C3_rings+dipoles+quadrupoles"): 13,
            (129, "C1_rings"): 5,
            (129, "C2_rings+dipoles"): 11,
            (129, "C3_rings+dipoles+quadrupoles"): 20,
        }

        def fisher_p(ps: list[float]) -> float:
            if not ps:
                return float("nan")
            eps = 1.0 / float(n_perms + 1)
            ps2 = [min(1.0, max(float(p), eps)) for p in ps]
            stat = -2.0 * float(np.sum(np.log(ps2)))
            return float(stats.chi2.sf(stat, 2 * len(ps2)))

        def fmt(x: float) -> str:
            if not np.isfinite(x):
                return "nan"
            ax = abs(float(x))
            if ax != 0.0 and (ax < 1e-3 or ax >= 1e3):
                return f"{x:.3e}"
            return f"{x:.4f}"

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r0 in rows:
                out.append("| " + " | ".join(r0.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        def safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            am = a - float(a.mean())
            bm = b - float(b.mean())
            denom = float(np.linalg.norm(am) * np.linalg.norm(bm)) + 1e-12
            return float((am @ bm) / denom)

        def relrmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
            e = y_pred - y_true
            rmse = float(np.sqrt(np.mean(e * e)))
            sd = float(np.std(y_true))
            return rmse / (sd + 1e-12)

        def solve_ridge(XtX: np.ndarray, Xty: np.ndarray, alpha_r: float) -> np.ndarray:
            try:
                return np.linalg.solve(XtX + float(alpha_r) * np.eye(XtX.shape[0]), Xty)
            except np.linalg.LinAlgError:
                w, *_ = np.linalg.lstsq(XtX + float(alpha_r) * np.eye(XtX.shape[0]), Xty, rcond=None)
                return w

        # Pixel baseline: ridge-like SGD (scalar target), with fold-specific y scaling.
        def fit_pixel_model(
            train_fields: list[int],
            fields_z: list[np.ndarray],
            centers_by_field: list[np.ndarray],
            w: int,
            y_by_field: list[np.ndarray],
        ) -> tuple[SGDRegressor, float, float]:
            y_all = np.concatenate([y_by_field[fid] for fid in train_fields], axis=0).astype(np.float64, copy=False)
            y_mu = float(np.mean(y_all))
            y_sd = float(np.std(y_all))
            if not np.isfinite(y_sd) or y_sd <= 0:
                y_sd = 1.0

            eta0_eff = float(pixel_eta0) / float(w)
            reg = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(pixel_alpha),
                learning_rate="invscaling",
                eta0=float(eta0_eff),
                power_t=float(pixel_power_t),
                max_iter=1,
                tol=None,
                fit_intercept=True,
                average=True,
                random_state=int(placebo_seed) + 0,
            )
            rloc = w // 2
            for _epoch in range(pixel_epochs):
                for fid in train_fields:
                    field = fields_z[fid]
                    centers = centers_by_field[fid]
                    y = y_by_field[fid]
                    win = sliding_window_view(field, (w, w))
                    perm = rng_pixel.permutation(centers.shape[0])
                    for start in range(0, centers.shape[0], pixel_batch_size):
                        idx = perm[start : start + pixel_batch_size]
                        cx = centers[idx, 0].astype(np.int64)
                        cy = centers[idx, 1].astype(np.int64)
                        Xb = win[cx - rloc, cy - rloc].reshape(len(idx), -1).astype(np.float32, copy=False)
                        reg.partial_fit(Xb, (y[idx] - y_mu) / y_sd)
            return reg, y_mu, y_sd

        def predict_pixel(reg: SGDRegressor, y_mu: float, y_sd: float, field_z: np.ndarray, centers: np.ndarray, w: int) -> np.ndarray:
            rloc = w // 2
            win = sliding_window_view(field_z, (w, w))
            out = np.empty((centers.shape[0],), dtype=np.float64)
            for start in range(0, centers.shape[0], pixel_batch_size):
                idx = slice(start, min(centers.shape[0], start + pixel_batch_size))
                cx = centers[idx, 0].astype(np.int64)
                cy = centers[idx, 1].astype(np.int64)
                Xb = win[cx - rloc, cy - rloc].reshape(len(cx), -1).astype(np.float32, copy=False)
                out[idx] = reg.predict(Xb) * y_sd + y_mu
            return out

        # Generate fields (rho in [0,1]) and full gx, plus z-scored rho for pixels.
        rho01_by_field: list[np.ndarray] = []
        rhoz_by_field: list[np.ndarray] = []
        gx_full_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
            sol = solve_poisson_periodic_fft_2d(rho01)
            rho01_by_field.append(rho01.astype(np.float64, copy=False))
            rhoz_by_field.append(_zscore_field(rho01).astype(np.float64, copy=False))
            gx_full_by_field.append(sol.gx.astype(np.float64, copy=False))

        # Centers (match E27 seeding convention).
        r_max = max(w_local, max(w_bigs)) // 2
        centers_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_cent = np.random.default_rng(seed + 444_444 + 1_000 * field_id)
            cx = rng_cent.integers(r_max, grid_size - r_max, size=patches_per_field, dtype=np.int64)
            cy = rng_cent.integers(r_max, grid_size - r_max, size=patches_per_field, dtype=np.int64)
            centers_by_field.append(np.column_stack([cx, cy]).astype(np.int64, copy=False))

        # y=gx_full at centers.
        y_gx_by_field: list[np.ndarray] = []
        for fid in range(n_fields):
            centers = centers_by_field[fid]
            cx = centers[:, 0].astype(np.int64)
            cy = centers[:, 1].astype(np.int64)
            y_gx_by_field.append(gx_full_by_field[fid][cx, cy].astype(np.float64, copy=False))

        # Local B features (w_local only).
        B_by_field: list[np.ndarray] = []
        rloc = w_local // 2
        for field_id in range(n_fields):
            field = rho01_by_field[field_id]
            centers = centers_by_field[field_id]
            cx = centers[:, 0]
            cy = centers[:, 1]
            pref1 = _prefix_sum_2d(field)
            pref2 = _prefix_sum_2d(field * field)
            x0 = (cx - rloc).astype(np.int64)
            x1 = (cx + rloc + 1).astype(np.int64)
            y0 = (cy - rloc).astype(np.int64)
            y1 = (cy + rloc + 1).astype(np.int64)
            mass = _box_sum_2d(pref1, x0, x1, y0, y1)
            nvox = float(w_local * w_local)
            mean = mass / nvox
            sumsq = _box_sum_2d(pref2, x0, x1, y0, y1)
            var = np.maximum(0.0, sumsq / nvox - mean * mean)
            mass2 = mass * mass
            max_grid = ndimage.maximum_filter(field, size=w_local, mode="constant", cval=-np.inf)
            mx = max_grid[cx, cy]
            gx_f, gy_f = np.gradient(field)
            egrid = gx_f * gx_f + gy_f * gy_f
            eavg = ndimage.uniform_filter(egrid, size=w_local, mode="constant", cval=0.0)
            ge = eavg[cx, cy]
            B = np.column_stack([mass, mass2, var, mx, ge]).astype(np.float64, copy=False)
            B_by_field.append(B)

        # Nonlocal moments blocks for w_big=65 and 129 (same definitions as E27).
        w_sizes = [33, 65, 129]
        xs = np.arange(grid_size, dtype=np.float64)[:, None]
        ys = np.arange(grid_size, dtype=np.float64)[None, :]
        X = np.broadcast_to(xs, (grid_size, grid_size))
        Y = np.broadcast_to(ys, (grid_size, grid_size))

        def _window_sums_at_centers(arr: np.ndarray, centers: np.ndarray, w: int) -> np.ndarray:
            area = float(w * w)
            grid = ndimage.uniform_filter(arr, size=w, mode="constant", cval=0.0) * area
            cx = centers[:, 0].astype(np.int64)
            cy = centers[:, 1].astype(np.int64)
            return grid[cx, cy].astype(np.float64, copy=False)

        names_full_by_wbig: dict[int, list[str]] = {
            65: [
                "M33",
                "M65",
                "Ring1_M65m33",
                "Dx33",
                "Dy33",
                "DxRing1",
                "DyRing1",
                "Qxx33",
                "Qyy33",
                "Qxy33",
                "QxxRing1",
                "QyyRing1",
                "QxyRing1",
            ],
            129: [
                "M33",
                "M65",
                "M129",
                "Ring1_M65m33",
                "Ring2_M129m65",
                "Dx33",
                "Dy33",
                "DxRing1",
                "DyRing1",
                "DxRing2",
                "DyRing2",
                "Qxx33",
                "Qyy33",
                "Qxy33",
                "QxxRing1",
                "QyyRing1",
                "QxyRing1",
                "QxxRing2",
                "QyyRing2",
                "QxyRing2",
            ],
        }

        Ffull_by_wbig_field: dict[int, list[np.ndarray]] = {65: [], 129: []}
        for field_id in range(n_fields):
            rho = rho01_by_field[field_id]
            centers = centers_by_field[field_id]
            cx = centers[:, 0].astype(np.float64)
            cy = centers[:, 1].astype(np.float64)

            rho_x = rho * X
            rho_y = rho * Y
            rho_x2 = rho * (X * X)
            rho_y2 = rho * (Y * Y)
            rho_xy = rho * (X * Y)

            M: dict[int, np.ndarray] = {}
            Sx: dict[int, np.ndarray] = {}
            Sy: dict[int, np.ndarray] = {}
            Sxx: dict[int, np.ndarray] = {}
            Syy: dict[int, np.ndarray] = {}
            Sxy: dict[int, np.ndarray] = {}
            for w in w_sizes:
                M[w] = _window_sums_at_centers(rho, centers, w)
                Sx[w] = _window_sums_at_centers(rho_x, centers, w)
                Sy[w] = _window_sums_at_centers(rho_y, centers, w)
                Sxx[w] = _window_sums_at_centers(rho_x2, centers, w)
                Syy[w] = _window_sums_at_centers(rho_y2, centers, w)
                Sxy[w] = _window_sums_at_centers(rho_xy, centers, w)

            Dx: dict[int, np.ndarray] = {}
            Dy: dict[int, np.ndarray] = {}
            Qxx: dict[int, np.ndarray] = {}
            Qyy: dict[int, np.ndarray] = {}
            Qxy: dict[int, np.ndarray] = {}
            for w in w_sizes:
                Dx[w] = (Sx[w] - cx * M[w]) / float(w)
                Dy[w] = (Sy[w] - cy * M[w]) / float(w)
                ww2 = float(w * w)
                Qxx[w] = (Sxx[w] - 2.0 * cx * Sx[w] + (cx * cx) * M[w]) / ww2
                Qyy[w] = (Syy[w] - 2.0 * cy * Sy[w] + (cy * cy) * M[w]) / ww2
                Qxy[w] = (Sxy[w] - cx * Sy[w] - cy * Sx[w] + (cx * cy) * M[w]) / ww2

            # w_big=65 block
            Ring1_65 = M[65] - M[33]
            DxR1_65 = Dx[65] - Dx[33]
            DyR1_65 = Dy[65] - Dy[33]
            QxxR1_65 = Qxx[65] - Qxx[33]
            QyyR1_65 = Qyy[65] - Qyy[33]
            QxyR1_65 = Qxy[65] - Qxy[33]
            F65 = np.column_stack(
                [
                    M[33],
                    M[65],
                    Ring1_65,
                    Dx[33],
                    Dy[33],
                    DxR1_65,
                    DyR1_65,
                    Qxx[33],
                    Qyy[33],
                    Qxy[33],
                    QxxR1_65,
                    QyyR1_65,
                    QxyR1_65,
                ]
            ).astype(np.float64, copy=False)
            assert_finite("F65", F65)
            Ffull_by_wbig_field[65].append(F65)

            # w_big=129 block
            Ring1_129 = M[65] - M[33]
            Ring2_129 = M[129] - M[65]
            DxR1_129 = Dx[65] - Dx[33]
            DyR1_129 = Dy[65] - Dy[33]
            DxR2_129 = Dx[129] - Dx[65]
            DyR2_129 = Dy[129] - Dy[65]
            QxxR1_129 = Qxx[65] - Qxx[33]
            QyyR1_129 = Qyy[65] - Qyy[33]
            QxyR1_129 = Qxy[65] - Qxy[33]
            QxxR2_129 = Qxx[129] - Qxx[65]
            QyyR2_129 = Qyy[129] - Qyy[65]
            QxyR2_129 = Qxy[129] - Qxy[65]
            F129 = np.column_stack(
                [
                    M[33],
                    M[65],
                    M[129],
                    Ring1_129,
                    Ring2_129,
                    Dx[33],
                    Dy[33],
                    DxR1_129,
                    DyR1_129,
                    DxR2_129,
                    DyR2_129,
                    Qxx[33],
                    Qyy[33],
                    Qxy[33],
                    QxxR1_129,
                    QyyR1_129,
                    QxyR1_129,
                    QxxR2_129,
                    QyyR2_129,
                    QxyR2_129,
                ]
            ).astype(np.float64, copy=False)
            assert_finite("F129", F129)
            Ffull_by_wbig_field[129].append(F129)

        # Save ring masks (these are square annuli, not radial/circular).
        def plot_ring_masks(wb: int) -> Path:
            r = wb // 2

            def square(w: int) -> np.ndarray:
                m = np.zeros((wb, wb), dtype=np.float64)
                rr = w // 2
                m[r - rr : r + rr + 1, r - rr : r + rr + 1] = 1.0
                return m

            m33 = square(33)
            m65 = square(65)
            if wb == 65:
                ring1 = m65 - m33
                fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                items = [("M33", m33), ("M65", m65), ("Ring1 (M65-M33)", ring1)]
            else:
                mb = square(wb)
                ring1 = m65 - m33
                ring2 = mb - m65
                fig, axes = plt.subplots(1, 5, figsize=(15, 3))
                items = [("M33", m33), ("M65", m65), (f"M{wb}", mb), ("Ring1 (M65-M33)", ring1), (f"Ring2 (M{wb}-M65)", ring2)]

            for ax, (title, arr) in zip(axes, items):
                vmin = float(arr.min())
                vmax = float(arr.max())
                if vmin < 0:
                    v = max(abs(vmin), abs(vmax))
                    im = ax.imshow(arr, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
                else:
                    im = ax.imshow(arr, cmap="viridis", vmin=0.0, vmax=max(1.0, vmax), interpolation="nearest")
                ax.set_title(title)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle(f"Ring masks (square windows), w_big={wb}")
            out = paths.run_dir / f"ring_masks_wbig{wb}.png"
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)
            return out

        ring_png_65 = plot_ring_masks(65)
        ring_png_129 = plot_ring_masks(129)

        # Storage for sweep results across folds.
        keys = [(wb, v, a) for wb in w_bigs for v in variants for a in ridge_alpha_grid]
        dp_by_key: dict[tuple[int, str, float], list[float]] = {k: [] for k in keys}
        dr_by_key: dict[tuple[int, str, float], list[float]] = {k: [] for k in keys}
        pvals_by_key: dict[tuple[int, str, float], list[float]] = {k: [] for k in keys}

        # Capture diagnostic fold (train) info for conditioning checks.
        diag_info: dict[tuple[int, str], Any] = {}

        for test_field in range(n_fields):
            train_fields = [i for i in range(n_fields) if i != test_field]
            reg, y_mu, y_sd = fit_pixel_model(train_fields, rhoz_by_field, centers_by_field, w_local, y_gx_by_field)

            # residual targets + B blocks
            r_tr_list: list[np.ndarray] = []
            Btr_list: list[np.ndarray] = []
            for fid in train_fields:
                y = y_gx_by_field[fid]
                ypred = predict_pixel(reg, y_mu, y_sd, rhoz_by_field[fid], centers_by_field[fid], w_local)
                r_tr_list.append((y - ypred).astype(np.float64, copy=False))
                Btr_list.append(B_by_field[fid])
            r_tr = np.concatenate(r_tr_list, axis=0)
            Btr = np.concatenate(Btr_list, axis=0)

            y_te = y_gx_by_field[test_field]
            ypred_te = predict_pixel(reg, y_mu, y_sd, rhoz_by_field[test_field], centers_by_field[test_field], w_local)
            r_te = (y_te - ypred_te).astype(np.float64, copy=False)
            Bte = B_by_field[test_field]

            # Standardize B on train
            B_mu = Btr.mean(axis=0)
            B_sd = np.where(Btr.std(axis=0) > 0, Btr.std(axis=0), 1.0)
            Btr_s = (Btr - B_mu) / B_sd
            Bte_s = (Bte - B_mu) / B_sd

            # Center residual target
            r_mu = float(np.mean(r_tr))
            rc = (r_tr - r_mu).reshape(-1, 1)

            A = Btr_s.T @ Btr_s
            uB = Btr_s.T @ rc

            # Precompute B baseline metrics for each ridge_alpha.
            pB_by_alpha: dict[float, float] = {}
            rrB_by_alpha: dict[float, float] = {}
            for a in ridge_alpha_grid:
                wB = solve_ridge(A, uB, a).reshape(-1)
                rB_te = Bte_s @ wB + r_mu
                pB_by_alpha[a] = safe_corr_1d(r_te, rB_te)
                rrB_by_alpha[a] = relrmse_1d(r_te, rB_te)

            # Shared placebo helper
            Z = np.concatenate([Btr_s, rc], axis=1)  # (n_train, dB+1)
            ZT = Z.T
            n_train = Btr_s.shape[0]
            rt = (r_te - float(r_te.mean())).astype(np.float64, copy=False)
            norm_rt = float(np.linalg.norm(rt)) + 1e-12

            # Process each w_big
            for wb in w_bigs:
                Ftr_raw_full = np.concatenate([Ffull_by_wbig_field[wb][fid] for fid in train_fields], axis=0)
                Fte_raw_full = Ffull_by_wbig_field[wb][test_field]
                names_full = names_full_by_wbig[wb]

                F_mu = Ftr_raw_full.mean(axis=0)
                F_sd = np.where(Ftr_raw_full.std(axis=0) > 0, Ftr_raw_full.std(axis=0), 1.0)
                Ftr_s_full = (Ftr_raw_full - F_mu) / F_sd
                Fte_s_full = (Fte_raw_full - F_mu) / F_sd
                assert_finite("Ftr_s_full", Ftr_s_full)
                assert_finite("Fte_s_full", Fte_s_full)

                C_full = Ftr_s_full.T @ Ftr_s_full

                # Precompute test correlation stats for full [B, F_full] and slice per variant.
                Xte_full = np.concatenate([Bte_s, Fte_s_full], axis=1)
                mX = Xte_full.mean(axis=0)
                Xc = Xte_full - mX
                t_full = (Xc.T @ rt.reshape(-1, 1)).reshape(-1)
                S_full = Xc.T @ Xc
                dB = Bte_s.shape[1]

                # Fit real C models per alpha and store their pC (needed for placebo comparison).
                pC_real: dict[tuple[str, float], float] = {}
                for vname in variants:
                    k = k_by_wbig_variant[(wb, vname)]
                    Ftr_s = Ftr_s_full[:, :k]
                    Fte_s = Fte_s_full[:, :k]
                    XtrC = np.concatenate([Btr_s, Ftr_s], axis=1)
                    XteC = np.concatenate([Bte_s, Fte_s], axis=1)
                    XtX = XtrC.T @ XtrC
                    Xty = XtrC.T @ rc
                    for a in ridge_alpha_grid:
                        wC = solve_ridge(XtX, Xty, a).reshape(-1)
                        rC_te = XteC @ wC + r_mu
                        pC = safe_corr_1d(r_te, rC_te)
                        rrC = relrmse_1d(r_te, rC_te)

                        pB = pB_by_alpha[a]
                        rrB = rrB_by_alpha[a]
                        dp = float(pC - pB)
                        dr = float(rrC - rrB)

                        key = (wb, vname, float(a))
                        dp_by_key[key].append(dp)
                        dr_by_key[key].append(dr)
                        pC_real[(vname, float(a))] = float(pC)

                # Store conditioning diagnostics on diag_fold (train fold only).
                if test_field == diag_fold:
                    diag_info[(wb, "Ftr_s_full")] = Ftr_s_full
                    diag_info[(wb, "names_full")] = names_full
                    diag_info[(wb, "r_tr")] = r_tr

                # Placebo permutations: count p_perm >= p_real per alpha, per variant.
                counts_ge: dict[tuple[str, float], int] = {(v, float(a)): 0 for v in variants for a in ridge_alpha_grid}
                for _ in range(n_perms):
                    pidx = rng_null.permutation(n_train)
                    Fp_full = Ftr_s_full[pidx]
                    M_full = ZT @ Fp_full
                    D_full = M_full[:-1, :]
                    v_full = M_full[-1, :].reshape(-1, 1)

                    for vname in variants:
                        k = k_by_wbig_variant[(wb, vname)]
                        D = D_full[:, :k]
                        v = v_full[:k, :]
                        Cmat = C_full[:k, :k]

                        XtX_p = np.block([[A, D], [D.T, Cmat]])
                        Xty_p = np.vstack([uB, v])

                        idx_cols = np.r_[np.arange(dB), dB + np.arange(k)]
                        t_sub = t_full[idx_cols]
                        S_sub = S_full[np.ix_(idx_cols, idx_cols)]

                        for a in ridge_alpha_grid:
                            wP = solve_ridge(XtX_p, Xty_p, a).reshape(-1)
                            denom = norm_rt * (float(np.sqrt(max(0.0, wP @ (S_sub @ wP)))) + 1e-12)
                            pP = float((t_sub @ wP) / denom)
                            if pP >= float(pC_real[(vname, float(a))]):
                                counts_ge[(vname, float(a))] += 1

                for vname in variants:
                    for a in ridge_alpha_grid:
                        cge = counts_ge[(vname, float(a))]
                        p_emp = float((cge + 1.0) / (float(n_perms) + 1.0))
                        pvals_by_key[(wb, vname, float(a))].append(p_emp)

        # Aggregate results
        rows_summary: list[dict[str, Any]] = []
        for wb in w_bigs:
            for vname in variants:
                for a in ridge_alpha_grid:
                    key = (wb, vname, float(a))
                    dp = np.asarray(dp_by_key[key], dtype=np.float64)
                    dr = np.asarray(dr_by_key[key], dtype=np.float64)
                    n_pos = int(np.sum(dp > 0.0))
                    fp = fisher_p(pvals_by_key[key])
                    verdict = (float(dp.mean()) > 0.0) and (float(dr.mean()) < 0.0) and (fp < 0.05) and (n_pos >= 7)
                    rows_summary.append(
                        {
                            "w_big": int(wb),
                            "variant": str(vname),
                            "ridge_alpha": float(a),
                            "deltaP_mean": float(dp.mean()),
                            "deltaP_std": float(dp.std(ddof=1)) if len(dp) > 1 else 0.0,
                            "deltaR_mean": float(dr.mean()),
                            "deltaR_std": float(dr.std(ddof=1)) if len(dr) > 1 else 0.0,
                            "fisher_p": float(fp),
                            "n_pos": int(n_pos),
                            "verdict": bool(verdict),
                        }
                    )

        # Best alpha per (w_big, variant) under PASS criteria.
        best_rows: list[dict[str, Any]] = []
        for wb in w_bigs:
            for vname in variants:
                cand = [r for r in rows_summary if int(r["w_big"]) == int(wb) and str(r["variant"]) == vname and bool(r["verdict"])]
                if not cand:
                    best_rows.append({"w_big": int(wb), "variant": vname, "best_alpha": float("nan"), "best_deltaP_mean": float("nan"), "best_fisher_p": float("nan"), "verdict": False})
                    continue
                cand.sort(key=lambda r: (-float(r["deltaP_mean"]), float(r["ridge_alpha"])))
                b = cand[0]
                best_rows.append({"w_big": int(wb), "variant": vname, "best_alpha": float(b["ridge_alpha"]), "best_deltaP_mean": float(b["deltaP_mean"]), "best_fisher_p": float(b["fisher_p"]), "verdict": True})

        # Conditioning/collinearity check on representative fold (train).
        cond_lines: list[str] = []
        if (65, "Ftr_s_full") in diag_info and (129, "Ftr_s_full") in diag_info:
            rtr = np.asarray(diag_info[(65, "r_tr")], dtype=np.float64).reshape(-1)
            for wb in w_bigs:
                Ffull = np.asarray(diag_info[(wb, "Ftr_s_full")], dtype=np.float64)
                names_full = list(diag_info[(wb, "names_full")])
                for vname in variants:
                    k = k_by_wbig_variant[(wb, vname)]
                    F = Ffull[:, :k]
                    names = names_full[:k]
                    C = np.corrcoef(F, rowvar=False)
                    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
                    Cabs = np.abs(C)
                    np.fill_diagonal(Cabs, 0.0)
                    ij = np.unravel_index(int(np.argmax(Cabs)), Cabs.shape)
                    max_abs = float(Cabs[ij]) if Cabs.size else float("nan")

                    corr_fr = [safe_corr_1d(F[:, j], rtr) for j in range(F.shape[1])]
                    top = np.argsort(-np.abs(np.asarray(corr_fr)))[: min(10, len(corr_fr))]
                    top_txt = ", ".join([f"{names[j]}:{corr_fr[j]:+.3f}" for j in top])
                    cond_lines.append(f"- fold={diag_fold} train | w_big={wb} {vname}: max|corr(feat,feat)|={max_abs:.3f} (pair={names[ij[0]]},{names[ij[1]]}); top corr(feat,r) = {top_txt}")

        # Markdown summary
        md_rows: list[dict[str, str]] = []
        for r0 in sorted(rows_summary, key=lambda r: (int(r["w_big"]), variant_order[str(r["variant"])], float(r["ridge_alpha"]))):
            md_rows.append(
                {
                    "w_big": str(int(r0["w_big"])),
                    "variant": str(r0["variant"]),
                    "ridge_alpha": fmt(float(r0["ridge_alpha"])),
                    "ΔPearson mean±std": f"{float(r0['deltaP_mean']):.4f} ± {float(r0['deltaP_std']):.4f}",
                    "ΔrelRMSE mean±std": f"{float(r0['deltaR_mean']):.4f} ± {float(r0['deltaR_std']):.4f}",
                    "Fisher p": fmt(float(r0["fisher_p"])),
                    "#folds ΔP>0": f"{int(r0['n_pos'])}/{n_fields}",
                }
            )
        md_best_rows: list[dict[str, str]] = []
        for b in best_rows:
            md_best_rows.append(
                {
                    "w_big": str(int(b["w_big"])),
                    "variant": str(b["variant"]),
                    "best_alpha": fmt(float(b["best_alpha"])),
                    "ΔPearson_mean": fmt(float(b["best_deltaP_mean"])),
                    "Fisher p": fmt(float(b["best_fisher_p"])),
                    "verdict": "PASS" if bool(b["verdict"]) else "FAIL",
                }
            )

        summary_md = (
            "# E28 — Diagnose nonlocal moments anomaly (w_local=65)\n\n"
            f"- run: `{paths.run_dir}`\n"
            f"- grid_size={grid_size}, alpha={alpha}, n_fields={n_fields}, patches_per_field={patches_per_field}\n"
            f"- w_local={w_local}, w_bigs={w_bigs}\n"
            f"- ridge_alpha_grid={ridge_alpha_grid}\n"
            f"- perms={n_perms} per fold, placebo_seed={placebo_seed}\n\n"
            "## Sweep (LOFO)\n\n"
            + md_table(md_rows, ["w_big", "variant", "ridge_alpha", "ΔPearson mean±std", "ΔrelRMSE mean±std", "Fisher p", "#folds ΔP>0"])
            + "\n\n## Best alpha (PASS-only)\n\n"
            + md_table(md_best_rows, ["w_big", "variant", "best_alpha", "ΔPearson_mean", "Fisher p", "verdict"])
            + "\n\n## Conditioning / Collinearity (fold train diagnostics)\n\n"
            + ("\n".join(cond_lines) if cond_lines else "- (diagnostics unavailable)\n")
            + "\n\n## Ring masks\n\n"
            + f"- `ring_masks_wbig65.png`: `{ring_png_65}`\n"
            + f"- `ring_masks_wbig129.png`: `{ring_png_129}`\n"
            + "\n"
            + "Note: these are *square* window-difference masks (square annuli), not radial/circular annuli.\n"
        )
        (paths.run_dir / "summary_e28_nonlocal_anomaly_diag.md").write_text(summary_md, encoding="utf-8")

        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "grid_size": grid_size,
                "alpha": alpha,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "w_local": w_local,
                "w_bigs": w_bigs,
                "ridge_alpha_grid": ridge_alpha_grid,
                "n_perms": n_perms,
                "rows_summary": rows_summary,
                "best_rows": best_rows,
                "diag_fold": diag_fold,
                "ring_masks": {"wbig65": str(ring_png_65), "wbig129": str(ring_png_129)},
            },
        )
        return paths

    if experiment == "e29":
        # Annulus multipoles (far-field-only) to test nonlocal residual structure (LOFO).
        #
        # We build an "outer annulus" feature block on a big window, excluding the inner local square core:
        # - core: |dx|<=r_local AND |dy|<=r_local
        # - outer: patch_big minus core
        # - radial rings: bin r=sqrt(dx^2+dy^2) on outer region
        # - multipoles on outer: M0, dipole (Dx,Dy), quadrupole (Qxx,Qxy), plus normalized versions.
        from numpy.lib.stride_tricks import sliding_window_view
        from scipy import ndimage, stats
        from sklearn.linear_model import SGDRegressor

        import matplotlib.pyplot as plt

        from .data_synth import solve_poisson_periodic_fft_2d
        from .features import nonlocal_annulus_moments

        grid_size = int(cfg.get("grid_size", 256))
        alpha = float(cfg.get("alpha", 2.0))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 10_000))

        settings_cfg = cfg.get("settings")
        if not isinstance(settings_cfg, list) or not settings_cfg:
            raise ValueError("E29 requires 'settings' as a non-empty list of {w_local,w_big} dicts")
        settings: list[tuple[int, int]] = []
        for item in settings_cfg:
            if not isinstance(item, dict):
                raise ValueError("settings entries must be dicts")
            wl = _require_odd("w_local", int(item.get("w_local", 0)))
            wb = _require_odd("w_big", int(item.get("w_big", 0)))
            settings.append((wl, wb))

        n_radial_bins = int(cfg.get("n_radial_bins", 6))
        eps_norm = float(cfg.get("eps_norm", 1e-12))
        validate_fft_n = int(cfg.get("validate_fft_n", 2))
        validate_fft_rtol = float(cfg.get("validate_fft_rtol", 1e-6))
        validate_fft_atol = float(cfg.get("validate_fft_atol", 1e-6))
        if n_radial_bins <= 0:
            raise ValueError("n_radial_bins must be > 0")
        if eps_norm <= 0:
            raise ValueError("eps_norm must be > 0")
        if validate_fft_n < 0:
            raise ValueError("validate_fft_n must be >= 0")
        if validate_fft_rtol < 0 or validate_fft_atol < 0:
            raise ValueError("validate_fft_rtol/atol must be >= 0")

        variants = [
            "C_annulus_rings",
            "C_annulus_rings+dipole",
            "C_annulus_rings+dipole+quad",
        ]

        n_perms = int(cfg.get("n_perms", 200))
        ridge_alpha = float(cfg.get("ridge_alpha", 1.0))

        pixel_epochs = int(cfg.get("pixel_epochs", 2))
        pixel_batch_size = int(cfg.get("pixel_batch_size", 1024))
        pixel_alpha = float(cfg.get("pixel_alpha", 1e-4))
        pixel_eta0 = float(cfg.get("pixel_eta0", 0.01))
        pixel_power_t = float(cfg.get("pixel_power_t", 0.25))

        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if n_fields < 2:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        if any(wl > wb for wl, wb in settings):
            raise ValueError("Each setting must satisfy w_local <= w_big")
        if any(wb > grid_size for _, wb in settings):
            raise ValueError("All w_big must be <= grid_size")
        if n_perms <= 0:
            raise ValueError("n_perms must be > 0")
        if ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be > 0")

        rng_pixel = np.random.default_rng(int(placebo_seed) + 404_111)
        rng_null = np.random.default_rng(int(placebo_seed) + 404_222)

        def fisher_p(ps: list[float]) -> float:
            if not ps:
                return float("nan")
            eps = 1.0 / float(n_perms + 1)
            ps2 = [min(1.0, max(float(p), eps)) for p in ps]
            stat = -2.0 * float(np.sum(np.log(ps2)))
            return float(stats.chi2.sf(stat, 2 * len(ps2)))

        def fmt(x: float) -> str:
            if not np.isfinite(x):
                return "nan"
            ax = abs(float(x))
            if ax != 0.0 and (ax < 1e-3 or ax >= 1e3):
                return f"{x:.3e}"
            return f"{x:.4f}"

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r0 in rows:
                out.append("| " + " | ".join(r0.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        def safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            am = a - float(a.mean())
            bm = b - float(b.mean())
            denom = float(np.linalg.norm(am) * np.linalg.norm(bm)) + 1e-12
            return float((am @ bm) / denom)

        def relrmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
            e = y_pred - y_true
            rmse = float(np.sqrt(np.mean(e * e)))
            sd = float(np.std(y_true))
            return rmse / (sd + 1e-12)

        def solve_ridge(XtX: np.ndarray, Xty: np.ndarray) -> np.ndarray:
            try:
                return np.linalg.solve(XtX + ridge_alpha * np.eye(XtX.shape[0]), Xty)
            except np.linalg.LinAlgError:
                w, *_ = np.linalg.lstsq(XtX + ridge_alpha * np.eye(XtX.shape[0]), Xty, rcond=None)
                return w

        # Pixel baseline: ridge-like SGD (scalar), with fold-specific y scaling.
        def fit_pixel_model(
            train_fields: list[int],
            fields_z: list[np.ndarray],
            centers_by_field: list[np.ndarray],
            w: int,
            y_by_field: list[np.ndarray],
        ) -> tuple[SGDRegressor, float, float]:
            y_all = np.concatenate([y_by_field[fid] for fid in train_fields], axis=0).astype(np.float64, copy=False)
            y_mu = float(np.mean(y_all))
            y_sd = float(np.std(y_all))
            if not np.isfinite(y_sd) or y_sd <= 0:
                y_sd = 1.0

            eta0_eff = float(pixel_eta0) / float(w)
            reg = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(pixel_alpha),
                learning_rate="invscaling",
                eta0=float(eta0_eff),
                power_t=float(pixel_power_t),
                max_iter=1,
                tol=None,
                fit_intercept=True,
                average=True,
                random_state=int(placebo_seed) + 0,
            )
            rloc = w // 2
            for _epoch in range(pixel_epochs):
                for fid in train_fields:
                    field = fields_z[fid]
                    centers = centers_by_field[fid]
                    y = y_by_field[fid]
                    win = sliding_window_view(field, (w, w))
                    perm = rng_pixel.permutation(centers.shape[0])
                    for start in range(0, centers.shape[0], pixel_batch_size):
                        idx = perm[start : start + pixel_batch_size]
                        cx = centers[idx, 0].astype(np.int64)
                        cy = centers[idx, 1].astype(np.int64)
                        Xb = win[cx - rloc, cy - rloc].reshape(len(idx), -1).astype(np.float32, copy=False)
                        reg.partial_fit(Xb, (y[idx] - y_mu) / y_sd)
            return reg, y_mu, y_sd

        def predict_pixel(reg: SGDRegressor, y_mu: float, y_sd: float, field_z: np.ndarray, centers: np.ndarray, w: int) -> np.ndarray:
            rloc = w // 2
            win = sliding_window_view(field_z, (w, w))
            out = np.empty((centers.shape[0],), dtype=np.float64)
            for start in range(0, centers.shape[0], pixel_batch_size):
                idx = slice(start, min(centers.shape[0], start + pixel_batch_size))
                cx = centers[idx, 0].astype(np.int64)
                cy = centers[idx, 1].astype(np.int64)
                Xb = win[cx - rloc, cy - rloc].reshape(len(cx), -1).astype(np.float32, copy=False)
                out[idx] = reg.predict(Xb) * y_sd + y_mu
            return out

        # Data generation: rho01 in [0,1], full gx.
        rho01_by_field: list[np.ndarray] = []
        rhoz_by_field: list[np.ndarray] = []
        gx_full_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
            sol = solve_poisson_periodic_fft_2d(rho01)
            rho01_by_field.append(rho01.astype(np.float64, copy=False))
            rhoz_by_field.append(_zscore_field(rho01).astype(np.float64, copy=False))
            gx_full_by_field.append(sol.gx.astype(np.float64, copy=False))

        # Centers once per field, safe for max w_big.
        r_max = max(wb for _, wb in settings) // 2
        centers_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_cent = np.random.default_rng(seed + 444_444 + 1_000 * field_id)
            cx = rng_cent.integers(r_max, grid_size - r_max, size=patches_per_field, dtype=np.int64)
            cy = rng_cent.integers(r_max, grid_size - r_max, size=patches_per_field, dtype=np.int64)
            centers_by_field.append(np.column_stack([cx, cy]).astype(np.int64, copy=False))

        y_gx_by_field: list[np.ndarray] = []
        for fid in range(n_fields):
            centers = centers_by_field[fid]
            cx = centers[:, 0].astype(np.int64)
            cy = centers[:, 1].astype(np.int64)
            y_gx_by_field.append(gx_full_by_field[fid][cx, cy].astype(np.float64, copy=False))

        # Local B features per w_local.
        w_local_set = sorted(set(wl for wl, _ in settings))
        B_by_wlocal_field: dict[int, list[np.ndarray]] = {wl: [] for wl in w_local_set}
        for wl in w_local_set:
            rloc = wl // 2
            for field_id in range(n_fields):
                field = rho01_by_field[field_id]
                centers = centers_by_field[field_id]
                cx = centers[:, 0]
                cy = centers[:, 1]
                pref1 = _prefix_sum_2d(field)
                pref2 = _prefix_sum_2d(field * field)
                x0 = (cx - rloc).astype(np.int64)
                x1 = (cx + rloc + 1).astype(np.int64)
                y0 = (cy - rloc).astype(np.int64)
                y1 = (cy + rloc + 1).astype(np.int64)
                mass = _box_sum_2d(pref1, x0, x1, y0, y1)
                nvox = float(wl * wl)
                mean = mass / nvox
                sumsq = _box_sum_2d(pref2, x0, x1, y0, y1)
                var = np.maximum(0.0, sumsq / nvox - mean * mean)
                mass2 = mass * mass
                max_grid = ndimage.maximum_filter(field, size=wl, mode="constant", cval=-np.inf)
                mx = max_grid[cx, cy]
                gx_f, gy_f = np.gradient(field)
                egrid = gx_f * gx_f + gy_f * gy_f
                eavg = ndimage.uniform_filter(egrid, size=wl, mode="constant", cval=0.0)
                ge = eavg[cx, cy]
                B = np.column_stack([mass, mass2, var, mx, ge]).astype(np.float64, copy=False)
                B_by_wlocal_field[wl].append(B)

        # Full annulus feature block order:
        # [M0, ring_sum..., ring_frac..., Dx,Dy,Dx_n,Dy_n, Qxx,Qxy,Qxx_n,Qxy_n]
        rings_end = 1 + 2 * n_radial_bins
        dip_end = rings_end + 4
        quad_end = dip_end + 4

        k_by_variant: dict[str, int] = {
            "C_annulus_rings": rings_end,
            "C_annulus_rings+dipole": dip_end,
            "C_annulus_rings+dipole+quad": quad_end,
        }
        n_full = quad_end

        def kernel_fft_centered(kernel: np.ndarray) -> np.ndarray:
            """
            Return FFT of a *centered* spatial kernel for circular correlation.

            We want, for each center (i,j), the correlation:
              S[i,j] = sum_{dx,dy} field[i+dx, j+dy] * kernel[dx,dy]

            Using FFT multiplication computes circular convolution, which includes a kernel flip.
            To get correlation we flip the kernel first, then embed it into a full grid with its
            center at (N//2,N//2) before applying ifftshift (so negative offsets wrap to the end).
            """
            kernel = np.asarray(kernel, dtype=np.float64)
            if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
                raise ValueError(f"kernel must be square 2D, got {kernel.shape}")
            wb = int(kernel.shape[0])
            if (wb % 2) == 0:
                raise ValueError(f"kernel size must be odd, got {wb}")
            if wb > grid_size:
                raise ValueError(f"kernel size {wb} exceeds grid_size {grid_size}")

            r = wb // 2
            kr = np.flip(kernel, axis=(0, 1))
            full = np.zeros((grid_size, grid_size), dtype=np.float64)
            cx = grid_size // 2
            cy = grid_size // 2
            full[cx - r : cx + r + 1, cy - r : cy + r + 1] = kr
            full0 = np.fft.ifftshift(full)
            return np.fft.fftn(full0)

        def build_kernels(wb: int, wl: int) -> dict[str, np.ndarray]:
            r_big = wb // 2
            r_local = wl // 2
            coords = np.arange(wb, dtype=np.float64) - float(r_big)
            dx = coords[:, None]
            dy = coords[None, :]
            core = (np.abs(dx) <= float(r_local)) & (np.abs(dy) <= float(r_local))
            outer = ~core
            outer_f = outer.astype(np.float64)
            r = np.sqrt(dx * dx + dy * dy, dtype=np.float64)

            kernels: dict[str, np.ndarray] = {"M0": outer_f}
            if r_big > r_local and outer.any():
                edges = np.linspace(float(r_local), float(r_big) + 1e-9, n_radial_bins + 1, dtype=np.float64)
                for k in range(n_radial_bins):
                    lo = edges[k]
                    hi = edges[k + 1]
                    if k == n_radial_bins - 1:
                        m = outer & (r >= lo) & (r <= hi)
                    else:
                        m = outer & (r >= lo) & (r < hi)
                    kernels[f"ring{k}"] = m.astype(np.float64)
            else:
                for k in range(n_radial_bins):
                    kernels[f"ring{k}"] = np.zeros((wb, wb), dtype=np.float64)

            kernels["Dx"] = (dx * outer_f).astype(np.float64, copy=False)
            kernels["Dy"] = (dy * outer_f).astype(np.float64, copy=False)
            kernels["Qxx"] = ((dx * dx - dy * dy) * outer_f).astype(np.float64, copy=False)
            kernels["Qxy"] = ((2.0 * dx * dy) * outer_f).astype(np.float64, copy=False)
            return kernels

        def plot_annulus_masks(wb: int, wl: int) -> Path:
            r_big = wb // 2
            r_local = wl // 2
            coords = np.arange(wb, dtype=np.float64) - float(r_big)
            dx = coords[:, None]
            dy = coords[None, :]
            core = (np.abs(dx) <= float(r_local)) & (np.abs(dy) <= float(r_local))
            outer = (~core).astype(np.float64)
            r = np.sqrt(dx * dx + dy * dy, dtype=np.float64)
            if r_big <= r_local:
                bins: list[np.ndarray] = []
            else:
                edges = np.linspace(float(r_local), float(r_big) + 1e-9, n_radial_bins + 1, dtype=np.float64)
                bins = []
                for k in range(n_radial_bins):
                    lo = edges[k]
                    hi = edges[k + 1]
                    if k == n_radial_bins - 1:
                        m = (r >= lo) & (r <= hi) & (~core)
                    else:
                        m = (r >= lo) & (r < hi) & (~core)
                    bins.append(m.astype(np.float64))

            ncols = 1 + len(bins)
            fig, axes = plt.subplots(1, ncols, figsize=(3.0 * ncols, 3.0))
            if ncols == 1:
                axes = [axes]
            axes[0].imshow(outer, cmap="gray", interpolation="nearest")
            axes[0].set_title(f"outer (w_big={wb}, w_local={wl})")
            axes[0].set_xticks([])
            axes[0].set_yticks([])
            for k, m in enumerate(bins):
                axes[k + 1].imshow(m, cmap="gray", interpolation="nearest")
                axes[k + 1].set_title(f"ring {k}")
                axes[k + 1].set_xticks([])
                axes[k + 1].set_yticks([])
            fig.suptitle("Annulus + radial bins (true rings on outer region)")
            out = paths.run_dir / f"annulus_masks_wlocal{wl}_wbig{wb}.png"
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)
            return out

        # Precompute annulus blocks per setting and field.
        mask_pngs: dict[str, str] = {}
        Ffull_by_setting_field: dict[tuple[int, int], list[np.ndarray]] = {}
        for wl, wb in settings:
            if (wl, wb) in Ffull_by_setting_field:
                continue
            mask_pngs[f"wlocal{wl}_wbig{wb}"] = str(plot_annulus_masks(wb, wl))

            if wb <= wl:
                zeros = np.zeros((patches_per_field, n_full), dtype=np.float64)
                Ffull_by_setting_field[(wl, wb)] = [zeros.copy() for _ in range(n_fields)]
                continue

            kernels = build_kernels(wb, wl)
            r_big = wb // 2
            kernel_ffts = {name: kernel_fft_centered(k) for name, k in kernels.items()}
            per_field: list[np.ndarray] = []
            for field_id in range(n_fields):
                field = rho01_by_field[field_id]
                centers = centers_by_field[field_id]
                cx = centers[:, 0].astype(np.int64)
                cy = centers[:, 1].astype(np.int64)
                F = np.zeros((patches_per_field, n_full), dtype=np.float64)

                field_fft = np.fft.fftn(field)

                m0_grid = np.fft.ifftn(field_fft * kernel_ffts["M0"]).real
                M0 = m0_grid[cx, cy].astype(np.float64, copy=False)
                denom = M0 + eps_norm
                F[:, 0] = M0

                ring_sums = np.zeros((patches_per_field, n_radial_bins), dtype=np.float64)
                for k in range(n_radial_bins):
                    gk = np.fft.ifftn(field_fft * kernel_ffts[f"ring{k}"]).real
                    ring_sums[:, k] = gk[cx, cy].astype(np.float64, copy=False)
                ring_fracs = ring_sums / denom[:, None]
                F[:, 1 : 1 + n_radial_bins] = ring_sums
                F[:, 1 + n_radial_bins : 1 + 2 * n_radial_bins] = ring_fracs

                dx_grid = np.fft.ifftn(field_fft * kernel_ffts["Dx"]).real
                dy_grid = np.fft.ifftn(field_fft * kernel_ffts["Dy"]).real
                qxx_grid = np.fft.ifftn(field_fft * kernel_ffts["Qxx"]).real
                qxy_grid = np.fft.ifftn(field_fft * kernel_ffts["Qxy"]).real
                Dx = dx_grid[cx, cy].astype(np.float64, copy=False)
                Dy = dy_grid[cx, cy].astype(np.float64, copy=False)
                Qxx = qxx_grid[cx, cy].astype(np.float64, copy=False)
                Qxy = qxy_grid[cx, cy].astype(np.float64, copy=False)

                F[:, rings_end + 0] = Dx
                F[:, rings_end + 1] = Dy
                F[:, rings_end + 2] = Dx / denom
                F[:, rings_end + 3] = Dy / denom

                F[:, dip_end + 0] = Qxx
                F[:, dip_end + 1] = Qxy
                F[:, dip_end + 2] = Qxx / denom
                F[:, dip_end + 3] = Qxy / denom

                if field_id == 0 and validate_fft_n > 0:
                    n_check = min(int(validate_fft_n), patches_per_field)
                    feat_names = (
                        ["M0"]
                        + [f"ring_sum_{k}" for k in range(n_radial_bins)]
                        + [f"ring_frac_{k}" for k in range(n_radial_bins)]
                        + ["Dx", "Dy", "Dx_n", "Dy_n", "Qxx", "Qxy", "Qxx_n", "Qxy_n"]
                    )
                    for j in range(n_check):
                        cxi = int(cx[j])
                        cyi = int(cy[j])
                        patch_big = field[cxi - r_big : cxi + r_big + 1, cyi - r_big : cyi + r_big + 1]
                        d = nonlocal_annulus_moments(
                            patch_big,
                            w_local=wl,
                            n_radial_bins=n_radial_bins,
                            include_dipole=True,
                            include_quadrupole=True,
                            eps=eps_norm,
                        )
                        row_d = np.zeros((n_full,), dtype=np.float64)
                        row_d[0] = float(d["M0"])
                        for k in range(n_radial_bins):
                            row_d[1 + k] = float(d[f"ring_sum_{k}"])
                            row_d[1 + n_radial_bins + k] = float(d[f"ring_frac_{k}"])
                        row_d[rings_end + 0] = float(d["Dx"])
                        row_d[rings_end + 1] = float(d["Dy"])
                        row_d[rings_end + 2] = float(d["Dx_n"])
                        row_d[rings_end + 3] = float(d["Dy_n"])
                        row_d[dip_end + 0] = float(d["Qxx"])
                        row_d[dip_end + 1] = float(d["Qxy"])
                        row_d[dip_end + 2] = float(d["Qxx_n"])
                        row_d[dip_end + 3] = float(d["Qxy_n"])

                        for idx, name in enumerate(feat_names):
                            a = float(F[j, idx])
                            b = float(row_d[idx])
                            if not np.isclose(a, b, rtol=validate_fft_rtol, atol=validate_fft_atol):
                                raise RuntimeError(
                                    "E29 annulus FFT feature mismatch "
                                    f"(w_local={wl}, w_big={wb}, field_id={field_id}, sample={j}, {name}): "
                                    f"fft={a:.6g} direct={b:.6g} diff={abs(a-b):.6g}"
                                )

                assert_finite("annulus_F", F)
                per_field.append(F)
            Ffull_by_setting_field[(wl, wb)] = per_field

        rows_summary: list[dict[str, Any]] = []
        for wl, wb in settings:
            B_by_field = B_by_wlocal_field[wl]
            F_by_field_full = Ffull_by_setting_field[(wl, wb)]
            if wb <= wl:
                for variant in variants:
                    rows_summary.append(
                        {
                            "w_local": int(wl),
                            "w_big": int(wb),
                            "variant": str(variant),
                            "deltaP_mean": 0.0,
                            "deltaP_std": 0.0,
                            "deltaR_mean": 0.0,
                            "deltaR_std": 0.0,
                            "fisher_p": 1.0,
                            "n_pos": 0,
                            "verdict": False,
                        }
                    )
                continue
            for variant in variants:
                k = int(k_by_variant[variant])
                fold_rows: list[dict[str, Any]] = []
                pvals: list[float] = []

                for test_field in range(n_fields):
                    train_fields = [i for i in range(n_fields) if i != test_field]
                    reg, y_mu, y_sd = fit_pixel_model(train_fields, rhoz_by_field, centers_by_field, wl, y_gx_by_field)

                    r_tr_list: list[np.ndarray] = []
                    Btr_list: list[np.ndarray] = []
                    Ftr_list: list[np.ndarray] = []
                    for fid in train_fields:
                        y = y_gx_by_field[fid]
                        ypred = predict_pixel(reg, y_mu, y_sd, rhoz_by_field[fid], centers_by_field[fid], wl)
                        r_tr_list.append((y - ypred).astype(np.float64, copy=False))
                        Btr_list.append(B_by_field[fid])
                        Ftr_list.append(F_by_field_full[fid][:, :k])
                    r_tr = np.concatenate(r_tr_list, axis=0)
                    Btr = np.concatenate(Btr_list, axis=0)
                    Ftr_raw = np.concatenate(Ftr_list, axis=0)

                    y_te = y_gx_by_field[test_field]
                    ypred_te = predict_pixel(reg, y_mu, y_sd, rhoz_by_field[test_field], centers_by_field[test_field], wl)
                    r_te = (y_te - ypred_te).astype(np.float64, copy=False)
                    Bte = B_by_field[test_field]
                    Fte_raw = F_by_field_full[test_field][:, :k]

                    B_mu = Btr.mean(axis=0)
                    B_sd = np.where(Btr.std(axis=0) > 0, Btr.std(axis=0), 1.0)
                    Btr_s = (Btr - B_mu) / B_sd
                    Bte_s = (Bte - B_mu) / B_sd

                    F_mu = Ftr_raw.mean(axis=0)
                    F_sd = np.where(Ftr_raw.std(axis=0) > 0, Ftr_raw.std(axis=0), 1.0)
                    Ftr_s = (Ftr_raw - F_mu) / F_sd
                    Fte_s = (Fte_raw - F_mu) / F_sd

                    r_mu = float(np.mean(r_tr))
                    rc = (r_tr - r_mu).reshape(-1, 1)

                    A = Btr_s.T @ Btr_s
                    uB = Btr_s.T @ rc
                    wB = solve_ridge(A, uB).reshape(-1)
                    rB_te = Bte_s @ wB + r_mu
                    pB = safe_corr_1d(r_te, rB_te)
                    rrB = relrmse_1d(r_te, rB_te)

                    XtrC = np.concatenate([Btr_s, Ftr_s], axis=1)
                    XteC = np.concatenate([Bte_s, Fte_s], axis=1)
                    XtX = XtrC.T @ XtrC
                    Xty = XtrC.T @ rc
                    wC = solve_ridge(XtX, Xty).reshape(-1)
                    rC_te = XteC @ wC + r_mu
                    pC = safe_corr_1d(r_te, rC_te)
                    rrC = relrmse_1d(r_te, rC_te)

                    deltaP = float(pC - pB)
                    deltaR = float(rrC - rrB)

                    # Placebo permutations: shuffle annulus features (TRAIN rows only).
                    Cmat = Ftr_s.T @ Ftr_s
                    Z = np.concatenate([Btr_s, rc], axis=1)
                    null_dP: list[float] = []
                    for _ in range(n_perms):
                        pidx = rng_null.permutation(Ftr_s.shape[0])
                        Fp = Ftr_s[pidx]
                        M = Z.T @ Fp
                        D = M[:-1, :]
                        v = M[-1, :].reshape(-1, 1)
                        XtX_p = np.block([[A, D], [D.T, Cmat]])
                        Xty_p = np.vstack([uB, v])
                        wP = solve_ridge(XtX_p, Xty_p).reshape(-1)
                        rP_te = XteC @ wP + r_mu
                        pP = safe_corr_1d(r_te, rP_te)
                        null_dP.append(float(pP - pB))
                    null = np.asarray(null_dP, dtype=np.float64)
                    p_emp = float((np.sum(null >= deltaP) + 1.0) / (float(n_perms) + 1.0))
                    pvals.append(p_emp)

                    fold_rows.append(
                        {
                            "field_id": int(test_field),
                            "pearson_B": float(pB),
                            "relRMSE_B": float(rrB),
                            "pearson_C": float(pC),
                            "relRMSE_C": float(rrC),
                            "deltaP": float(deltaP),
                            "deltaR": float(deltaR),
                            "p_emp": float(p_emp),
                        }
                    )

                dp = np.asarray([r0["deltaP"] for r0 in fold_rows], dtype=np.float64)
                dr = np.asarray([r0["deltaR"] for r0 in fold_rows], dtype=np.float64)
                n_pos = int(np.sum(dp > 0.0))
                fp = fisher_p(pvals)
                verdict = (float(dp.mean()) > 0.0) and (float(dr.mean()) < 0.0) and (fp < 0.05) and (n_pos >= 7)

                rows_summary.append(
                    {
                        "w_local": int(wl),
                        "w_big": int(wb),
                        "variant": str(variant),
                        "deltaP_mean": float(dp.mean()),
                        "deltaP_std": float(dp.std(ddof=1)) if len(dp) > 1 else 0.0,
                        "deltaR_mean": float(dr.mean()),
                        "deltaR_std": float(dr.std(ddof=1)) if len(dr) > 1 else 0.0,
                        "fisher_p": float(fp),
                        "n_pos": int(n_pos),
                        "verdict": bool(verdict),
                    }
                )

                tag = variant.replace("+", "plus").replace(" ", "")
                header = list(fold_rows[0].keys()) if fold_rows else []
                write_csv(
                    paths.run_dir / f"lofo_by_field_wlocal{wl}_wbig{wb}_{tag}.csv",
                    header,
                    [[r0[h] for h in header] for r0 in fold_rows],
                )

        md_rows: list[dict[str, str]] = []
        for r0 in rows_summary:
            md_rows.append(
                {
                    "w_local": str(int(r0["w_local"])),
                    "w_big": str(int(r0["w_big"])),
                    "variant": str(r0["variant"]),
                    "ΔPearson mean±std": f"{float(r0['deltaP_mean']):.4f} ± {float(r0['deltaP_std']):.4f}",
                    "ΔrelRMSE mean±std": f"{float(r0['deltaR_mean']):.4f} ± {float(r0['deltaR_std']):.4f}",
                    "Fisher p": fmt(float(r0["fisher_p"])),
                    "#folds ΔP>0": f"{int(r0['n_pos'])}/{n_fields}",
                    "verdict": "PASS" if bool(r0["verdict"]) else "FAIL",
                }
            )

        mask_lines = "\n".join([f"- `{k}`: `{v}`" for k, v in mask_pngs.items()])
        summary_md = (
            "# E29 — Annulus multipoles (far-field-only) on full-g residual (LOFO)\n\n"
            f"- run: `{paths.run_dir}`\n"
            f"- grid_size={grid_size}, alpha={alpha}, n_fields={n_fields}, patches_per_field={patches_per_field}\n"
            f"- settings={settings}\n"
            f"- n_radial_bins={n_radial_bins}, eps_norm={eps_norm}\n"
            f"- pixel: SGD L2 (ridge-like) alpha={pixel_alpha}, epochs={pixel_epochs}, batch={pixel_batch_size}\n"
            f"- residual ridge_alpha={ridge_alpha}, perms={n_perms}\n\n"
            + md_table(
                md_rows,
                ["w_local", "w_big", "variant", "ΔPearson mean±std", "ΔrelRMSE mean±std", "Fisher p", "#folds ΔP>0", "verdict"],
            )
            + "\n\n## Mask PNGs\n\n"
            + mask_lines
            + "\n"
        )
        (paths.run_dir / "summary_e29_annulus_multipoles.md").write_text(summary_md, encoding="utf-8")

        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "grid_size": grid_size,
                "alpha": alpha,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "settings": [{"w_local": int(wl), "w_big": int(wb)} for wl, wb in settings],
                "n_radial_bins": n_radial_bins,
                "eps_norm": eps_norm,
                "pixel": {
                    "solver": "sgd_l2",
                    "alpha": pixel_alpha,
                    "epochs": pixel_epochs,
                    "batch_size": pixel_batch_size,
                    "eta0": pixel_eta0,
                    "power_t": pixel_power_t,
                },
                "residual": {"ridge_alpha": ridge_alpha, "n_perms": n_perms},
                "rows_summary": rows_summary,
                "mask_pngs": mask_pngs,
            },
        )
        return paths

    if experiment == "e30":
        # E30 — Show that annulus dipole predicts the low-k (far-field) component of g in LOFO.
        #
        # Targets: (gx_low, gy_low) at patch centers, from a band-split Poisson solve.
        # Features:
        # - B_low: local_B (mass, mass2, var, max, grad_energy) on rho01.
        # - Ck: B_low + annulus multipoles on rho01 (rings, dipole, quadrupole), computed on the OUTER region
        #       of a big window (w_big) excluding the inner local core (w_local).
        from numpy.lib.stride_tricks import sliding_window_view
        from scipy import ndimage, stats

        import matplotlib.pyplot as plt

        from .features import nonlocal_annulus_moments

        grid_size = int(cfg.get("grid_size", 256))
        alpha = float(cfg.get("alpha", 2.0))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 10_000))
        w_local = _require_odd("w_local", int(cfg.get("w_local", 65)))
        w_big = _require_odd("w_big", int(cfg.get("w_big", 129)))
        n_radial_bins = int(cfg.get("n_radial_bins", 6))
        eps_norm = float(cfg.get("eps_norm", 1e-12))
        k0_fracs = [float(x) for x in cfg.get("k0_fracs", [0.10, 0.15, 0.25])]
        ridge_alpha = float(cfg.get("ridge_alpha", 1.0))
        n_perms = int(cfg.get("n_perms", 200))
        validate_fft_n = int(cfg.get("validate_fft_n", 1))
        validate_fft_rtol = float(cfg.get("validate_fft_rtol", 1e-6))
        validate_fft_atol = float(cfg.get("validate_fft_atol", 1e-6))

        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if n_fields < 2:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        if w_local > w_big:
            raise ValueError("w_local must be <= w_big")
        if w_big > grid_size:
            raise ValueError("w_big must be <= grid_size")
        if n_radial_bins <= 0:
            raise ValueError("n_radial_bins must be > 0")
        if eps_norm <= 0:
            raise ValueError("eps_norm must be > 0")
        if not k0_fracs:
            raise ValueError("k0_fracs must be non-empty")
        if any((k <= 0.0) or (k >= 0.5) for k in k0_fracs):
            raise ValueError("k0_fracs entries must be in (0,0.5)")
        if ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be > 0")
        if n_perms <= 0:
            raise ValueError("n_perms must be > 0")
        if validate_fft_n < 0:
            raise ValueError("validate_fft_n must be >= 0")
        if validate_fft_rtol < 0 or validate_fft_atol < 0:
            raise ValueError("validate_fft_rtol/atol must be >= 0")

        rng_null = np.random.default_rng(int(placebo_seed) + 303_303)

        def fisher_p(ps: list[float]) -> float:
            if not ps:
                return float("nan")
            eps = 1.0 / float(n_perms + 1)
            ps2 = [min(1.0, max(float(p), eps)) for p in ps]
            stat = -2.0 * float(np.sum(np.log(ps2)))
            return float(stats.chi2.sf(stat, 2 * len(ps2)))

        def safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            am = a - float(a.mean())
            bm = b - float(b.mean())
            denom = float(np.linalg.norm(am) * np.linalg.norm(bm)) + 1e-12
            return float((am @ bm) / denom)

        def relrmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
            e = y_pred - y_true
            rmse = float(np.sqrt(np.mean(e * e)))
            sd = float(np.std(y_true))
            return rmse / (sd + 1e-12)

        def pearson_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            if y_true.shape != y_pred.shape or y_true.ndim != 2 or y_true.shape[1] != 2:
                raise ValueError(f"invalid 2D targets shapes: {y_true.shape} vs {y_pred.shape}")
            px = safe_corr_1d(y_true[:, 0], y_pred[:, 0])
            py = safe_corr_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (px + py)

        def relrmse_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            rx = relrmse_1d(y_true[:, 0], y_pred[:, 0])
            ry = relrmse_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (rx + ry)

        def solve_ridge(XtX: np.ndarray, Xty: np.ndarray) -> np.ndarray:
            d = int(XtX.shape[0])
            I = np.eye(d, dtype=np.float64)
            try:
                return np.linalg.solve(XtX + ridge_alpha * I, Xty)
            except np.linalg.LinAlgError:
                w, *_ = np.linalg.lstsq(XtX + ridge_alpha * I, Xty, rcond=None)
                return w

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r0 in rows:
                out.append("| " + " | ".join(r0.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        def fmt(x: float) -> str:
            if not np.isfinite(x):
                return "nan"
            if x < 1e-3:
                return f"{x:.3e}"
            return f"{x:.4f}"

        # Generate fields and centers (shared across k0_fracs).
        rho01_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
            rho01_by_field.append(rho01.astype(np.float64, copy=False))

        r_big = w_big // 2
        centers_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_cent = np.random.default_rng(seed + 444_444 + 1_000 * field_id)
            cx = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
            cy = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
            centers_by_field.append(np.column_stack([cx, cy]).astype(np.int64, copy=False))

        # Local_B features (w_local) on rho01.
        # NOTE: avoid shadowing module-level helpers used by other experiments.
        def _prefix_sum_2d_local(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64)
            return np.pad(x, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)

        def _box_sum_2d_local(pref: np.ndarray, x0: np.ndarray, x1: np.ndarray, y0: np.ndarray, y1: np.ndarray) -> np.ndarray:
            return pref[x1, y1] - pref[x0, y1] - pref[x1, y0] + pref[x0, y0]

        B_by_field: list[np.ndarray] = []
        rloc = w_local // 2
        for field_id in range(n_fields):
            field = rho01_by_field[field_id]
            centers = centers_by_field[field_id]
            cx = centers[:, 0]
            cy = centers[:, 1]
            pref1 = _prefix_sum_2d_local(field)
            pref2 = _prefix_sum_2d_local(field * field)
            x0 = (cx - rloc).astype(np.int64)
            x1 = (cx + rloc + 1).astype(np.int64)
            y0 = (cy - rloc).astype(np.int64)
            y1 = (cy + rloc + 1).astype(np.int64)
            mass = _box_sum_2d_local(pref1, x0, x1, y0, y1)
            nvox = float(w_local * w_local)
            mean = mass / nvox
            sumsq = _box_sum_2d_local(pref2, x0, x1, y0, y1)
            var = np.maximum(0.0, sumsq / nvox - mean * mean)
            mass2 = mass * mass
            max_grid = ndimage.maximum_filter(field, size=w_local, mode="constant", cval=-np.inf)
            mx = max_grid[cx, cy]
            gx_f, gy_f = np.gradient(field)
            egrid = gx_f * gx_f + gy_f * gy_f
            eavg = ndimage.uniform_filter(egrid, size=w_local, mode="constant", cval=0.0)
            ge = eavg[cx, cy]
            B = np.column_stack([mass, mass2, var, mx, ge]).astype(np.float64, copy=False)
            assert_finite("B_local", B)
            B_by_field.append(B)

        # Annulus multipoles on rho01 (outer-only).
        rings_end = 1 + 2 * n_radial_bins
        dip_end = rings_end + 4
        quad_end = dip_end + 4
        n_full = quad_end

        variants: list[tuple[str, int]] = [
            ("C1_annulus_rings", rings_end),
            ("C2_annulus_rings+dipole", dip_end),
            ("C3_annulus_rings+dipole+quad", quad_end),
        ]

        def kernel_fft_centered(kernel: np.ndarray) -> np.ndarray:
            """
            FFT of a *centered* spatial kernel for circular correlation (see E29 for details).
            """
            kernel = np.asarray(kernel, dtype=np.float64)
            if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
                raise ValueError(f"kernel must be square 2D, got {kernel.shape}")
            wb = int(kernel.shape[0])
            if (wb % 2) == 0:
                raise ValueError(f"kernel size must be odd, got {wb}")
            if wb > grid_size:
                raise ValueError(f"kernel size {wb} exceeds grid_size {grid_size}")
            r = wb // 2
            kr = np.flip(kernel, axis=(0, 1))
            full = np.zeros((grid_size, grid_size), dtype=np.float64)
            cx = grid_size // 2
            cy = grid_size // 2
            full[cx - r : cx + r + 1, cy - r : cy + r + 1] = kr
            full0 = np.fft.ifftshift(full)
            return np.fft.fftn(full0)

        def build_kernels() -> dict[str, np.ndarray]:
            r_local = w_local // 2
            coords = np.arange(w_big, dtype=np.float64) - float(r_big)
            dx = coords[:, None]
            dy = coords[None, :]
            core = (np.abs(dx) <= float(r_local)) & (np.abs(dy) <= float(r_local))
            outer = ~core
            outer_f = outer.astype(np.float64)
            r = np.sqrt(dx * dx + dy * dy, dtype=np.float64)

            kernels: dict[str, np.ndarray] = {"M0": outer_f}
            if r_big > r_local and outer.any():
                edges = np.linspace(float(r_local), float(r_big) + 1e-9, n_radial_bins + 1, dtype=np.float64)
                for k in range(n_radial_bins):
                    lo = edges[k]
                    hi = edges[k + 1]
                    if k == n_radial_bins - 1:
                        m = outer & (r >= lo) & (r <= hi)
                    else:
                        m = outer & (r >= lo) & (r < hi)
                    kernels[f"ring{k}"] = m.astype(np.float64)
            else:
                for k in range(n_radial_bins):
                    kernels[f"ring{k}"] = np.zeros((w_big, w_big), dtype=np.float64)

            kernels["Dx"] = (dx * outer_f).astype(np.float64, copy=False)
            kernels["Dy"] = (dy * outer_f).astype(np.float64, copy=False)
            kernels["Qxx"] = ((dx * dx - dy * dy) * outer_f).astype(np.float64, copy=False)
            kernels["Qxy"] = ((2.0 * dx * dy) * outer_f).astype(np.float64, copy=False)
            return kernels

        def plot_annulus_masks() -> Path:
            r_local = w_local // 2
            coords = np.arange(w_big, dtype=np.float64) - float(r_big)
            dx = coords[:, None]
            dy = coords[None, :]
            core = (np.abs(dx) <= float(r_local)) & (np.abs(dy) <= float(r_local))
            outer = (~core).astype(np.float64)
            r = np.sqrt(dx * dx + dy * dy, dtype=np.float64)

            edges = np.linspace(float(r_local), float(r_big) + 1e-9, n_radial_bins + 1, dtype=np.float64)
            bins: list[np.ndarray] = []
            for k in range(n_radial_bins):
                lo = edges[k]
                hi = edges[k + 1]
                if k == n_radial_bins - 1:
                    m = (r >= lo) & (r <= hi) & (~core)
                else:
                    m = (r >= lo) & (r < hi) & (~core)
                bins.append(m.astype(np.float64))

            ncols = 1 + len(bins)
            fig, axes = plt.subplots(1, ncols, figsize=(3.0 * ncols, 3.0))
            if ncols == 1:
                axes = [axes]
            axes[0].imshow(outer, cmap="gray", interpolation="nearest")
            axes[0].set_title(f"outer (w_big={w_big}, w_local={w_local})")
            axes[0].set_xticks([])
            axes[0].set_yticks([])
            for k, m in enumerate(bins):
                axes[k + 1].imshow(m, cmap="gray", interpolation="nearest")
                axes[k + 1].set_title(f"ring {k}")
                axes[k + 1].set_xticks([])
                axes[k + 1].set_yticks([])
            fig.suptitle("E30 annulus + radial bins (outer-only)")
            out = paths.run_dir / f"annulus_masks_wlocal{w_local}_wbig{w_big}.png"
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)
            return out

        mask_png = str(plot_annulus_masks())
        kernels = build_kernels()
        kernel_ffts = {name: kernel_fft_centered(k) for name, k in kernels.items()}

        Ffull_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            field = rho01_by_field[field_id]
            centers = centers_by_field[field_id]
            cx = centers[:, 0].astype(np.int64)
            cy = centers[:, 1].astype(np.int64)
            F = np.zeros((patches_per_field, n_full), dtype=np.float64)

            field_fft = np.fft.fftn(field)

            m0_grid = np.fft.ifftn(field_fft * kernel_ffts["M0"]).real
            M0 = m0_grid[cx, cy].astype(np.float64, copy=False)
            denom = M0 + eps_norm
            F[:, 0] = M0

            ring_sums = np.zeros((patches_per_field, n_radial_bins), dtype=np.float64)
            for k in range(n_radial_bins):
                gk = np.fft.ifftn(field_fft * kernel_ffts[f"ring{k}"]).real
                ring_sums[:, k] = gk[cx, cy].astype(np.float64, copy=False)
            ring_fracs = ring_sums / denom[:, None]
            F[:, 1 : 1 + n_radial_bins] = ring_sums
            F[:, 1 + n_radial_bins : 1 + 2 * n_radial_bins] = ring_fracs

            dx_grid = np.fft.ifftn(field_fft * kernel_ffts["Dx"]).real
            dy_grid = np.fft.ifftn(field_fft * kernel_ffts["Dy"]).real
            qxx_grid = np.fft.ifftn(field_fft * kernel_ffts["Qxx"]).real
            qxy_grid = np.fft.ifftn(field_fft * kernel_ffts["Qxy"]).real
            Dx = dx_grid[cx, cy].astype(np.float64, copy=False)
            Dy = dy_grid[cx, cy].astype(np.float64, copy=False)
            Qxx = qxx_grid[cx, cy].astype(np.float64, copy=False)
            Qxy = qxy_grid[cx, cy].astype(np.float64, copy=False)

            F[:, rings_end + 0] = Dx
            F[:, rings_end + 1] = Dy
            F[:, rings_end + 2] = Dx / denom
            F[:, rings_end + 3] = Dy / denom

            F[:, dip_end + 0] = Qxx
            F[:, dip_end + 1] = Qxy
            F[:, dip_end + 2] = Qxx / denom
            F[:, dip_end + 3] = Qxy / denom

            if field_id == 0 and validate_fft_n > 0:
                n_check = min(int(validate_fft_n), patches_per_field)
                feat_names = (
                    ["M0"]
                    + [f"ring_sum_{k}" for k in range(n_radial_bins)]
                    + [f"ring_frac_{k}" for k in range(n_radial_bins)]
                    + ["Dx", "Dy", "Dx_n", "Dy_n", "Qxx", "Qxy", "Qxx_n", "Qxy_n"]
                )
                for j in range(n_check):
                    cxi = int(cx[j])
                    cyi = int(cy[j])
                    patch_big = field[cxi - r_big : cxi + r_big + 1, cyi - r_big : cyi + r_big + 1]
                    d = nonlocal_annulus_moments(
                        patch_big,
                        w_local=w_local,
                        n_radial_bins=n_radial_bins,
                        include_dipole=True,
                        include_quadrupole=True,
                        eps=eps_norm,
                    )
                    row_d = np.zeros((n_full,), dtype=np.float64)
                    row_d[0] = float(d["M0"])
                    for k in range(n_radial_bins):
                        row_d[1 + k] = float(d[f"ring_sum_{k}"])
                        row_d[1 + n_radial_bins + k] = float(d[f"ring_frac_{k}"])
                    row_d[rings_end + 0] = float(d["Dx"])
                    row_d[rings_end + 1] = float(d["Dy"])
                    row_d[rings_end + 2] = float(d["Dx_n"])
                    row_d[rings_end + 3] = float(d["Dy_n"])
                    row_d[dip_end + 0] = float(d["Qxx"])
                    row_d[dip_end + 1] = float(d["Qxy"])
                    row_d[dip_end + 2] = float(d["Qxx_n"])
                    row_d[dip_end + 3] = float(d["Qxy_n"])

                    for idx, name in enumerate(feat_names):
                        a = float(F[j, idx])
                        b = float(row_d[idx])
                        if not np.isclose(a, b, rtol=validate_fft_rtol, atol=validate_fft_atol):
                            raise RuntimeError(
                                "E30 annulus FFT feature mismatch "
                                f"(field_id={field_id}, sample={j}, {name}): fft={a:.6g} direct={b:.6g}"
                            )

            assert_finite("annulus_F", F)
            Ffull_by_field.append(F)

        # Targets per k0_frac: (gx_low, gy_low) sampled at centers.
        y_by_k0_field: dict[float, list[np.ndarray]] = {}
        split_sanity: dict[float, dict[str, float]] = {}
        for k0 in k0_fracs:
            y_list: list[np.ndarray] = []
            rel_errs_gx: list[float] = []
            rel_errs_gy: list[float] = []
            var_fracs_gx: list[float] = []
            var_fracs_gy: list[float] = []
            for field_id in range(n_fields):
                rho01 = rho01_by_field[field_id]
                split = band_split_poisson_2d(rho01, k0_frac=float(k0))
                rel_errs_gx.append(float(split.rel_err_gx))
                rel_errs_gy.append(float(split.rel_err_gy))
                var_fracs_gx.append(float(np.var(split.low.gx) / (np.var(split.full.gx) + 1e-12)))
                var_fracs_gy.append(float(np.var(split.low.gy) / (np.var(split.full.gy) + 1e-12)))

                centers = centers_by_field[field_id]
                cx = centers[:, 0].astype(np.int64)
                cy = centers[:, 1].astype(np.int64)
                gx_low_c = split.low.gx[cx, cy]
                gy_low_c = split.low.gy[cx, cy]
                y = np.column_stack([gx_low_c, gy_low_c]).astype(np.float64, copy=False)
                assert_finite("y_low", y)
                y_list.append(y)

            y_by_k0_field[float(k0)] = y_list
            split_sanity[float(k0)] = {
                "rel_err_gx_mean": float(np.mean(rel_errs_gx)),
                "rel_err_gy_mean": float(np.mean(rel_errs_gy)),
                "var_frac_gx_low_mean": float(np.mean(var_fracs_gx)),
                "var_frac_gy_low_mean": float(np.mean(var_fracs_gy)),
            }

        # Sanity plot: variance fractions vs k0_frac (field-averaged).
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        ks = np.array(sorted(split_sanity.keys()), dtype=np.float64)
        vx = np.array([split_sanity[k]["var_frac_gx_low_mean"] for k in ks], dtype=np.float64)
        vy = np.array([split_sanity[k]["var_frac_gy_low_mean"] for k in ks], dtype=np.float64)
        ax.plot(ks, vx, "-o", label="var_frac gx_low/full")
        ax.plot(ks, vy, "-o", label="var_frac gy_low/full")
        ax.set_xlabel("k0_frac")
        ax.set_ylabel("variance fraction")
        ax.set_title("E30 band-split sanity (field-avg)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        sanity_png = paths.run_dir / "band_split_sanity_varfrac.png"
        fig.tight_layout()
        fig.savefig(sanity_png, dpi=150)
        plt.close(fig)

        # LOFO fits (Ridge multi-output), with permutation placebo for the annulus block.
        rows_summary: list[dict[str, Any]] = []
        for k0 in sorted(y_by_k0_field.keys()):
            y_by_field = y_by_k0_field[k0]
            for variant, k in variants:
                fold_rows: list[dict[str, Any]] = []
                pvals: list[float] = []

                for test_field in range(n_fields):
                    train_fields = [i for i in range(n_fields) if i != test_field]

                    Btr = np.concatenate([B_by_field[fid] for fid in train_fields], axis=0)
                    Ftr_raw_full = np.concatenate([Ffull_by_field[fid] for fid in train_fields], axis=0)
                    ytr = np.concatenate([y_by_field[fid] for fid in train_fields], axis=0)

                    Bte = B_by_field[test_field]
                    Fte_raw_full = Ffull_by_field[test_field]
                    yte = y_by_field[test_field]

                    # Standardize on train only.
                    B_mu = Btr.mean(axis=0)
                    B_sd = np.where(Btr.std(axis=0) > 0, Btr.std(axis=0), 1.0)
                    Btr_s = (Btr - B_mu) / B_sd
                    Bte_s = (Bte - B_mu) / B_sd

                    F_mu = Ftr_raw_full.mean(axis=0)
                    F_sd = np.where(Ftr_raw_full.std(axis=0) > 0, Ftr_raw_full.std(axis=0), 1.0)
                    Ftr_s_full = (Ftr_raw_full - F_mu) / F_sd
                    Fte_s_full = (Fte_raw_full - F_mu) / F_sd

                    y_mu = ytr.mean(axis=0)
                    yc = ytr - y_mu

                    # Baseline B_low.
                    A = Btr_s.T @ Btr_s
                    U = Btr_s.T @ yc
                    wB = solve_ridge(A, U)
                    yB_te = Bte_s @ wB + y_mu
                    pB = pearson_mean_2d(yte, yB_te)
                    rrB = relrmse_mean_2d(yte, yB_te)

                    # Candidate Ck.
                    Ftr_s = Ftr_s_full[:, :k]
                    Fte_s = Fte_s_full[:, :k]
                    Xtr = np.concatenate([Btr_s, Ftr_s], axis=1)
                    Xte = np.concatenate([Bte_s, Fte_s], axis=1)
                    XtX = Xtr.T @ Xtr
                    Xty = Xtr.T @ yc
                    wC = solve_ridge(XtX, Xty)
                    yC_te = Xte @ wC + y_mu
                    pC = pearson_mean_2d(yte, yC_te)
                    rrC = relrmse_mean_2d(yte, yC_te)

                    deltaP = float(pC - pB)
                    deltaR = float(rrC - rrB)

                    # Placebo permutations: shuffle annulus features (TRAIN rows only), reuse full block.
                    Cmat_full = Ftr_s_full.T @ Ftr_s_full
                    null_dP: list[float] = []
                    for _ in range(n_perms):
                        pidx = rng_null.permutation(Ftr_s_full.shape[0])
                        Fp_full = Ftr_s_full[pidx]
                        D_full = Btr_s.T @ Fp_full
                        V_full = Fp_full.T @ yc

                        D = D_full[:, :k]
                        V = V_full[:k, :]
                        Cmat = Cmat_full[:k, :k]
                        XtX_p = np.block([[A, D], [D.T, Cmat]])
                        Xty_p = np.vstack([U, V])
                        wP = solve_ridge(XtX_p, Xty_p)
                        yP_te = Xte @ wP + y_mu
                        pP = pearson_mean_2d(yte, yP_te)
                        null_dP.append(float(pP - pB))
                    null = np.asarray(null_dP, dtype=np.float64)
                    p_emp = float((np.sum(null >= deltaP) + 1.0) / (float(n_perms) + 1.0))
                    pvals.append(p_emp)

                    fold_rows.append(
                        {
                            "field_id": int(test_field),
                            "pearson_B": float(pB),
                            "relRMSE_B": float(rrB),
                            "pearson_C": float(pC),
                            "relRMSE_C": float(rrC),
                            "deltaP": float(deltaP),
                            "deltaR": float(deltaR),
                            "p_emp": float(p_emp),
                        }
                    )

                dp = np.asarray([r0["deltaP"] for r0 in fold_rows], dtype=np.float64)
                dr = np.asarray([r0["deltaR"] for r0 in fold_rows], dtype=np.float64)
                pBv = np.asarray([r0["pearson_B"] for r0 in fold_rows], dtype=np.float64)
                pCv = np.asarray([r0["pearson_C"] for r0 in fold_rows], dtype=np.float64)
                rrBv = np.asarray([r0["relRMSE_B"] for r0 in fold_rows], dtype=np.float64)
                rrCv = np.asarray([r0["relRMSE_C"] for r0 in fold_rows], dtype=np.float64)
                n_pos = int(np.sum(dp > 0.0))
                fp = fisher_p(pvals)
                verdict = (float(dp.mean()) > 0.0) and (float(dr.mean()) < 0.0) and (fp < 0.05) and (n_pos >= 7)

                rows_summary.append(
                    {
                        "k0_frac": float(k0),
                        "variant": str(variant),
                        "pearson_B_mean": float(pBv.mean()),
                        "pearson_B_std": float(pBv.std(ddof=1)) if len(pBv) > 1 else 0.0,
                        "pearson_C_mean": float(pCv.mean()),
                        "pearson_C_std": float(pCv.std(ddof=1)) if len(pCv) > 1 else 0.0,
                        "relRMSE_B_mean": float(rrBv.mean()),
                        "relRMSE_B_std": float(rrBv.std(ddof=1)) if len(rrBv) > 1 else 0.0,
                        "relRMSE_C_mean": float(rrCv.mean()),
                        "relRMSE_C_std": float(rrCv.std(ddof=1)) if len(rrCv) > 1 else 0.0,
                        "deltaP_mean": float(dp.mean()),
                        "deltaP_std": float(dp.std(ddof=1)) if len(dp) > 1 else 0.0,
                        "deltaR_mean": float(dr.mean()),
                        "deltaR_std": float(dr.std(ddof=1)) if len(dr) > 1 else 0.0,
                        "fisher_p": float(fp),
                        "n_pos": int(n_pos),
                        "verdict": bool(verdict),
                    }
                )

        md_rows: list[dict[str, str]] = []
        for r0 in rows_summary:
            md_rows.append(
                {
                    "k0_frac": f"{float(r0['k0_frac']):.2f}",
                    "variant": str(r0["variant"]),
                    "Pearson_B mean±std": f"{float(r0['pearson_B_mean']):.4f} ± {float(r0['pearson_B_std']):.4f}",
                    "Pearson_C mean±std": f"{float(r0['pearson_C_mean']):.4f} ± {float(r0['pearson_C_std']):.4f}",
                    "ΔPearson mean±std": f"{float(r0['deltaP_mean']):.4f} ± {float(r0['deltaP_std']):.4f}",
                    "ΔrelRMSE mean±std": f"{float(r0['deltaR_mean']):.4f} ± {float(r0['deltaR_std']):.4f}",
                    "Fisher p": fmt(float(r0["fisher_p"])),
                    "#folds ΔP>0": f"{int(r0['n_pos'])}/{n_fields}",
                    "verdict": "PASS" if bool(r0["verdict"]) else "FAIL",
                }
            )

        sanity_lines = "\n".join(
            [
                f"- k0_frac={k:.2f}: rel_err_gx_mean={split_sanity[k]['rel_err_gx_mean']:.3e}, "
                f"rel_err_gy_mean={split_sanity[k]['rel_err_gy_mean']:.3e}, "
                f"var_frac_gx_low_mean={split_sanity[k]['var_frac_gx_low_mean']:.3f}, "
                f"var_frac_gy_low_mean={split_sanity[k]['var_frac_gy_low_mean']:.3f}"
                for k in sorted(split_sanity.keys())
            ]
        )

        summary_md = (
            "# E30 — Predict low-k g from annulus dipole (LOFO)\n\n"
            f"- run: `{paths.run_dir}`\n"
            f"- grid_size={grid_size}, alpha={alpha}, n_fields={n_fields}, patches_per_field={patches_per_field}\n"
            f"- w_local={w_local}, w_big={w_big}, n_radial_bins={n_radial_bins}\n"
            f"- k0_fracs={sorted(k0_fracs)} (fraction of Nyquist)\n"
            f"- ridge_alpha={ridge_alpha}, perms={n_perms}\n"
            f"- mask: `{mask_png}`\n"
            f"- sanity plot: `{sanity_png}`\n\n"
            "## Band-split sanity (field-avg)\n\n"
            + sanity_lines
            + "\n\n## LOFO results (targets: gx_low, gy_low)\n\n"
            + md_table(
                md_rows,
                [
                    "k0_frac",
                    "variant",
                    "Pearson_B mean±std",
                    "Pearson_C mean±std",
                    "ΔPearson mean±std",
                    "ΔrelRMSE mean±std",
                    "Fisher p",
                    "#folds ΔP>0",
                    "verdict",
                ],
            )
            + "\n"
        )
        (paths.run_dir / "summary_e30_lowk_from_dipole.md").write_text(summary_md, encoding="utf-8")

        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "grid_size": grid_size,
                "alpha": alpha,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "w_local": w_local,
                "w_big": w_big,
                "n_radial_bins": n_radial_bins,
                "k0_fracs": sorted([float(x) for x in k0_fracs]),
                "ridge_alpha": ridge_alpha,
                "n_perms": n_perms,
                "rows_summary": rows_summary,
                "split_sanity": split_sanity,
                "mask_png": mask_png,
                "sanity_png": str(sanity_png),
            },
        )
        return paths

    if experiment == "e31":
        # E31 — Two-channel LOFO model: g_full = g_high(local) + g_low(dipole).
        #
        # Decompose g via band-split in Fourier (k0_frac) and fit:
        # - low-k channel: ridge on local_B + annulus rings+dipole (and rings-only control)
        # - high-k channel: pixels-only ridge-like SGD on rho_high patches (and fixed truncated kernel control)
        # Recompose: g_full_pred = g_low_pred + g_high_pred, evaluate LOFO on g_full.
        from numpy.lib.stride_tricks import sliding_window_view
        from scipy import ndimage, stats
        from sklearn.linear_model import SGDRegressor

        import matplotlib.pyplot as plt

        from .features import nonlocal_annulus_moments

        grid_size = int(cfg.get("grid_size", 256))
        alpha = float(cfg.get("alpha", 2.0))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 10_000))
        k0_frac = float(cfg.get("k0_frac", 0.15))
        w_local = _require_odd("w_local", int(cfg.get("w_local", 65)))
        w_big = _require_odd("w_big", int(cfg.get("w_big", 129)))
        n_radial_bins = int(cfg.get("n_radial_bins", 6))
        eps_norm = float(cfg.get("eps_norm", 1e-12))

        ridge_alpha = float(cfg.get("ridge_alpha", 1.0))
        n_perms = int(cfg.get("n_perms", 200))

        pixel_epochs_full = int(cfg.get("pixel_epochs_full", 1))
        pixel_epochs_high = int(cfg.get("pixel_epochs_high", 1))
        pixel_batch_size = int(cfg.get("pixel_batch_size", 1024))
        pixel_alpha = float(cfg.get("pixel_alpha", 1e-4))
        pixel_eta0 = float(cfg.get("pixel_eta0", 0.01))
        pixel_power_t = float(cfg.get("pixel_power_t", 0.25))

        validate_fft_n = int(cfg.get("validate_fft_n", 1))
        validate_fft_rtol = float(cfg.get("validate_fft_rtol", 1e-6))
        validate_fft_atol = float(cfg.get("validate_fft_atol", 1e-6))

        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if n_fields < 2:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        if not (0.0 < float(k0_frac) < 0.5):
            raise ValueError("k0_frac must be in (0,0.5)")
        if w_local > w_big:
            raise ValueError("w_local must be <= w_big")
        if w_big > grid_size:
            raise ValueError("w_big must be <= grid_size")
        if n_radial_bins <= 0:
            raise ValueError("n_radial_bins must be > 0")
        if eps_norm <= 0:
            raise ValueError("eps_norm must be > 0")
        if ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be > 0")
        if n_perms <= 0:
            raise ValueError("n_perms must be > 0")
        if pixel_epochs_full <= 0 or pixel_epochs_high <= 0:
            raise ValueError("pixel_epochs_full/high must be > 0")
        if pixel_batch_size <= 0:
            raise ValueError("pixel_batch_size must be > 0")
        if pixel_alpha <= 0:
            raise ValueError("pixel_alpha must be > 0")
        if pixel_eta0 <= 0:
            raise ValueError("pixel_eta0 must be > 0")
        if validate_fft_n < 0:
            raise ValueError("validate_fft_n must be >= 0")
        if validate_fft_rtol < 0 or validate_fft_atol < 0:
            raise ValueError("validate_fft_rtol/atol must be >= 0")

        rng_pixel = np.random.default_rng(int(placebo_seed) + 311_311)
        rng_null = np.random.default_rng(int(placebo_seed) + 322_322)

        def fisher_p(ps: list[float]) -> float:
            if not ps:
                return float("nan")
            eps = 1.0 / float(n_perms + 1)
            ps2 = [min(1.0, max(float(p), eps)) for p in ps]
            stat = -2.0 * float(np.sum(np.log(ps2)))
            return float(stats.chi2.sf(stat, 2 * len(ps2)))

        def fmt(x: float) -> str:
            if not np.isfinite(x):
                return "nan"
            if abs(float(x)) < 1e-3:
                return f"{x:.3e}"
            return f"{x:.4f}"

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r0 in rows:
                out.append("| " + " | ".join(r0.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        def safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            am = a - float(a.mean())
            bm = b - float(b.mean())
            denom = float(np.linalg.norm(am) * np.linalg.norm(bm)) + 1e-12
            return float((am @ bm) / denom)

        def relrmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
            e = y_pred - y_true
            rmse = float(np.sqrt(np.mean(e * e)))
            sd = float(np.std(y_true))
            return rmse / (sd + 1e-12)

        def pearson_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            if y_true.shape != y_pred.shape or y_true.ndim != 2 or y_true.shape[1] != 2:
                raise ValueError(f"invalid 2D targets shapes: {y_true.shape} vs {y_pred.shape}")
            px = safe_corr_1d(y_true[:, 0], y_pred[:, 0])
            py = safe_corr_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (px + py)

        def relrmse_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            rx = relrmse_1d(y_true[:, 0], y_pred[:, 0])
            ry = relrmse_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (rx + ry)

        def solve_ridge(XtX: np.ndarray, Xty: np.ndarray) -> np.ndarray:
            d = int(XtX.shape[0])
            I = np.eye(d, dtype=np.float64)
            try:
                return np.linalg.solve(XtX + ridge_alpha * I, Xty)
            except np.linalg.LinAlgError:
                w, *_ = np.linalg.lstsq(XtX + ridge_alpha * I, Xty, rcond=None)
                return w

        def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            mu = train.mean(axis=0)
            sd = train.std(axis=0)
            sd = np.where(sd > 0, sd, 1.0)
            return (train - mu) / sd, (test - mu) / sd, mu, sd

        # Data generation + band split.
        rho01_by_field: list[np.ndarray] = []
        rhoz_by_field: list[np.ndarray] = []
        rho_high_by_field: list[np.ndarray] = []
        rho_high_z_by_field: list[np.ndarray] = []
        gx_full_by_field: list[np.ndarray] = []
        gy_full_by_field: list[np.ndarray] = []
        gx_low_by_field: list[np.ndarray] = []
        gy_low_by_field: list[np.ndarray] = []
        gx_high_by_field: list[np.ndarray] = []
        gy_high_by_field: list[np.ndarray] = []

        rel_recon_gx: list[float] = []
        rel_recon_gy: list[float] = []
        var_frac_gx_low: list[float] = []
        var_frac_gy_low: list[float] = []

        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
            split = band_split_poisson_2d(rho01, k0_frac=float(k0_frac))
            rho01_by_field.append(rho01.astype(np.float64, copy=False))
            rhoz_by_field.append(_zscore_field(rho01).astype(np.float64, copy=False))
            rho_high_by_field.append(np.asarray(split.rho_high, dtype=np.float64))
            rho_high_z_by_field.append(_zscore_field(split.rho_high).astype(np.float64, copy=False))
            gx_full_by_field.append(np.asarray(split.full.gx, dtype=np.float64))
            gy_full_by_field.append(np.asarray(split.full.gy, dtype=np.float64))
            gx_low_by_field.append(np.asarray(split.low.gx, dtype=np.float64))
            gy_low_by_field.append(np.asarray(split.low.gy, dtype=np.float64))
            gx_high_by_field.append(np.asarray(split.high.gx, dtype=np.float64))
            gy_high_by_field.append(np.asarray(split.high.gy, dtype=np.float64))

            denom_gx = float(np.mean(np.abs(split.full.gx))) + 1e-12
            denom_gy = float(np.mean(np.abs(split.full.gy))) + 1e-12
            rel_recon_gx.append(float(np.mean(np.abs(split.full.gx - (split.low.gx + split.high.gx))) / denom_gx))
            rel_recon_gy.append(float(np.mean(np.abs(split.full.gy - (split.low.gy + split.high.gy))) / denom_gy))
            var_frac_gx_low.append(float(np.var(split.low.gx) / (np.var(split.full.gx) + 1e-12)))
            var_frac_gy_low.append(float(np.var(split.low.gy) / (np.var(split.full.gy) + 1e-12)))

        # Centers: valid for w_big.
        r_big = w_big // 2
        centers_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_cent = np.random.default_rng(seed + 444_444 + 1_000 * field_id)
            cx = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
            cy = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
            centers_by_field.append(np.column_stack([cx, cy]).astype(np.int64, copy=False))

        def sample_vec_at_centers(gx: np.ndarray, gy: np.ndarray, centers: np.ndarray) -> np.ndarray:
            cx = centers[:, 0].astype(np.int64)
            cy = centers[:, 1].astype(np.int64)
            return np.column_stack([gx[cx, cy], gy[cx, cy]]).astype(np.float64, copy=False)

        y_full_by_field: list[np.ndarray] = []
        y_low_by_field: list[np.ndarray] = []
        y_high_by_field: list[np.ndarray] = []
        for fid in range(n_fields):
            centers = centers_by_field[fid]
            y_full_by_field.append(sample_vec_at_centers(gx_full_by_field[fid], gy_full_by_field[fid], centers))
            y_low_by_field.append(sample_vec_at_centers(gx_low_by_field[fid], gy_low_by_field[fid], centers))
            y_high_by_field.append(sample_vec_at_centers(gx_high_by_field[fid], gy_high_by_field[fid], centers))

        # Local_B features on rho01 (w_local).
        rloc = w_local // 2
        B_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            field = rho01_by_field[field_id]
            centers = centers_by_field[field_id]
            cx = centers[:, 0]
            cy = centers[:, 1]
            pref1 = _prefix_sum_2d(field)
            pref2 = _prefix_sum_2d(field * field)
            x0 = (cx - rloc).astype(np.int64)
            x1 = (cx + rloc + 1).astype(np.int64)
            y0 = (cy - rloc).astype(np.int64)
            y1 = (cy + rloc + 1).astype(np.int64)
            mass = _box_sum_2d(pref1, x0, x1, y0, y1)
            nvox = float(w_local * w_local)
            mean = mass / nvox
            sumsq = _box_sum_2d(pref2, x0, x1, y0, y1)
            var = np.maximum(0.0, sumsq / nvox - mean * mean)
            mass2 = mass * mass
            max_grid = ndimage.maximum_filter(field, size=w_local, mode="constant", cval=-np.inf)
            mx = max_grid[cx, cy]
            gx_f, gy_f = np.gradient(field)
            egrid = gx_f * gx_f + gy_f * gy_f
            eavg = ndimage.uniform_filter(egrid, size=w_local, mode="constant", cval=0.0)
            ge = eavg[cx, cy]
            B = np.column_stack([mass, mass2, var, mx, ge]).astype(np.float64, copy=False)
            assert_finite("B_local", B)
            B_by_field.append(B)

        # Annulus feature block: [M0, ring_sum..., ring_frac..., Dx,Dy,Dx_n,Dy_n, Qxx,Qxy,Qxx_n,Qxy_n]
        rings_end = 1 + 2 * n_radial_bins
        dip_end = rings_end + 4
        quad_end = dip_end + 4
        n_full = quad_end

        def kernel_fft_centered(kernel: np.ndarray) -> np.ndarray:
            kernel = np.asarray(kernel, dtype=np.float64)
            if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
                raise ValueError(f"kernel must be square 2D, got {kernel.shape}")
            wb = int(kernel.shape[0])
            if (wb % 2) == 0:
                raise ValueError(f"kernel size must be odd, got {wb}")
            if wb > grid_size:
                raise ValueError(f"kernel size {wb} exceeds grid_size {grid_size}")
            r = wb // 2
            kr = np.flip(kernel, axis=(0, 1))
            full = np.zeros((grid_size, grid_size), dtype=np.float64)
            cx = grid_size // 2
            cy = grid_size // 2
            full[cx - r : cx + r + 1, cy - r : cy + r + 1] = kr
            full0 = np.fft.ifftshift(full)
            return np.fft.fftn(full0)

        def build_annulus_kernels() -> dict[str, np.ndarray]:
            r_local = w_local // 2
            coords = np.arange(w_big, dtype=np.float64) - float(r_big)
            dx = coords[:, None]
            dy = coords[None, :]
            core = (np.abs(dx) <= float(r_local)) & (np.abs(dy) <= float(r_local))
            outer = ~core
            outer_f = outer.astype(np.float64)
            r = np.sqrt(dx * dx + dy * dy, dtype=np.float64)

            kernels: dict[str, np.ndarray] = {"M0": outer_f}
            edges = np.linspace(float(r_local), float(r_big) + 1e-9, n_radial_bins + 1, dtype=np.float64)
            for k in range(n_radial_bins):
                lo = edges[k]
                hi = edges[k + 1]
                if k == n_radial_bins - 1:
                    m = outer & (r >= lo) & (r <= hi)
                else:
                    m = outer & (r >= lo) & (r < hi)
                kernels[f"ring{k}"] = m.astype(np.float64)

            kernels["Dx"] = (dx * outer_f).astype(np.float64, copy=False)
            kernels["Dy"] = (dy * outer_f).astype(np.float64, copy=False)
            kernels["Qxx"] = ((dx * dx - dy * dy) * outer_f).astype(np.float64, copy=False)
            kernels["Qxy"] = ((2.0 * dx * dy) * outer_f).astype(np.float64, copy=False)
            return kernels

        def plot_annulus_masks() -> Path:
            r_local = w_local // 2
            coords = np.arange(w_big, dtype=np.float64) - float(r_big)
            dx = coords[:, None]
            dy = coords[None, :]
            core = (np.abs(dx) <= float(r_local)) & (np.abs(dy) <= float(r_local))
            outer = (~core).astype(np.float64)
            r = np.sqrt(dx * dx + dy * dy, dtype=np.float64)

            edges = np.linspace(float(r_local), float(r_big) + 1e-9, n_radial_bins + 1, dtype=np.float64)
            bins: list[np.ndarray] = []
            for k in range(n_radial_bins):
                lo = edges[k]
                hi = edges[k + 1]
                if k == n_radial_bins - 1:
                    m = (r >= lo) & (r <= hi) & (~core)
                else:
                    m = (r >= lo) & (r < hi) & (~core)
                bins.append(m.astype(np.float64))

            ncols = 1 + len(bins)
            fig, axes = plt.subplots(1, ncols, figsize=(3.0 * ncols, 3.0))
            if ncols == 1:
                axes = [axes]
            axes[0].imshow(outer, cmap="gray", interpolation="nearest")
            axes[0].set_title(f"outer (w_big={w_big}, w_local={w_local})")
            axes[0].set_xticks([])
            axes[0].set_yticks([])
            for k, m in enumerate(bins):
                axes[k + 1].imshow(m, cmap="gray", interpolation="nearest")
                axes[k + 1].set_title(f"ring {k}")
                axes[k + 1].set_xticks([])
                axes[k + 1].set_yticks([])
            fig.suptitle("E31 annulus + radial bins (outer-only)")
            out = paths.run_dir / f"annulus_masks_wlocal{w_local}_wbig{w_big}.png"
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)
            return out

        mask_png = str(plot_annulus_masks())
        kernels = build_annulus_kernels()
        kernel_ffts = {name: kernel_fft_centered(k) for name, k in kernels.items()}

        Ffull_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            field = rho01_by_field[field_id]
            centers = centers_by_field[field_id]
            cx = centers[:, 0].astype(np.int64)
            cy = centers[:, 1].astype(np.int64)
            F = np.zeros((patches_per_field, n_full), dtype=np.float64)
            field_fft = np.fft.fftn(field)

            m0_grid = np.fft.ifftn(field_fft * kernel_ffts["M0"]).real
            M0 = m0_grid[cx, cy].astype(np.float64, copy=False)
            denom = M0 + eps_norm
            F[:, 0] = M0

            ring_sums = np.zeros((patches_per_field, n_radial_bins), dtype=np.float64)
            for k in range(n_radial_bins):
                gk = np.fft.ifftn(field_fft * kernel_ffts[f"ring{k}"]).real
                ring_sums[:, k] = gk[cx, cy].astype(np.float64, copy=False)
            ring_fracs = ring_sums / denom[:, None]
            F[:, 1 : 1 + n_radial_bins] = ring_sums
            F[:, 1 + n_radial_bins : 1 + 2 * n_radial_bins] = ring_fracs

            dx_grid = np.fft.ifftn(field_fft * kernel_ffts["Dx"]).real
            dy_grid = np.fft.ifftn(field_fft * kernel_ffts["Dy"]).real
            qxx_grid = np.fft.ifftn(field_fft * kernel_ffts["Qxx"]).real
            qxy_grid = np.fft.ifftn(field_fft * kernel_ffts["Qxy"]).real
            Dx = dx_grid[cx, cy].astype(np.float64, copy=False)
            Dy = dy_grid[cx, cy].astype(np.float64, copy=False)
            Qxx = qxx_grid[cx, cy].astype(np.float64, copy=False)
            Qxy = qxy_grid[cx, cy].astype(np.float64, copy=False)

            F[:, rings_end + 0] = Dx
            F[:, rings_end + 1] = Dy
            F[:, rings_end + 2] = Dx / denom
            F[:, rings_end + 3] = Dy / denom

            F[:, dip_end + 0] = Qxx
            F[:, dip_end + 1] = Qxy
            F[:, dip_end + 2] = Qxx / denom
            F[:, dip_end + 3] = Qxy / denom

            if field_id == 0 and validate_fft_n > 0:
                n_check = min(int(validate_fft_n), patches_per_field)
                feat_names = (
                    ["M0"]
                    + [f"ring_sum_{k}" for k in range(n_radial_bins)]
                    + [f"ring_frac_{k}" for k in range(n_radial_bins)]
                    + ["Dx", "Dy", "Dx_n", "Dy_n", "Qxx", "Qxy", "Qxx_n", "Qxy_n"]
                )
                for j in range(n_check):
                    cxi = int(cx[j])
                    cyi = int(cy[j])
                    patch_big = field[cxi - r_big : cxi + r_big + 1, cyi - r_big : cyi + r_big + 1]
                    d = nonlocal_annulus_moments(
                        patch_big,
                        w_local=w_local,
                        n_radial_bins=n_radial_bins,
                        include_dipole=True,
                        include_quadrupole=True,
                        eps=eps_norm,
                    )
                    row_d = np.zeros((n_full,), dtype=np.float64)
                    row_d[0] = float(d["M0"])
                    for k in range(n_radial_bins):
                        row_d[1 + k] = float(d[f"ring_sum_{k}"])
                        row_d[1 + n_radial_bins + k] = float(d[f"ring_frac_{k}"])
                    row_d[rings_end + 0] = float(d["Dx"])
                    row_d[rings_end + 1] = float(d["Dy"])
                    row_d[rings_end + 2] = float(d["Dx_n"])
                    row_d[rings_end + 3] = float(d["Dy_n"])
                    row_d[dip_end + 0] = float(d["Qxx"])
                    row_d[dip_end + 1] = float(d["Qxy"])
                    row_d[dip_end + 2] = float(d["Qxx_n"])
                    row_d[dip_end + 3] = float(d["Qxy_n"])

                    for idx, name in enumerate(feat_names):
                        a = float(F[j, idx])
                        b = float(row_d[idx])
                        if not np.isclose(a, b, rtol=validate_fft_rtol, atol=validate_fft_atol):
                            raise RuntimeError(
                                "E31 annulus FFT feature mismatch "
                                f"(field_id={field_id}, sample={j}, {name}): fft={a:.6g} direct={b:.6g}"
                            )

            assert_finite("annulus_F", F)
            Ffull_by_field.append(F)

        # High-k fixed kernel (truncated to w_local) via impulse response through the *exact* pipeline.
        rho_delta = np.zeros((grid_size, grid_size), dtype=np.float64)
        c0 = grid_size // 2
        rho_delta[c0, c0] = 1.0
        split_delta = band_split_poisson_2d(rho_delta, k0_frac=float(k0_frac))
        r = w_local // 2
        kcorr_gx = split_delta.high.gx[c0 - r : c0 + r + 1, c0 - r : c0 + r + 1][::-1, ::-1].astype(np.float64, copy=False)
        kcorr_gy = split_delta.high.gy[c0 - r : c0 + r + 1, c0 - r : c0 + r + 1][::-1, ::-1].astype(np.float64, copy=False)
        kfft_gx = kernel_fft_centered(kcorr_gx)
        kfft_gy = kernel_fft_centered(kcorr_gy)

        y_high_kern_by_field: list[np.ndarray] = []
        for fid in range(n_fields):
            rho0 = rho01_by_field[fid] - float(rho01_by_field[fid].mean())
            field_fft = np.fft.fftn(rho0)
            gx_pred = np.fft.ifftn(field_fft * kfft_gx).real
            gy_pred = np.fft.ifftn(field_fft * kfft_gy).real
            y_high_kern_by_field.append(sample_vec_at_centers(gx_pred, gy_pred, centers_by_field[fid]))

        # Pixel baselines (vector), streamed per fold.
        def fit_pixel_model_vec(
            train_fields: list[int],
            fields_z: list[np.ndarray],
            centers_by_field: list[np.ndarray],
            w: int,
            y_by_field: list[np.ndarray],
            *,
            epochs: int,
            random_state_base: int,
        ) -> tuple[tuple[SGDRegressor, SGDRegressor], np.ndarray, np.ndarray]:
            y_all = np.concatenate([y_by_field[fid] for fid in train_fields], axis=0).astype(np.float64, copy=False)
            y_mu = y_all.mean(axis=0)
            y_sd = y_all.std(axis=0)
            y_sd = np.where((~np.isfinite(y_sd)) | (y_sd <= 0), 1.0, y_sd)

            eta0_eff = float(pixel_eta0) / float(w)
            reg_gx = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(pixel_alpha),
                learning_rate="invscaling",
                eta0=float(eta0_eff),
                power_t=float(pixel_power_t),
                max_iter=1,
                tol=None,
                fit_intercept=True,
                average=True,
                random_state=int(random_state_base) + 0,
            )
            reg_gy = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(pixel_alpha),
                learning_rate="invscaling",
                eta0=float(eta0_eff),
                power_t=float(pixel_power_t),
                max_iter=1,
                tol=None,
                fit_intercept=True,
                average=True,
                random_state=int(random_state_base) + 1,
            )

            rloc = w // 2
            for _epoch in range(int(epochs)):
                for fid in train_fields:
                    field = fields_z[fid]
                    centers = centers_by_field[fid]
                    y = y_by_field[fid]
                    win = sliding_window_view(field, (w, w))
                    perm = rng_pixel.permutation(centers.shape[0])
                    for start in range(0, centers.shape[0], pixel_batch_size):
                        idx = perm[start : start + pixel_batch_size]
                        cx = centers[idx, 0].astype(np.int64)
                        cy = centers[idx, 1].astype(np.int64)
                        Xb = win[cx - rloc, cy - rloc].reshape(len(idx), -1).astype(np.float32, copy=False)
                        reg_gx.partial_fit(Xb, (y[idx, 0] - y_mu[0]) / y_sd[0])
                        reg_gy.partial_fit(Xb, (y[idx, 1] - y_mu[1]) / y_sd[1])
            return (reg_gx, reg_gy), y_mu, y_sd

        def predict_pixel_vec(
            regs: tuple[SGDRegressor, SGDRegressor],
            y_mu: np.ndarray,
            y_sd: np.ndarray,
            field_z: np.ndarray,
            centers: np.ndarray,
            w: int,
        ) -> np.ndarray:
            reg_gx, reg_gy = regs
            rloc = w // 2
            win = sliding_window_view(field_z, (w, w))
            out = np.empty((centers.shape[0], 2), dtype=np.float64)
            for start in range(0, centers.shape[0], pixel_batch_size):
                idx = slice(start, min(centers.shape[0], start + pixel_batch_size))
                cx = centers[idx, 0].astype(np.int64)
                cy = centers[idx, 1].astype(np.int64)
                Xb = win[cx - rloc, cy - rloc].reshape(len(cx), -1).astype(np.float32, copy=False)
                out[idx, 0] = reg_gx.predict(Xb) * y_sd[0] + y_mu[0]
                out[idx, 1] = reg_gy.predict(Xb) * y_sd[1] + y_mu[1]
            return out

        # LOFO evaluation.
        fold_rows_low: dict[str, list[dict[str, Any]]] = {"C1": [], "C2": []}
        fold_rows_high: dict[str, list[dict[str, Any]]] = {"P_high": [], "K_high": []}
        fold_rows_full: dict[str, list[dict[str, Any]]] = {"B_full": [], "P_full": [], "two_Phigh": [], "two_Khigh": []}

        pvals_low_C2: list[float] = []
        pvals_full_two_Phigh: list[float] = []
        pvals_full_two_Khigh: list[float] = []

        for test_field in range(n_fields):
            train_fields = [i for i in range(n_fields) if i != test_field]

            # Assemble train/test blocks.
            Btr_raw = np.concatenate([B_by_field[fid] for fid in train_fields], axis=0)
            Bte_raw = B_by_field[test_field]
            Ftr_raw_full = np.concatenate([Ffull_by_field[fid] for fid in train_fields], axis=0)
            Fte_raw_full = Ffull_by_field[test_field]

            y_low_tr = np.concatenate([y_low_by_field[fid] for fid in train_fields], axis=0)
            y_low_te = y_low_by_field[test_field]
            y_high_te = y_high_by_field[test_field]
            y_full_tr = np.concatenate([y_full_by_field[fid] for fid in train_fields], axis=0)
            y_full_te = y_full_by_field[test_field]

            # Standardize features on train only.
            Btr, Bte, _, _ = standardize(Btr_raw, Bte_raw)
            Ftr_full, Fte_full, _, _ = standardize(Ftr_raw_full, Fte_raw_full)

            ymu_low = y_low_tr.mean(axis=0)
            yclow = y_low_tr - ymu_low
            ymu_full = y_full_tr.mean(axis=0)
            ycfull = y_full_tr - ymu_full

            # Low-k baseline B_low.
            A = Btr.T @ Btr
            U_low = Btr.T @ yclow
            wB_low = solve_ridge(A, U_low)
            y_low_B_te = Bte @ wB_low + ymu_low
            pB_low = pearson_mean_2d(y_low_te, y_low_B_te)
            rrB_low = relrmse_mean_2d(y_low_te, y_low_B_te)

            # Low-k C1 rings-only.
            k1 = rings_end
            Xtr1 = np.concatenate([Btr, Ftr_full[:, :k1]], axis=1)
            Xte1 = np.concatenate([Bte, Fte_full[:, :k1]], axis=1)
            wC1 = solve_ridge(Xtr1.T @ Xtr1, Xtr1.T @ yclow)
            y_low_C1_te = Xte1 @ wC1 + ymu_low
            pC1 = pearson_mean_2d(y_low_te, y_low_C1_te)
            rrC1 = relrmse_mean_2d(y_low_te, y_low_C1_te)

            # Low-k C2 rings+dipole (for two-channel).
            k2 = dip_end
            Xtr2 = np.concatenate([Btr, Ftr_full[:, :k2]], axis=1)
            Xte2 = np.concatenate([Bte, Fte_full[:, :k2]], axis=1)
            wC2 = solve_ridge(Xtr2.T @ Xtr2, Xtr2.T @ yclow)
            y_low_C2_te = Xte2 @ wC2 + ymu_low
            pC2 = pearson_mean_2d(y_low_te, y_low_C2_te)
            rrC2 = relrmse_mean_2d(y_low_te, y_low_C2_te)

            fold_rows_low["C1"].append(
                {
                    "field_id": int(test_field),
                    "pearson_B_low": float(pB_low),
                    "relRMSE_B_low": float(rrB_low),
                    "pearson_C1": float(pC1),
                    "relRMSE_C1": float(rrC1),
                    "deltaP_C1_vs_B": float(pC1 - pB_low),
                    "deltaR_C1_vs_B": float(rrC1 - rrB_low),
                }
            )
            fold_rows_low["C2"].append(
                {
                    "field_id": int(test_field),
                    "pearson_B_low": float(pB_low),
                    "relRMSE_B_low": float(rrB_low),
                    "pearson_C2": float(pC2),
                    "relRMSE_C2": float(rrC2),
                    "deltaP_C2_vs_B": float(pC2 - pB_low),
                    "deltaR_C2_vs_B": float(rrC2 - rrB_low),
                }
            )

            # High-k channel: pixels-only on rho_high, and fixed kernel (local truncation).
            regs_high, ymu_high_tr, ysd_high_tr = fit_pixel_model_vec(
                train_fields,
                rho_high_z_by_field,
                centers_by_field,
                w_local,
                y_high_by_field,
                epochs=pixel_epochs_high,
                random_state_base=int(placebo_seed) + 4000 + 10_000 * test_field,
            )
            y_high_P_te = predict_pixel_vec(
                regs_high,
                ymu_high_tr,
                ysd_high_tr,
                rho_high_z_by_field[test_field],
                centers_by_field[test_field],
                w_local,
            )
            y_high_K_te = y_high_kern_by_field[test_field]

            p_high_P = pearson_mean_2d(y_high_te, y_high_P_te)
            rr_high_P = relrmse_mean_2d(y_high_te, y_high_P_te)
            p_high_K = pearson_mean_2d(y_high_te, y_high_K_te)
            rr_high_K = relrmse_mean_2d(y_high_te, y_high_K_te)
            fold_rows_high["P_high"].append({"field_id": int(test_field), "pearson": float(p_high_P), "relRMSE": float(rr_high_P)})
            fold_rows_high["K_high"].append({"field_id": int(test_field), "pearson": float(p_high_K), "relRMSE": float(rr_high_K)})

            # Full-g baselines: B_full (local_B) and P_full (pixels on rho01 z-scored).
            U_full = Btr.T @ ycfull
            wB_full = solve_ridge(A, U_full)
            y_full_B_te = Bte @ wB_full + ymu_full
            pB_full = pearson_mean_2d(y_full_te, y_full_B_te)
            rrB_full = relrmse_mean_2d(y_full_te, y_full_B_te)

            regs_full, ymu_full_tr, ysd_full_tr = fit_pixel_model_vec(
                train_fields,
                rhoz_by_field,
                centers_by_field,
                w_local,
                y_full_by_field,
                epochs=pixel_epochs_full,
                random_state_base=int(placebo_seed) + 5000 + 10_000 * test_field,
            )
            y_full_P_te = predict_pixel_vec(
                regs_full,
                ymu_full_tr,
                ysd_full_tr,
                rhoz_by_field[test_field],
                centers_by_field[test_field],
                w_local,
            )
            pP_full = pearson_mean_2d(y_full_te, y_full_P_te)
            rrP_full = relrmse_mean_2d(y_full_te, y_full_P_te)

            # Two-channel recomposition (use low=C2, high={P,K}).
            y_full_two_P = y_low_C2_te + y_high_P_te
            y_full_two_K = y_low_C2_te + y_high_K_te
            p_two_P = pearson_mean_2d(y_full_te, y_full_two_P)
            rr_two_P = relrmse_mean_2d(y_full_te, y_full_two_P)
            p_two_K = pearson_mean_2d(y_full_te, y_full_two_K)
            rr_two_K = relrmse_mean_2d(y_full_te, y_full_two_K)

            fold_rows_full["B_full"].append({"field_id": int(test_field), "pearson": float(pB_full), "relRMSE": float(rrB_full)})
            fold_rows_full["P_full"].append({"field_id": int(test_field), "pearson": float(pP_full), "relRMSE": float(rrP_full)})
            fold_rows_full["two_Phigh"].append(
                {
                    "field_id": int(test_field),
                    "pearson": float(p_two_P),
                    "relRMSE": float(rr_two_P),
                    "deltaP_vs_Pfull": float(p_two_P - pP_full),
                    "deltaR_vs_Pfull": float(rr_two_P - rrP_full),
                }
            )
            fold_rows_full["two_Khigh"].append(
                {
                    "field_id": int(test_field),
                    "pearson": float(p_two_K),
                    "relRMSE": float(rr_two_K),
                    "deltaP_vs_Pfull": float(p_two_K - pP_full),
                    "deltaR_vs_Pfull": float(rr_two_K - rrP_full),
                }
            )

            # Placebo on low channel (shuffle annulus features in TRAIN), and propagate to full recomposition deltas vs P_full.
            delta_low_real = float(pC2 - pB_low)
            delta_full_P_real = float(p_two_P - pP_full)
            delta_full_K_real = float(p_two_K - pP_full)

            # Precompute blocks for fast permuted solves.
            Z_low = np.concatenate([Btr, yclow], axis=1)  # (n_train, 5+2)
            Cmat = (Ftr_full[:, :k2]).T @ (Ftr_full[:, :k2])
            null_dP_low: list[float] = []
            null_dP_full_P: list[float] = []
            null_dP_full_K: list[float] = []
            for _ in range(n_perms):
                pidx = rng_null.permutation(Ftr_full.shape[0])
                Fp = Ftr_full[pidx, :k2]
                M = Z_low.T @ Fp  # (7,k2) => [D; Yc^T Fp]
                D = M[: Btr.shape[1], :]
                V = M[Btr.shape[1] :, :].T  # (k2,2)
                XtX_p = np.block([[A, D], [D.T, Cmat]])
                Xty_p = np.vstack([U_low, V])
                wP = solve_ridge(XtX_p, Xty_p)
                y_low_perm_te = Xte2 @ wP + ymu_low
                p_low_perm = pearson_mean_2d(y_low_te, y_low_perm_te)
                null_dP_low.append(float(p_low_perm - pB_low))

                y_full_perm_P = y_low_perm_te + y_high_P_te
                y_full_perm_K = y_low_perm_te + y_high_K_te
                p_full_perm_P = pearson_mean_2d(y_full_te, y_full_perm_P)
                p_full_perm_K = pearson_mean_2d(y_full_te, y_full_perm_K)
                null_dP_full_P.append(float(p_full_perm_P - pP_full))
                null_dP_full_K.append(float(p_full_perm_K - pP_full))

            null_low = np.asarray(null_dP_low, dtype=np.float64)
            null_full_P = np.asarray(null_dP_full_P, dtype=np.float64)
            null_full_K = np.asarray(null_dP_full_K, dtype=np.float64)

            p_emp_low = float((np.sum(null_low >= delta_low_real) + 1.0) / (float(n_perms) + 1.0))
            p_emp_full_P = float((np.sum(null_full_P >= delta_full_P_real) + 1.0) / (float(n_perms) + 1.0))
            p_emp_full_K = float((np.sum(null_full_K >= delta_full_K_real) + 1.0) / (float(n_perms) + 1.0))
            pvals_low_C2.append(p_emp_low)
            pvals_full_two_Phigh.append(p_emp_full_P)
            pvals_full_two_Khigh.append(p_emp_full_K)

        # Aggregate fold metrics.
        def agg(rows: list[dict[str, Any]], key: str) -> tuple[float, float]:
            v = np.asarray([float(r0[key]) for r0 in rows], dtype=np.float64)
            return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0

        # Low channel summary (C2 vs B).
        low_C2_dp = np.asarray([r0["deltaP_C2_vs_B"] for r0 in fold_rows_low["C2"]], dtype=np.float64)
        low_C2_dr = np.asarray([r0["deltaR_C2_vs_B"] for r0 in fold_rows_low["C2"]], dtype=np.float64)
        low_C2_npos = int(np.sum(low_C2_dp > 0))
        low_C2_fp = fisher_p(pvals_low_C2)
        low_C2_verdict = (float(low_C2_dp.mean()) > 0.0) and (float(low_C2_dr.mean()) < 0.0) and (low_C2_fp < 0.05) and (low_C2_npos >= 7)

        # Full channel improvements vs P_full.
        full_twoP_dp = np.asarray([r0["deltaP_vs_Pfull"] for r0 in fold_rows_full["two_Phigh"]], dtype=np.float64)
        full_twoP_dr = np.asarray([r0["deltaR_vs_Pfull"] for r0 in fold_rows_full["two_Phigh"]], dtype=np.float64)
        full_twoP_npos = int(np.sum(full_twoP_dp > 0))
        full_twoP_fp = fisher_p(pvals_full_two_Phigh)

        full_twoK_dp = np.asarray([r0["deltaP_vs_Pfull"] for r0 in fold_rows_full["two_Khigh"]], dtype=np.float64)
        full_twoK_dr = np.asarray([r0["deltaR_vs_Pfull"] for r0 in fold_rows_full["two_Khigh"]], dtype=np.float64)
        full_twoK_npos = int(np.sum(full_twoK_dp > 0))
        full_twoK_fp = fisher_p(pvals_full_two_Khigh)

        # Tables.
        md_low = [
            {
                "model": "B_low (local_B)",
                "Pearson mean±std": f"{agg([{'pearson': r['pearson_B_low']} for r in fold_rows_low['C2']], 'pearson')[0]:.4f} ± {agg([{'pearson': r['pearson_B_low']} for r in fold_rows_low['C2']], 'pearson')[1]:.4f}",
                "relRMSE mean±std": f"{agg([{'rel': r['relRMSE_B_low']} for r in fold_rows_low['C2']], 'rel')[0]:.4f} ± {agg([{'rel': r['relRMSE_B_low']} for r in fold_rows_low['C2']], 'rel')[1]:.4f}",
            },
            {
                "model": "C1_low (rings)",
                "Pearson mean±std": f"{agg([{'pearson': r['pearson_C1']} for r in fold_rows_low['C1']], 'pearson')[0]:.4f} ± {agg([{'pearson': r['pearson_C1']} for r in fold_rows_low['C1']], 'pearson')[1]:.4f}",
                "relRMSE mean±std": f"{agg([{'rel': r['relRMSE_C1']} for r in fold_rows_low['C1']], 'rel')[0]:.4f} ± {agg([{'rel': r['relRMSE_C1']} for r in fold_rows_low['C1']], 'rel')[1]:.4f}",
            },
            {
                "model": "C2_low (rings+dipole)",
                "Pearson mean±std": f"{agg([{'pearson': r['pearson_C2']} for r in fold_rows_low['C2']], 'pearson')[0]:.4f} ± {agg([{'pearson': r['pearson_C2']} for r in fold_rows_low['C2']], 'pearson')[1]:.4f}",
                "relRMSE mean±std": f"{agg([{'rel': r['relRMSE_C2']} for r in fold_rows_low['C2']], 'rel')[0]:.4f} ± {agg([{'rel': r['relRMSE_C2']} for r in fold_rows_low['C2']], 'rel')[1]:.4f}",
            },
            {
                "model": "Δ(C2-B_low) + placebo",
                "Pearson mean±std": f"{float(low_C2_dp.mean()):.4f} ± {float(low_C2_dp.std(ddof=1)):.4f}",
                "relRMSE mean±std": f"{float(low_C2_dr.mean()):.4f} ± {float(low_C2_dr.std(ddof=1)):.4f} (Fisher p={fmt(low_C2_fp)}, {low_C2_npos}/{n_fields}, {('PASS' if low_C2_verdict else 'FAIL')})",
            },
        ]

        md_high = [
            {
                "model": "P_high (pixels on rho_high)",
                "Pearson mean±std": f"{agg(fold_rows_high['P_high'], 'pearson')[0]:.4f} ± {agg(fold_rows_high['P_high'], 'pearson')[1]:.4f}",
                "relRMSE mean±std": f"{agg(fold_rows_high['P_high'], 'relRMSE')[0]:.4f} ± {agg(fold_rows_high['P_high'], 'relRMSE')[1]:.4f}",
            },
            {
                "model": "K_high (fixed kernel, w_local)",
                "Pearson mean±std": f"{agg(fold_rows_high['K_high'], 'pearson')[0]:.4f} ± {agg(fold_rows_high['K_high'], 'pearson')[1]:.4f}",
                "relRMSE mean±std": f"{agg(fold_rows_high['K_high'], 'relRMSE')[0]:.4f} ± {agg(fold_rows_high['K_high'], 'relRMSE')[1]:.4f}",
            },
        ]

        md_full = [
            {
                "model": "B_full (local_B)",
                "Pearson mean±std": f"{agg(fold_rows_full['B_full'], 'pearson')[0]:.4f} ± {agg(fold_rows_full['B_full'], 'pearson')[1]:.4f}",
                "relRMSE mean±std": f"{agg(fold_rows_full['B_full'], 'relRMSE')[0]:.4f} ± {agg(fold_rows_full['B_full'], 'relRMSE')[1]:.4f}",
            },
            {
                "model": "P_full (pixels on rho)",
                "Pearson mean±std": f"{agg(fold_rows_full['P_full'], 'pearson')[0]:.4f} ± {agg(fold_rows_full['P_full'], 'pearson')[1]:.4f}",
                "relRMSE mean±std": f"{agg(fold_rows_full['P_full'], 'relRMSE')[0]:.4f} ± {agg(fold_rows_full['P_full'], 'relRMSE')[1]:.4f}",
            },
            {
                "model": "two-channel (low=C2, high=P_high)",
                "Pearson mean±std": f"{agg(fold_rows_full['two_Phigh'], 'pearson')[0]:.4f} ± {agg(fold_rows_full['two_Phigh'], 'pearson')[1]:.4f}",
                "relRMSE mean±std": f"{agg(fold_rows_full['two_Phigh'], 'relRMSE')[0]:.4f} ± {agg(fold_rows_full['two_Phigh'], 'relRMSE')[1]:.4f}",
            },
            {
                "model": "Δ(two-P_high vs P_full) + placebo",
                "Pearson mean±std": f"{float(full_twoP_dp.mean()):.4f} ± {float(full_twoP_dp.std(ddof=1)):.4f}",
                "relRMSE mean±std": f"{float(full_twoP_dr.mean()):.4f} ± {float(full_twoP_dr.std(ddof=1)):.4f} (Fisher p={fmt(full_twoP_fp)}, {full_twoP_npos}/{n_fields})",
            },
            {
                "model": "two-channel (low=C2, high=K_high)",
                "Pearson mean±std": f"{agg(fold_rows_full['two_Khigh'], 'pearson')[0]:.4f} ± {agg(fold_rows_full['two_Khigh'], 'pearson')[1]:.4f}",
                "relRMSE mean±std": f"{agg(fold_rows_full['two_Khigh'], 'relRMSE')[0]:.4f} ± {agg(fold_rows_full['two_Khigh'], 'relRMSE')[1]:.4f}",
            },
            {
                "model": "Δ(two-K_high vs P_full) + placebo",
                "Pearson mean±std": f"{float(full_twoK_dp.mean()):.4f} ± {float(full_twoK_dp.std(ddof=1)):.4f}",
                "relRMSE mean±std": f"{float(full_twoK_dr.mean()):.4f} ± {float(full_twoK_dr.std(ddof=1)):.4f} (Fisher p={fmt(full_twoK_fp)}, {full_twoK_npos}/{n_fields})",
            },
        ]

        # Save a quick plot of recon sanity + variance fractions (field-avg).
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        ax.plot([1], [float(np.mean(rel_recon_gx))], "o", label="rel_recon_gx_mean")
        ax.plot([2], [float(np.mean(rel_recon_gy))], "o", label="rel_recon_gy_mean")
        ax.plot([3], [float(np.mean(var_frac_gx_low))], "o", label="var_frac_gx_low_mean")
        ax.plot([4], [float(np.mean(var_frac_gy_low))], "o", label="var_frac_gy_low_mean")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["recon_gx", "recon_gy", "varfrac_gx", "varfrac_gy"])
        ax.set_ylabel("value")
        ax.set_title("E31 band-split sanity (field-avg)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        sanity_png = paths.run_dir / "band_split_sanity.png"
        fig.tight_layout()
        fig.savefig(sanity_png, dpi=150)
        plt.close(fig)

        summary_md = (
            "# E31 — Two-channel full-g predictor (LOFO)\n\n"
            f"- run: `{paths.run_dir}`\n"
            f"- grid_size={grid_size}, alpha={alpha}, n_fields={n_fields}, patches_per_field={patches_per_field}\n"
            f"- k0_frac={k0_frac}, w_local={w_local}, w_big={w_big}, n_radial_bins={n_radial_bins}\n"
            f"- ridge_alpha={ridge_alpha}, perms={n_perms}\n"
            f"- pixels: alpha={pixel_alpha}, batch={pixel_batch_size}, epochs_full={pixel_epochs_full}, epochs_high={pixel_epochs_high}\n"
            f"- mask: `{mask_png}`\n"
            f"- sanity: `{sanity_png}`\n\n"
            "## Decomposition sanity (field-avg)\n\n"
            f"- rel_recon_gx_mean={float(np.mean(rel_recon_gx)):.3e}, rel_recon_gy_mean={float(np.mean(rel_recon_gy)):.3e}\n"
            f"- var_frac_gx_low_mean={float(np.mean(var_frac_gx_low)):.3f}, var_frac_gy_low_mean={float(np.mean(var_frac_gy_low)):.3f}\n\n"
            "## Low-k channel (targets: gx_low, gy_low)\n\n"
            + md_table(md_low, ["model", "Pearson mean±std", "relRMSE mean±std"])
            + "\n\n## High-k channel (targets: gx_high, gy_high)\n\n"
            + md_table(md_high, ["model", "Pearson mean±std", "relRMSE mean±std"])
            + "\n\n## Full g (targets: gx_full, gy_full)\n\n"
            + md_table(md_full, ["model", "Pearson mean±std", "relRMSE mean±std"])
            + "\n"
        )
        (paths.run_dir / "summary_e31_two_channel_fullg.md").write_text(summary_md, encoding="utf-8")

        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "grid_size": grid_size,
                "alpha": alpha,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "k0_frac": k0_frac,
                "w_local": w_local,
                "w_big": w_big,
                "n_radial_bins": n_radial_bins,
                "ridge_alpha": ridge_alpha,
                "n_perms": n_perms,
                "pixels": {
                    "alpha": pixel_alpha,
                    "batch_size": pixel_batch_size,
                    "epochs_full": pixel_epochs_full,
                    "epochs_high": pixel_epochs_high,
                    "eta0": pixel_eta0,
                    "power_t": pixel_power_t,
                },
                "split_sanity": {
                    "rel_recon_gx_mean": float(np.mean(rel_recon_gx)),
                    "rel_recon_gy_mean": float(np.mean(rel_recon_gy)),
                    "var_frac_gx_low_mean": float(np.mean(var_frac_gx_low)),
                    "var_frac_gy_low_mean": float(np.mean(var_frac_gy_low)),
                },
                "fisher_p_low_C2": float(low_C2_fp),
                "fisher_p_full_two_Phigh": float(full_twoP_fp),
                "fisher_p_full_two_Khigh": float(full_twoK_fp),
                "fold_rows": {
                    "low_C1": fold_rows_low["C1"],
                    "low_C2": fold_rows_low["C2"],
                    "high_P": fold_rows_high["P_high"],
                    "high_K": fold_rows_high["K_high"],
                    "full_B": fold_rows_full["B_full"],
                    "full_P": fold_rows_full["P_full"],
                    "full_two_Phigh": fold_rows_full["two_Phigh"],
                    "full_two_Khigh": fold_rows_full["two_Khigh"],
                },
                "paths": {
                    "mask_png": mask_png,
                    "sanity_png": str(sanity_png),
                },
            },
        )
        return paths

    if experiment == "e32":
        # E32 — Improve low-k estimator with multi-annulus dipole profile (LOFO).
        #
        # Task: predict (gx_low, gy_low) at patch centers using a richer radial profile:
        # - Cmulti: local_B + {Mass_m, Dx_m, Dy_m} for m in concentric radial bins on a big window
        # Placebo: shuffle multi-annulus features (train rows) and test ΔPearson vs baselines.
        from scipy import ndimage, stats

        import matplotlib.pyplot as plt

        from .features import nonlocal_annulus_moments

        grid_size = int(cfg.get("grid_size", 256))
        alpha = float(cfg.get("alpha", 2.0))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 10_000))
        k0_frac = float(cfg.get("k0_frac", 0.15))
        w_local = _require_odd("w_local", int(cfg.get("w_local", 65)))

        w_bigs_cfg = cfg.get("w_bigs", [129])
        if isinstance(w_bigs_cfg, (int, float, str)):
            w_bigs = [_require_odd("w_big", int(w_bigs_cfg))]
        else:
            w_bigs = [_require_odd("w_big", int(w)) for w in list(w_bigs_cfg)]
        if not w_bigs:
            raise ValueError("w_bigs must be non-empty")

        n_annuli = int(cfg.get("n_annuli", 12))
        ring_bin_mode = str(cfg.get("ring_bin_mode", "uniform_r")).lower()
        eps_norm = float(cfg.get("eps_norm", 1e-12))

        ridge_alpha = float(cfg.get("ridge_alpha", 1.0))
        n_perms = int(cfg.get("n_perms", 200))

        validate_fft_n = int(cfg.get("validate_fft_n", 1))
        validate_fft_rtol = float(cfg.get("validate_fft_rtol", 1e-6))
        validate_fft_atol = float(cfg.get("validate_fft_atol", 1e-6))

        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if n_fields < 2:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        if not (0.0 < float(k0_frac) < 0.5):
            raise ValueError("k0_frac must be in (0,0.5)")
        if any(w_local > wb for wb in w_bigs):
            raise ValueError("All w_bigs must be >= w_local")
        if any(wb > grid_size for wb in w_bigs):
            raise ValueError("All w_bigs must be <= grid_size")
        if n_annuli <= 0:
            raise ValueError("n_annuli must be > 0")
        if ring_bin_mode not in {"uniform_r", "equal_area"}:
            raise ValueError("ring_bin_mode must be 'uniform_r' or 'equal_area'")
        if eps_norm <= 0:
            raise ValueError("eps_norm must be > 0")
        if ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be > 0")
        if n_perms <= 0:
            raise ValueError("n_perms must be > 0")
        if validate_fft_n < 0:
            raise ValueError("validate_fft_n must be >= 0")
        if validate_fft_rtol < 0 or validate_fft_atol < 0:
            raise ValueError("validate_fft_rtol/atol must be >= 0")

        rng_null = np.random.default_rng(int(placebo_seed) + 323_323)

        def fisher_p(ps: list[float]) -> float:
            if not ps:
                return float("nan")
            eps = 1.0 / float(n_perms + 1)
            ps2 = [min(1.0, max(float(p), eps)) for p in ps]
            stat = -2.0 * float(np.sum(np.log(ps2)))
            return float(stats.chi2.sf(stat, 2 * len(ps2)))

        def fmt(x: float) -> str:
            if not np.isfinite(x):
                return "nan"
            if abs(float(x)) < 1e-3:
                return f"{x:.3e}"
            return f"{x:.4f}"

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r0 in rows:
                out.append("| " + " | ".join(r0.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        def safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            am = a - float(a.mean())
            bm = b - float(b.mean())
            denom = float(np.linalg.norm(am) * np.linalg.norm(bm)) + 1e-12
            return float((am @ bm) / denom)

        def relrmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
            e = y_pred - y_true
            rmse = float(np.sqrt(np.mean(e * e)))
            sd = float(np.std(y_true))
            return rmse / (sd + 1e-12)

        def pearson_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            if y_true.shape != y_pred.shape or y_true.ndim != 2 or y_true.shape[1] != 2:
                raise ValueError(f"invalid 2D targets shapes: {y_true.shape} vs {y_pred.shape}")
            px = safe_corr_1d(y_true[:, 0], y_pred[:, 0])
            py = safe_corr_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (px + py)

        def relrmse_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            rx = relrmse_1d(y_true[:, 0], y_pred[:, 0])
            ry = relrmse_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (rx + ry)

        def solve_ridge(XtX: np.ndarray, Xty: np.ndarray) -> np.ndarray:
            d = int(XtX.shape[0])
            I = np.eye(d, dtype=np.float64)
            try:
                return np.linalg.solve(XtX + ridge_alpha * I, Xty)
            except np.linalg.LinAlgError:
                w, *_ = np.linalg.lstsq(XtX + ridge_alpha * I, Xty, rcond=None)
                return w

        def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            mu = train.mean(axis=0)
            sd = train.std(axis=0)
            sd = np.where(sd > 0, sd, 1.0)
            return (train - mu) / sd, (test - mu) / sd, mu, sd

        # Shared field generation + low-k splits.
        rho01_by_field: list[np.ndarray] = []
        split_cache: list[BandSplit2D] = []
        rel_recon_gx: list[float] = []
        rel_recon_gy: list[float] = []
        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
            split = band_split_poisson_2d(rho01, k0_frac=float(k0_frac))
            rho01_by_field.append(rho01.astype(np.float64, copy=False))
            split_cache.append(split)
            rel_recon_gx.append(float(split.rel_err_gx))
            rel_recon_gy.append(float(split.rel_err_gy))

        # Centers (per w_big), then targets at those centers.
        centers_by_wbig_field: dict[int, list[np.ndarray]] = {}
        for wb in w_bigs:
            r_big = wb // 2
            centers_by_field: list[np.ndarray] = []
            for field_id in range(n_fields):
                rng_cent = np.random.default_rng(seed + 444_444 + 10_000 * wb + 1_000 * field_id)
                cx = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
                cy = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
                centers_by_field.append(np.column_stack([cx, cy]).astype(np.int64, copy=False))
            centers_by_wbig_field[int(wb)] = centers_by_field

        def sample_low_at_centers(split: BandSplit2D, centers: np.ndarray) -> np.ndarray:
            cx = centers[:, 0].astype(np.int64)
            cy = centers[:, 1].astype(np.int64)
            return np.column_stack([split.low.gx[cx, cy], split.low.gy[cx, cy]]).astype(np.float64, copy=False)

        # Local_B features for a given centers set.
        def build_local_B(field: np.ndarray, centers: np.ndarray) -> np.ndarray:
            rloc = w_local // 2
            cx = centers[:, 0]
            cy = centers[:, 1]
            pref1 = _prefix_sum_2d(field)
            pref2 = _prefix_sum_2d(field * field)
            x0 = (cx - rloc).astype(np.int64)
            x1 = (cx + rloc + 1).astype(np.int64)
            y0 = (cy - rloc).astype(np.int64)
            y1 = (cy + rloc + 1).astype(np.int64)
            mass = _box_sum_2d(pref1, x0, x1, y0, y1)
            nvox = float(w_local * w_local)
            mean = mass / nvox
            sumsq = _box_sum_2d(pref2, x0, x1, y0, y1)
            var = np.maximum(0.0, sumsq / nvox - mean * mean)
            mass2 = mass * mass
            max_grid = ndimage.maximum_filter(field, size=w_local, mode="constant", cval=-np.inf)
            mx = max_grid[cx, cy]
            gx_f, gy_f = np.gradient(field)
            egrid = gx_f * gx_f + gy_f * gy_f
            eavg = ndimage.uniform_filter(egrid, size=w_local, mode="constant", cval=0.0)
            ge = eavg[cx, cy]
            B = np.column_stack([mass, mass2, var, mx, ge]).astype(np.float64, copy=False)
            assert_finite("B_local", B)
            return B

        # FFT correlation kernels.
        def kernel_fft_centered(kernel: np.ndarray, *, grid_size: int) -> np.ndarray:
            kernel = np.asarray(kernel, dtype=np.float64)
            if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
                raise ValueError(f"kernel must be square 2D, got {kernel.shape}")
            wb = int(kernel.shape[0])
            if (wb % 2) == 0:
                raise ValueError(f"kernel size must be odd, got {wb}")
            if wb > grid_size:
                raise ValueError(f"kernel size {wb} exceeds grid_size {grid_size}")
            r = wb // 2
            kr = np.flip(kernel, axis=(0, 1))
            full = np.zeros((grid_size, grid_size), dtype=np.float64)
            cx = grid_size // 2
            cy = grid_size // 2
            full[cx - r : cx + r + 1, cy - r : cy + r + 1] = kr
            full0 = np.fft.ifftshift(full)
            return np.fft.fftn(full0)

        def annulus_edges(r_min: float, r_max: float, n_bins: int) -> np.ndarray:
            if ring_bin_mode == "uniform_r":
                return np.linspace(r_min, r_max + 1e-9, n_bins + 1, dtype=np.float64)
            # equal_area: equal spacing in r^2
            rsq = np.linspace(r_min * r_min, r_max * r_max + 1e-9, n_bins + 1, dtype=np.float64)
            return np.sqrt(rsq, dtype=np.float64)

        # Results across w_big settings.
        rows_summary: list[dict[str, Any]] = []
        coeff_plots: dict[str, str] = {}

        for wb in w_bigs:
            r_big = int(wb) // 2
            r_local = w_local // 2
            if wb <= w_local:
                raise ValueError("E32 requires w_big > w_local to form annuli outside the local core")

            centers_by_field = centers_by_wbig_field[int(wb)]
            y_low_by_field = [sample_low_at_centers(split_cache[fid], centers_by_field[fid]) for fid in range(n_fields)]
            B_by_field = [build_local_B(rho01_by_field[fid], centers_by_field[fid]) for fid in range(n_fields)]

            # C2 baseline features (single outer annulus + 6 radial bins + dipole), matching E31.
            n_radial_bins_c2 = int(cfg.get("n_radial_bins_c2", 6))
            rings_end_c2 = 1 + 2 * n_radial_bins_c2
            dip_end_c2 = rings_end_c2 + 4

            # Multi-annulus features: per ring m: [Mass_m, Dx_m, Dy_m] (and optionally only dipoles).
            edges = annulus_edges(float(r_local), float(r_big), int(n_annuli))
            radii = 0.5 * (edges[:-1] + edges[1:])

            # Build kernels on w_big grid.
            coords = np.arange(int(wb), dtype=np.float64) - float(r_big)
            dx = coords[:, None]
            dy = coords[None, :]
            core = (np.abs(dx) <= float(r_local)) & (np.abs(dy) <= float(r_local))
            outer = ~core
            rgrid = np.sqrt(dx * dx + dy * dy, dtype=np.float64)

            # For C2:
            kernels_c2: dict[str, np.ndarray] = {"M0": outer.astype(np.float64)}
            edges_c2 = annulus_edges(float(r_local), float(r_big), n_radial_bins_c2)
            for k in range(n_radial_bins_c2):
                lo = edges_c2[k]
                hi = edges_c2[k + 1]
                if k == n_radial_bins_c2 - 1:
                    m = outer & (rgrid >= lo) & (rgrid <= hi)
                else:
                    m = outer & (rgrid >= lo) & (rgrid < hi)
                kernels_c2[f"ring{k}"] = m.astype(np.float64)
            kernels_c2["Dx"] = (dx * outer).astype(np.float64, copy=False)
            kernels_c2["Dy"] = (dy * outer).astype(np.float64, copy=False)

            # Multi-annulus kernels:
            kernels_multi: dict[str, np.ndarray] = {}
            for m in range(int(n_annuli)):
                lo = edges[m]
                hi = edges[m + 1]
                if m == int(n_annuli) - 1:
                    ring = outer & (rgrid >= lo) & (rgrid <= hi)
                else:
                    ring = outer & (rgrid >= lo) & (rgrid < hi)
                ring_f = ring.astype(np.float64)
                kernels_multi[f"mass{m}"] = ring_f
                kernels_multi[f"dx{m}"] = (dx * ring_f).astype(np.float64, copy=False)
                kernels_multi[f"dy{m}"] = (dy * ring_f).astype(np.float64, copy=False)

            # Precompute FFTs of kernels.
            kernel_ffts_c2 = {name: kernel_fft_centered(k, grid_size=grid_size) for name, k in kernels_c2.items()}
            kernel_ffts_multi = {name: kernel_fft_centered(k, grid_size=grid_size) for name, k in kernels_multi.items()}

            # Feature extraction per field.
            F_c2_by_field: list[np.ndarray] = []
            F_multi_by_field: list[np.ndarray] = []
            for field_id in range(n_fields):
                field = rho01_by_field[field_id]
                centers = centers_by_field[field_id]
                cx = centers[:, 0].astype(np.int64)
                cy = centers[:, 1].astype(np.int64)
                field_fft = np.fft.fftn(field)

                # C2 features: [M0, ring_sum..., ring_frac..., Dx,Dy,Dx_n,Dy_n]
                Fc2 = np.zeros((patches_per_field, dip_end_c2), dtype=np.float64)
                m0_grid = np.fft.ifftn(field_fft * kernel_ffts_c2["M0"]).real
                M0 = m0_grid[cx, cy].astype(np.float64, copy=False)
                denom = M0 + eps_norm
                Fc2[:, 0] = M0
                ring_sums = np.zeros((patches_per_field, n_radial_bins_c2), dtype=np.float64)
                for k in range(n_radial_bins_c2):
                    gk = np.fft.ifftn(field_fft * kernel_ffts_c2[f"ring{k}"]).real
                    ring_sums[:, k] = gk[cx, cy].astype(np.float64, copy=False)
                ring_fracs = ring_sums / denom[:, None]
                Fc2[:, 1 : 1 + n_radial_bins_c2] = ring_sums
                Fc2[:, 1 + n_radial_bins_c2 : 1 + 2 * n_radial_bins_c2] = ring_fracs
                dx_grid = np.fft.ifftn(field_fft * kernel_ffts_c2["Dx"]).real
                dy_grid = np.fft.ifftn(field_fft * kernel_ffts_c2["Dy"]).real
                Dx = dx_grid[cx, cy].astype(np.float64, copy=False)
                Dy = dy_grid[cx, cy].astype(np.float64, copy=False)
                Fc2[:, rings_end_c2 + 0] = Dx
                Fc2[:, rings_end_c2 + 1] = Dy
                Fc2[:, rings_end_c2 + 2] = Dx / denom
                Fc2[:, rings_end_c2 + 3] = Dy / denom
                assert_finite("F_c2", Fc2)

                # Multi-annulus: [Mass0,Dx0,Dy0, Mass1,Dx1,Dy1, ...]
                Fm = np.zeros((patches_per_field, 3 * int(n_annuli)), dtype=np.float64)
                for m in range(int(n_annuli)):
                    mass_grid = np.fft.ifftn(field_fft * kernel_ffts_multi[f"mass{m}"]).real
                    dxm_grid = np.fft.ifftn(field_fft * kernel_ffts_multi[f"dx{m}"]).real
                    dym_grid = np.fft.ifftn(field_fft * kernel_ffts_multi[f"dy{m}"]).real
                    base = 3 * m
                    Fm[:, base + 0] = mass_grid[cx, cy].astype(np.float64, copy=False)
                    Fm[:, base + 1] = dxm_grid[cx, cy].astype(np.float64, copy=False)
                    Fm[:, base + 2] = dym_grid[cx, cy].astype(np.float64, copy=False)
                assert_finite("F_multi", Fm)

                if field_id == 0 and validate_fft_n > 0:
                    n_check = min(int(validate_fft_n), patches_per_field)
                    for j in range(n_check):
                        cxi = int(cx[j])
                        cyi = int(cy[j])
                        patch_big = field[cxi - r_big : cxi + r_big + 1, cyi - r_big : cyi + r_big + 1]
                        # Direct compute by reusing nonlocal_annulus_moments for the outer totals (sanity),
                        # and explicit masks for ring features.
                        d0 = nonlocal_annulus_moments(
                            patch_big,
                            w_local=w_local,
                            n_radial_bins=n_radial_bins_c2,
                            include_dipole=True,
                            include_quadrupole=False,
                            eps=eps_norm,
                        )
                        if not np.isclose(Fc2[j, 0], float(d0["M0"]), rtol=validate_fft_rtol, atol=validate_fft_atol):
                            raise RuntimeError("E32 FFT mismatch on M0 (C2 validation)")
                        if not np.isclose(Fc2[j, rings_end_c2 + 0], float(d0["Dx"]), rtol=validate_fft_rtol, atol=validate_fft_atol):
                            raise RuntimeError("E32 FFT mismatch on Dx (C2 validation)")
                        if not np.isclose(Fc2[j, rings_end_c2 + 1], float(d0["Dy"]), rtol=validate_fft_rtol, atol=validate_fft_atol):
                            raise RuntimeError("E32 FFT mismatch on Dy (C2 validation)")

                        # Multi-annulus ring-by-ring direct checks for a couple of bins.
                        coords_l = np.arange(int(wb), dtype=np.float64) - float(r_big)
                        dx_l = coords_l[:, None]
                        dy_l = coords_l[None, :]
                        core_l = (np.abs(dx_l) <= float(r_local)) & (np.abs(dy_l) <= float(r_local))
                        outer_l = ~core_l
                        r_l = np.sqrt(dx_l * dx_l + dy_l * dy_l, dtype=np.float64)
                        for m in (0, int(n_annuli) - 1):
                            lo = edges[m]
                            hi = edges[m + 1]
                            if m == int(n_annuli) - 1:
                                ring = outer_l & (r_l >= lo) & (r_l <= hi)
                            else:
                                ring = outer_l & (r_l >= lo) & (r_l < hi)
                            mass_d = float(patch_big[ring].sum()) if ring.any() else 0.0
                            dx_d = float((patch_big * (dx_l * ring)).sum()) if ring.any() else 0.0
                            dy_d = float((patch_big * (dy_l * ring)).sum()) if ring.any() else 0.0
                            base = 3 * m
                            if not np.isclose(Fm[j, base + 0], mass_d, rtol=validate_fft_rtol, atol=validate_fft_atol):
                                raise RuntimeError("E32 FFT mismatch on multi mass validation")
                            if not np.isclose(Fm[j, base + 1], dx_d, rtol=validate_fft_rtol, atol=validate_fft_atol):
                                raise RuntimeError("E32 FFT mismatch on multi Dx validation")
                            if not np.isclose(Fm[j, base + 2], dy_d, rtol=validate_fft_rtol, atol=validate_fft_atol):
                                raise RuntimeError("E32 FFT mismatch on multi Dy validation")

                F_c2_by_field.append(Fc2)
                F_multi_by_field.append(Fm)

            # LOFO training/eval for this w_big.
            coef_raw_by_fold: list[np.ndarray] = []  # (2*M,2): rows [Dx0..DxM-1,Dy0..DyM-1]
            dp_multi_vs_c2: list[float] = []
            dr_multi_vs_c2: list[float] = []
            dp_multi_vs_b: list[float] = []
            dr_multi_vs_b: list[float] = []
            pvals_multi_vs_c2: list[float] = []
            pvals_multi_vs_b: list[float] = []

            perf_rows: list[dict[str, Any]] = []

            for test_field in range(n_fields):
                train_fields = [i for i in range(n_fields) if i != test_field]

                Btr_raw = np.concatenate([B_by_field[fid] for fid in train_fields], axis=0)
                Bte_raw = B_by_field[test_field]
                Fc2_tr_raw = np.concatenate([F_c2_by_field[fid] for fid in train_fields], axis=0)
                Fc2_te_raw = F_c2_by_field[test_field]
                Fm_tr_raw = np.concatenate([F_multi_by_field[fid] for fid in train_fields], axis=0)
                Fm_te_raw = F_multi_by_field[test_field]

                ytr = np.concatenate([y_low_by_field[fid] for fid in train_fields], axis=0)
                yte = y_low_by_field[test_field]

                # Standardize.
                Btr, Bte, _, _ = standardize(Btr_raw, Bte_raw)
                Fc2_tr, Fc2_te, _, _ = standardize(Fc2_tr_raw, Fc2_te_raw)
                Fm_tr, Fm_te, _, sd_m = standardize(Fm_tr_raw, Fm_te_raw)

                y_mu = ytr.mean(axis=0)
                yc = ytr - y_mu

                # B_low.
                A = Btr.T @ Btr
                U = Btr.T @ yc
                wB = solve_ridge(A, U)
                yB_te = Bte @ wB + y_mu
                pB = pearson_mean_2d(yte, yB_te)
                rrB = relrmse_mean_2d(yte, yB_te)

                # C2_low baseline (E31-style).
                Xtr2 = np.concatenate([Btr, Fc2_tr], axis=1)
                Xte2 = np.concatenate([Bte, Fc2_te], axis=1)
                wC2 = solve_ridge(Xtr2.T @ Xtr2, Xtr2.T @ yc)
                yC2_te = Xte2 @ wC2 + y_mu
                pC2 = pearson_mean_2d(yte, yC2_te)
                rrC2 = relrmse_mean_2d(yte, yC2_te)

                # Cmulti_low (B + multi-annulus features).
                Xtrm = np.concatenate([Btr, Fm_tr], axis=1)
                Xtem = np.concatenate([Bte, Fm_te], axis=1)
                wCm = solve_ridge(Xtrm.T @ Xtrm, Xtrm.T @ yc)
                yCm_te = Xtem @ wCm + y_mu
                pCm = pearson_mean_2d(yte, yCm_te)
                rrCm = relrmse_mean_2d(yte, yCm_te)

                # Optional dipole-only: keep only Dx_m, Dy_m (no Mass_m).
                dip_cols = np.concatenate([np.array([3 * m + 1, 3 * m + 2]) for m in range(int(n_annuli))]).astype(np.int64)
                Fd_tr = Fm_tr[:, dip_cols]
                Fd_te = Fm_te[:, dip_cols]
                Xtrd = np.concatenate([Btr, Fd_tr], axis=1)
                Xted = np.concatenate([Bte, Fd_te], axis=1)
                wCd = solve_ridge(Xtrd.T @ Xtrd, Xtrd.T @ yc)
                yCd_te = Xted @ wCd + y_mu
                pCd = pearson_mean_2d(yte, yCd_te)
                rrCd = relrmse_mean_2d(yte, yCd_te)

                # Deltas.
                dP_m_c2 = float(pCm - pC2)
                dR_m_c2 = float(rrCm - rrC2)
                dP_m_b = float(pCm - pB)
                dR_m_b = float(rrCm - rrB)
                dp_multi_vs_c2.append(dP_m_c2)
                dr_multi_vs_c2.append(dR_m_c2)
                dp_multi_vs_b.append(dP_m_b)
                dr_multi_vs_b.append(dR_m_b)

                # Store raw coeffs vs radius for Dx/Dy (effective on raw features: w_std/sd).
                w_std = wCm  # weights on standardized features (B+Fm)
                w_raw_multi = w_std[-Fm_tr.shape[1] :, :] / sd_m[:, None]
                coef_dx = np.zeros((int(n_annuli), 2), dtype=np.float64)
                coef_dy = np.zeros((int(n_annuli), 2), dtype=np.float64)
                for m in range(int(n_annuli)):
                    base = 3 * m
                    coef_dx[m, :] = w_raw_multi[base + 1, :]
                    coef_dy[m, :] = w_raw_multi[base + 2, :]
                coef_raw_by_fold.append(np.vstack([coef_dx, coef_dy]))

                # Placebo: shuffle multi features (TRAIN rows) and evaluate ΔPearson vs C2 and vs B.
                null_dP_c2: list[float] = []
                null_dP_b: list[float] = []
                for _ in range(n_perms):
                    pidx = rng_null.permutation(Fm_tr.shape[0])
                    Fp = Fm_tr[pidx]
                    Xtrp = np.concatenate([Btr, Fp], axis=1)
                    wp = solve_ridge(Xtrp.T @ Xtrp, Xtrp.T @ yc)
                    yp_te = Xtem @ wp + y_mu  # test uses real features
                    pP = pearson_mean_2d(yte, yp_te)
                    null_dP_c2.append(float(pP - pC2))
                    null_dP_b.append(float(pP - pB))

                null_c2 = np.asarray(null_dP_c2, dtype=np.float64)
                null_b = np.asarray(null_dP_b, dtype=np.float64)
                p_emp_c2 = float((np.sum(null_c2 >= dP_m_c2) + 1.0) / (float(n_perms) + 1.0))
                p_emp_b = float((np.sum(null_b >= dP_m_b) + 1.0) / (float(n_perms) + 1.0))
                pvals_multi_vs_c2.append(p_emp_c2)
                pvals_multi_vs_b.append(p_emp_b)

                perf_rows.append(
                    {
                        "field_id": int(test_field),
                        "pearson_B": float(pB),
                        "pearson_C2": float(pC2),
                        "pearson_Cmulti": float(pCm),
                        "pearson_Cmulti_dip": float(pCd),
                        "relRMSE_B": float(rrB),
                        "relRMSE_C2": float(rrC2),
                        "relRMSE_Cmulti": float(rrCm),
                        "relRMSE_Cmulti_dip": float(rrCd),
                        "deltaP_Cmulti_vs_C2": float(dP_m_c2),
                        "deltaP_Cmulti_vs_B": float(dP_m_b),
                        "p_emp_Cmulti_vs_C2": float(p_emp_c2),
                        "p_emp_Cmulti_vs_B": float(p_emp_b),
                    }
                )

            # Aggregate deltas and verdicts.
            dp_c2 = np.asarray(dp_multi_vs_c2, dtype=np.float64)
            dr_c2 = np.asarray(dr_multi_vs_c2, dtype=np.float64)
            dp_b = np.asarray(dp_multi_vs_b, dtype=np.float64)
            dr_b = np.asarray(dr_multi_vs_b, dtype=np.float64)
            npos_c2 = int(np.sum(dp_c2 > 0))
            npos_b = int(np.sum(dp_b > 0))
            fp_c2 = fisher_p(pvals_multi_vs_c2)
            fp_b = fisher_p(pvals_multi_vs_b)
            verdict_c2 = (float(dp_c2.mean()) > 0.0) and (float(dr_c2.mean()) < 0.0) and (fp_c2 < 0.05) and (npos_c2 >= 7)
            verdict_b = (float(dp_b.mean()) > 0.0) and (float(dr_b.mean()) < 0.0) and (fp_b < 0.05) and (npos_b >= 7)

            # Coef plot (mean±std across folds) for Dx/Dy.
            coef_stack = np.stack(coef_raw_by_fold, axis=0)  # (n_fields, 2M,2)
            coef_dx = coef_stack[:, : int(n_annuli), :]
            coef_dy = coef_stack[:, int(n_annuli) :, :]
            mean_dx = coef_dx.mean(axis=0)
            std_dx = coef_dx.std(axis=0, ddof=1) if coef_dx.shape[0] > 1 else np.zeros_like(mean_dx)
            mean_dy = coef_dy.mean(axis=0)
            std_dy = coef_dy.std(axis=0, ddof=1) if coef_dy.shape[0] > 1 else np.zeros_like(mean_dy)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
            ax = axes[0]
            ax.plot(radii, mean_dx[:, 0], "-o", label="gx <- Dx")
            ax.fill_between(radii, mean_dx[:, 0] - std_dx[:, 0], mean_dx[:, 0] + std_dx[:, 0], alpha=0.2)
            ax.plot(radii, mean_dy[:, 0], "-o", label="gx <- Dy")
            ax.fill_between(radii, mean_dy[:, 0] - std_dy[:, 0], mean_dy[:, 0] + std_dy[:, 0], alpha=0.2)
            ax.set_title("Coefficients for gx_low")
            ax.set_xlabel("radius (bin center)")
            ax.set_ylabel("coef on raw feature")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            ax = axes[1]
            ax.plot(radii, mean_dx[:, 1], "-o", label="gy <- Dx")
            ax.fill_between(radii, mean_dx[:, 1] - std_dx[:, 1], mean_dx[:, 1] + std_dx[:, 1], alpha=0.2)
            ax.plot(radii, mean_dy[:, 1], "-o", label="gy <- Dy")
            ax.fill_between(radii, mean_dy[:, 1] - std_dy[:, 1], mean_dy[:, 1] + std_dy[:, 1], alpha=0.2)
            ax.set_title("Coefficients for gy_low")
            ax.set_xlabel("radius (bin center)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            fig.suptitle(f"E32 multi-annulus weights (w_big={wb}, n_annuli={n_annuli}, mode={ring_bin_mode})")
            fig.tight_layout()
            coef_png = paths.run_dir / f"coeffs_vs_radius_wbig{wb}.png"
            fig.savefig(coef_png, dpi=150)
            plt.close(fig)
            coeff_plots[f"wbig{wb}"] = str(coef_png)

            # Add summary row.
            pBv = np.asarray([r0["pearson_B"] for r0 in perf_rows], dtype=np.float64)
            pC2v = np.asarray([r0["pearson_C2"] for r0 in perf_rows], dtype=np.float64)
            pCmv = np.asarray([r0["pearson_Cmulti"] for r0 in perf_rows], dtype=np.float64)
            pCdv = np.asarray([r0["pearson_Cmulti_dip"] for r0 in perf_rows], dtype=np.float64)
            rBv = np.asarray([r0["relRMSE_B"] for r0 in perf_rows], dtype=np.float64)
            rC2v = np.asarray([r0["relRMSE_C2"] for r0 in perf_rows], dtype=np.float64)
            rCmv = np.asarray([r0["relRMSE_Cmulti"] for r0 in perf_rows], dtype=np.float64)
            rCdv = np.asarray([r0["relRMSE_Cmulti_dip"] for r0 in perf_rows], dtype=np.float64)

            rows_summary.append(
                {
                    "w_big": int(wb),
                    "n_annuli": int(n_annuli),
                    "ring_bin_mode": ring_bin_mode,
                    "pearson_B_mean": float(pBv.mean()),
                    "pearson_B_std": float(pBv.std(ddof=1)) if len(pBv) > 1 else 0.0,
                    "pearson_C2_mean": float(pC2v.mean()),
                    "pearson_C2_std": float(pC2v.std(ddof=1)) if len(pC2v) > 1 else 0.0,
                    "pearson_Cmulti_mean": float(pCmv.mean()),
                    "pearson_Cmulti_std": float(pCmv.std(ddof=1)) if len(pCmv) > 1 else 0.0,
                    "pearson_Cmulti_dip_mean": float(pCdv.mean()),
                    "pearson_Cmulti_dip_std": float(pCdv.std(ddof=1)) if len(pCdv) > 1 else 0.0,
                    "relRMSE_B_mean": float(rBv.mean()),
                    "relRMSE_B_std": float(rBv.std(ddof=1)) if len(rBv) > 1 else 0.0,
                    "relRMSE_C2_mean": float(rC2v.mean()),
                    "relRMSE_C2_std": float(rC2v.std(ddof=1)) if len(rC2v) > 1 else 0.0,
                    "relRMSE_Cmulti_mean": float(rCmv.mean()),
                    "relRMSE_Cmulti_std": float(rCmv.std(ddof=1)) if len(rCmv) > 1 else 0.0,
                    "relRMSE_Cmulti_dip_mean": float(rCdv.mean()),
                    "relRMSE_Cmulti_dip_std": float(rCdv.std(ddof=1)) if len(rCdv) > 1 else 0.0,
                    "deltaP_Cmulti_vs_C2_mean": float(dp_c2.mean()),
                    "deltaP_Cmulti_vs_C2_std": float(dp_c2.std(ddof=1)) if len(dp_c2) > 1 else 0.0,
                    "deltaR_Cmulti_vs_C2_mean": float(dr_c2.mean()),
                    "deltaR_Cmulti_vs_C2_std": float(dr_c2.std(ddof=1)) if len(dr_c2) > 1 else 0.0,
                    "fisher_p_Cmulti_vs_C2": float(fp_c2),
                    "n_pos_Cmulti_vs_C2": int(npos_c2),
                    "verdict_Cmulti_vs_C2": bool(verdict_c2),
                    "deltaP_Cmulti_vs_B_mean": float(dp_b.mean()),
                    "deltaP_Cmulti_vs_B_std": float(dp_b.std(ddof=1)) if len(dp_b) > 1 else 0.0,
                    "deltaR_Cmulti_vs_B_mean": float(dr_b.mean()),
                    "deltaR_Cmulti_vs_B_std": float(dr_b.std(ddof=1)) if len(dr_b) > 1 else 0.0,
                    "fisher_p_Cmulti_vs_B": float(fp_b),
                    "n_pos_Cmulti_vs_B": int(npos_b),
                    "verdict_Cmulti_vs_B": bool(verdict_b),
                }
            )

            # Save per-fold CSV.
            header = list(perf_rows[0].keys()) if perf_rows else []
            write_csv(
                paths.run_dir / f"lofo_by_field_wbig{wb}_multiannulus.csv",
                header,
                [[r0[h] for h in header] for r0 in perf_rows],
            )

        md_rows: list[dict[str, str]] = []
        for r0 in rows_summary:
            md_rows.append(
                {
                    "w_big": str(int(r0["w_big"])),
                    "Pearson_B mean±std": f"{float(r0['pearson_B_mean']):.4f} ± {float(r0['pearson_B_std']):.4f}",
                    "relRMSE_B mean±std": f"{float(r0['relRMSE_B_mean']):.4f} ± {float(r0['relRMSE_B_std']):.4f}",
                    "Pearson_C2 mean±std": f"{float(r0['pearson_C2_mean']):.4f} ± {float(r0['pearson_C2_std']):.4f}",
                    "relRMSE_C2 mean±std": f"{float(r0['relRMSE_C2_mean']):.4f} ± {float(r0['relRMSE_C2_std']):.4f}",
                    "Pearson_Cmulti mean±std": f"{float(r0['pearson_Cmulti_mean']):.4f} ± {float(r0['pearson_Cmulti_std']):.4f}",
                    "relRMSE_Cmulti mean±std": f"{float(r0['relRMSE_Cmulti_mean']):.4f} ± {float(r0['relRMSE_Cmulti_std']):.4f}",
                    "Pearson_Cmulti_dip mean±std": f"{float(r0['pearson_Cmulti_dip_mean']):.4f} ± {float(r0['pearson_Cmulti_dip_std']):.4f}",
                    "relRMSE_Cmulti_dip mean±std": f"{float(r0['relRMSE_Cmulti_dip_mean']):.4f} ± {float(r0['relRMSE_Cmulti_dip_std']):.4f}",
                    "ΔP(Cmulti-C2) mean±std": f"{float(r0['deltaP_Cmulti_vs_C2_mean']):.4f} ± {float(r0['deltaP_Cmulti_vs_C2_std']):.4f}",
                    "ΔR(Cmulti-C2) mean±std": f"{float(r0['deltaR_Cmulti_vs_C2_mean']):.4f} ± {float(r0['deltaR_Cmulti_vs_C2_std']):.4f}",
                    "Fisher p (ΔP, vs C2)": fmt(float(r0["fisher_p_Cmulti_vs_C2"])),
                    "#folds ΔP>0 (vs C2)": f"{int(r0['n_pos_Cmulti_vs_C2'])}/{n_fields}",
                    "ΔP(Cmulti-B) mean±std": f"{float(r0['deltaP_Cmulti_vs_B_mean']):.4f} ± {float(r0['deltaP_Cmulti_vs_B_std']):.4f}",
                    "ΔR(Cmulti-B) mean±std": f"{float(r0['deltaR_Cmulti_vs_B_mean']):.4f} ± {float(r0['deltaR_Cmulti_vs_B_std']):.4f}",
                    "Fisher p (ΔP, vs B)": fmt(float(r0["fisher_p_Cmulti_vs_B"])),
                    "#folds ΔP>0 (vs B)": f"{int(r0['n_pos_Cmulti_vs_B'])}/{n_fields}",
                    "verdict (vs C2)": "PASS" if bool(r0["verdict_Cmulti_vs_C2"]) else "FAIL",
                }
            )

        coef_lines = "\n".join([f"- `{k}`: `{v}`" for k, v in coeff_plots.items()])
        summary_md = (
            "# E32 — Multi-annulus low-k estimator (LOFO)\n\n"
            f"- run: `{paths.run_dir}`\n"
            f"- grid_size={grid_size}, alpha={alpha}, n_fields={n_fields}, patches_per_field={patches_per_field}\n"
            f"- k0_frac={k0_frac}, w_local={w_local}, w_bigs={w_bigs}\n"
            f"- n_annuli={n_annuli}, ring_bin_mode={ring_bin_mode}\n"
            f"- ridge_alpha={ridge_alpha}, perms={n_perms}\n"
            f"- band-split recon: rel_err_gx_mean={float(np.mean(rel_recon_gx)):.3e}, rel_err_gy_mean={float(np.mean(rel_recon_gy)):.3e}\n\n"
            "## Aggregated (targets: gx_low, gy_low)\n\n"
            + md_table(
                md_rows,
                [
                    "w_big",
                    "Pearson_B mean±std",
                    "relRMSE_B mean±std",
                    "Pearson_C2 mean±std",
                    "relRMSE_C2 mean±std",
                    "Pearson_Cmulti mean±std",
                    "relRMSE_Cmulti mean±std",
                    "Pearson_Cmulti_dip mean±std",
                    "relRMSE_Cmulti_dip mean±std",
                    "ΔP(Cmulti-C2) mean±std",
                    "ΔR(Cmulti-C2) mean±std",
                    "Fisher p (ΔP, vs C2)",
                    "#folds ΔP>0 (vs C2)",
                    "ΔP(Cmulti-B) mean±std",
                    "ΔR(Cmulti-B) mean±std",
                    "Fisher p (ΔP, vs B)",
                    "#folds ΔP>0 (vs B)",
                    "verdict (vs C2)",
                ],
            )
            + "\n\n## Coefficient plots\n\n"
            + coef_lines
            + "\n"
        )
        (paths.run_dir / "summary_e32_multiannulus_lowk.md").write_text(summary_md, encoding="utf-8")

        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "grid_size": grid_size,
                "alpha": alpha,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "k0_frac": k0_frac,
                "w_local": w_local,
                "w_bigs": [int(w) for w in w_bigs],
                "n_annuli": n_annuli,
                "ring_bin_mode": ring_bin_mode,
                "ridge_alpha": ridge_alpha,
                "n_perms": n_perms,
                "rows_summary": rows_summary,
                "coef_plots": coeff_plots,
                "split_recon": {
                    "rel_err_gx_mean": float(np.mean(rel_recon_gx)),
                    "rel_err_gy_mean": float(np.mean(rel_recon_gy)),
                },
            },
        )
        return paths

    if experiment == "e33":
        # E33 — Low-k: push multi-annulus dipole toward the pixels-only upper bound (LOFO).
        #
        # Sweep over (w_big, M):
        # - Feature models: multi-annulus dipole-only and dipole+quad, with/without local_B.
        # - Baseline: C2 single-annulus dipole (E32-style outer-only annulus) for each w_big.
        # - Pixels ceiling: P_low(w_big) pixels-only SGD on rho patches of size w_big.
        from numpy.lib.stride_tricks import sliding_window_view
        from scipy import ndimage, stats
        from sklearn.linear_model import SGDRegressor

        import matplotlib.pyplot as plt

        grid_size = int(cfg.get("grid_size", 256))
        alpha = float(cfg.get("alpha", 2.0))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 10_000))
        k0_frac = float(cfg.get("k0_frac", 0.15))
        w_local = _require_odd("w_local", int(cfg.get("w_local", 65)))

        w_bigs = [_require_odd("w_big", int(w)) for w in cfg.get("w_bigs", [129, 193])]
        M_list = [int(m) for m in cfg.get("M_list", [8, 16])]
        ring_bin_mode = str(cfg.get("ring_bin_mode", "uniform_r")).lower()

        ridge_alpha = float(cfg.get("ridge_alpha", 1.0))
        n_perms = int(cfg.get("n_perms", 200))

        # Pixels-only ceiling.
        pixel_epochs = int(cfg.get("pixel_epochs", 1))
        pixel_batch_size = int(cfg.get("pixel_batch_size", 256))
        pixel_alpha = float(cfg.get("pixel_alpha", 1e-4))
        pixel_eta0 = float(cfg.get("pixel_eta0", 0.01))
        pixel_power_t = float(cfg.get("pixel_power_t", 0.25))

        validate_fft_n = int(cfg.get("validate_fft_n", 1))
        validate_fft_rtol = float(cfg.get("validate_fft_rtol", 1e-6))
        validate_fft_atol = float(cfg.get("validate_fft_atol", 1e-6))

        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if n_fields < 2:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        if not (0.0 < float(k0_frac) < 0.5):
            raise ValueError("k0_frac must be in (0,0.5)")
        if not w_bigs:
            raise ValueError("w_bigs must be non-empty")
        if any(wb < w_local for wb in w_bigs):
            raise ValueError("All w_bigs must be >= w_local")
        if any(wb > grid_size for wb in w_bigs):
            raise ValueError("All w_bigs must be <= grid_size")
        if not M_list or any(m <= 0 for m in M_list):
            raise ValueError("M_list must contain positive ints")
        if ring_bin_mode not in {"uniform_r", "equal_area"}:
            raise ValueError("ring_bin_mode must be 'uniform_r' or 'equal_area'")
        if ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be > 0")
        if n_perms <= 0:
            raise ValueError("n_perms must be > 0")
        if pixel_epochs <= 0:
            raise ValueError("pixel_epochs must be > 0")
        if pixel_batch_size <= 0:
            raise ValueError("pixel_batch_size must be > 0")
        if pixel_alpha <= 0:
            raise ValueError("pixel_alpha must be > 0")
        if pixel_eta0 <= 0:
            raise ValueError("pixel_eta0 must be > 0")
        if validate_fft_n < 0:
            raise ValueError("validate_fft_n must be >= 0")
        if validate_fft_rtol < 0 or validate_fft_atol < 0:
            raise ValueError("validate_fft_rtol/atol must be >= 0")

        rng_null = np.random.default_rng(int(placebo_seed) + 333_333)
        rng_pixel = np.random.default_rng(int(placebo_seed) + 344_444)

        def fisher_p(ps: list[float]) -> float:
            if not ps:
                return float("nan")
            eps = 1.0 / float(n_perms + 1)
            ps2 = [min(1.0, max(float(p), eps)) for p in ps]
            stat = -2.0 * float(np.sum(np.log(ps2)))
            return float(stats.chi2.sf(stat, 2 * len(ps2)))

        def fmt(x: float) -> str:
            if not np.isfinite(x):
                return "nan"
            if abs(float(x)) < 1e-3:
                return f"{x:.3e}"
            return f"{x:.4f}"

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r0 in rows:
                out.append("| " + " | ".join(r0.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        def safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            am = a - float(a.mean())
            bm = b - float(b.mean())
            denom = float(np.linalg.norm(am) * np.linalg.norm(bm)) + 1e-12
            return float((am @ bm) / denom)

        def relrmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
            e = y_pred - y_true
            rmse = float(np.sqrt(np.mean(e * e)))
            sd = float(np.std(y_true))
            return rmse / (sd + 1e-12)

        def pearson_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            px = safe_corr_1d(y_true[:, 0], y_pred[:, 0])
            py = safe_corr_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (px + py)

        def relrmse_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            rx = relrmse_1d(y_true[:, 0], y_pred[:, 0])
            ry = relrmse_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (rx + ry)

        def solve_ridge(XtX: np.ndarray, Xty: np.ndarray) -> np.ndarray:
            d = int(XtX.shape[0])
            I = np.eye(d, dtype=np.float64)
            try:
                return np.linalg.solve(XtX + ridge_alpha * I, Xty)
            except np.linalg.LinAlgError:
                w, *_ = np.linalg.lstsq(XtX + ridge_alpha * I, Xty, rcond=None)
                return w

        def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            mu = train.mean(axis=0)
            sd = train.std(axis=0)
            sd = np.where(sd > 0, sd, 1.0)
            return (train - mu) / sd, (test - mu) / sd, mu, sd

        def kernel_fft_centered(kernel: np.ndarray, *, grid_size: int) -> np.ndarray:
            kernel = np.asarray(kernel, dtype=np.float64)
            if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
                raise ValueError(f"kernel must be square 2D, got {kernel.shape}")
            wb = int(kernel.shape[0])
            if (wb % 2) == 0:
                raise ValueError(f"kernel size must be odd, got {wb}")
            if wb > grid_size:
                raise ValueError(f"kernel size {wb} exceeds grid_size {grid_size}")
            r = wb // 2
            kr = np.flip(kernel, axis=(0, 1))
            full = np.zeros((grid_size, grid_size), dtype=np.float64)
            cx = grid_size // 2
            cy = grid_size // 2
            full[cx - r : cx + r + 1, cy - r : cy + r + 1] = kr
            full0 = np.fft.ifftshift(full)
            return np.fft.fftn(full0)

        def annulus_edges(r_min: float, r_max: float, n_bins: int) -> np.ndarray:
            if ring_bin_mode == "uniform_r":
                return np.linspace(r_min, r_max + 1e-9, n_bins + 1, dtype=np.float64)
            rsq = np.linspace(r_min * r_min, r_max * r_max + 1e-9, n_bins + 1, dtype=np.float64)
            return np.sqrt(rsq, dtype=np.float64)

        # Shared fields and band-split cache.
        rho01_by_field: list[np.ndarray] = []
        rhoz_by_field: list[np.ndarray] = []
        split_cache: list[BandSplit2D] = []
        rel_recon_gx: list[float] = []
        rel_recon_gy: list[float] = []
        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
            split = band_split_poisson_2d(rho01, k0_frac=float(k0_frac))
            rho01_by_field.append(rho01.astype(np.float64, copy=False))
            rhoz_by_field.append(_zscore_field(rho01).astype(np.float64, copy=False))
            split_cache.append(split)
            rel_recon_gx.append(float(split.rel_err_gx))
            rel_recon_gy.append(float(split.rel_err_gy))

        def sample_low_at_centers(split: BandSplit2D, centers: np.ndarray) -> np.ndarray:
            cx = centers[:, 0].astype(np.int64)
            cy = centers[:, 1].astype(np.int64)
            return np.column_stack([split.low.gx[cx, cy], split.low.gy[cx, cy]]).astype(np.float64, copy=False)

        def build_local_B(field: np.ndarray, centers: np.ndarray) -> np.ndarray:
            rloc = w_local // 2
            cx = centers[:, 0]
            cy = centers[:, 1]
            pref1 = _prefix_sum_2d(field)
            pref2 = _prefix_sum_2d(field * field)
            x0 = (cx - rloc).astype(np.int64)
            x1 = (cx + rloc + 1).astype(np.int64)
            y0 = (cy - rloc).astype(np.int64)
            y1 = (cy + rloc + 1).astype(np.int64)
            mass = _box_sum_2d(pref1, x0, x1, y0, y1)
            nvox = float(w_local * w_local)
            mean = mass / nvox
            sumsq = _box_sum_2d(pref2, x0, x1, y0, y1)
            var = np.maximum(0.0, sumsq / nvox - mean * mean)
            mass2 = mass * mass
            max_grid = ndimage.maximum_filter(field, size=w_local, mode="constant", cval=-np.inf)
            mx = max_grid[cx, cy]
            gx_f, gy_f = np.gradient(field)
            egrid = gx_f * gx_f + gy_f * gy_f
            eavg = ndimage.uniform_filter(egrid, size=w_local, mode="constant", cval=0.0)
            ge = eavg[cx, cy]
            B = np.column_stack([mass, mass2, var, mx, ge]).astype(np.float64, copy=False)
            assert_finite("B_local", B)
            return B

        # Pixels-only ceiling (vector), streamed per fold.
        def fit_pixel_model_vec(
            train_fields: list[int],
            fields_z: list[np.ndarray],
            centers_by_field: list[np.ndarray],
            w: int,
            y_by_field: list[np.ndarray],
            *,
            random_state_base: int,
        ) -> tuple[tuple[SGDRegressor, SGDRegressor], np.ndarray, np.ndarray]:
            y_all = np.concatenate([y_by_field[fid] for fid in train_fields], axis=0).astype(np.float64, copy=False)
            y_mu = y_all.mean(axis=0)
            y_sd = y_all.std(axis=0)
            y_sd = np.where((~np.isfinite(y_sd)) | (y_sd <= 0), 1.0, y_sd)

            eta0_eff = float(pixel_eta0) / float(w)
            reg_gx = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(pixel_alpha),
                learning_rate="invscaling",
                eta0=float(eta0_eff),
                power_t=float(pixel_power_t),
                max_iter=1,
                tol=None,
                fit_intercept=True,
                average=True,
                random_state=int(random_state_base) + 0,
            )
            reg_gy = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(pixel_alpha),
                learning_rate="invscaling",
                eta0=float(eta0_eff),
                power_t=float(pixel_power_t),
                max_iter=1,
                tol=None,
                fit_intercept=True,
                average=True,
                random_state=int(random_state_base) + 1,
            )

            r = w // 2
            for _epoch in range(int(pixel_epochs)):
                for fid in train_fields:
                    field = fields_z[fid]
                    centers = centers_by_field[fid]
                    y = y_by_field[fid]
                    win = sliding_window_view(field, (w, w))
                    perm = rng_pixel.permutation(centers.shape[0])
                    for start in range(0, centers.shape[0], pixel_batch_size):
                        idx = perm[start : start + pixel_batch_size]
                        cx = centers[idx, 0].astype(np.int64)
                        cy = centers[idx, 1].astype(np.int64)
                        Xb = win[cx - r, cy - r].reshape(len(idx), -1).astype(np.float32, copy=False)
                        reg_gx.partial_fit(Xb, (y[idx, 0] - y_mu[0]) / y_sd[0])
                        reg_gy.partial_fit(Xb, (y[idx, 1] - y_mu[1]) / y_sd[1])
            return (reg_gx, reg_gy), y_mu, y_sd

        def predict_pixel_vec(
            regs: tuple[SGDRegressor, SGDRegressor],
            y_mu: np.ndarray,
            y_sd: np.ndarray,
            field_z: np.ndarray,
            centers: np.ndarray,
            w: int,
        ) -> np.ndarray:
            reg_gx, reg_gy = regs
            r = w // 2
            win = sliding_window_view(field_z, (w, w))
            out = np.empty((centers.shape[0], 2), dtype=np.float64)
            for start in range(0, centers.shape[0], pixel_batch_size):
                idx = slice(start, min(centers.shape[0], start + pixel_batch_size))
                cx = centers[idx, 0].astype(np.int64)
                cy = centers[idx, 1].astype(np.int64)
                Xb = win[cx - r, cy - r].reshape(len(cx), -1).astype(np.float32, copy=False)
                out[idx, 0] = reg_gx.predict(Xb) * y_sd[0] + y_mu[0]
                out[idx, 1] = reg_gy.predict(Xb) * y_sd[1] + y_mu[1]
            return out

        # Compute everything per w_big (centers differ).
        rows_summary: list[dict[str, Any]] = []
        coeff_plots: dict[str, str] = {}
        pixel_rows: list[dict[str, Any]] = []

        for wb in w_bigs:
            r_big = int(wb) // 2
            rloc = w_local // 2

            centers_by_field: list[np.ndarray] = []
            for field_id in range(n_fields):
                rng_cent = np.random.default_rng(seed + 555_555 + 10_000 * wb + 1_000 * field_id)
                cx = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
                cy = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
                centers_by_field.append(np.column_stack([cx, cy]).astype(np.int64, copy=False))

            y_low_by_field = [sample_low_at_centers(split_cache[fid], centers_by_field[fid]) for fid in range(n_fields)]
            B_by_field = [build_local_B(rho01_by_field[fid], centers_by_field[fid]) for fid in range(n_fields)]

            # C2 baseline features (outer-only annulus between r_local..r_big).
            n_radial_bins_c2 = int(cfg.get("n_radial_bins_c2", 6))
            edges_c2 = annulus_edges(float(rloc), float(r_big), n_radial_bins_c2)
            coords = np.arange(int(wb), dtype=np.float64) - float(r_big)
            dx = coords[:, None]
            dy = coords[None, :]
            core = (np.abs(dx) <= float(rloc)) & (np.abs(dy) <= float(rloc))
            outer = ~core
            rgrid = np.sqrt(dx * dx + dy * dy, dtype=np.float64)

            kernels_c2: dict[str, np.ndarray] = {"M0": outer.astype(np.float64)}
            for k in range(n_radial_bins_c2):
                lo = edges_c2[k]
                hi = edges_c2[k + 1]
                if k == n_radial_bins_c2 - 1:
                    ring = outer & (rgrid >= lo) & (rgrid <= hi)
                else:
                    ring = outer & (rgrid >= lo) & (rgrid < hi)
                kernels_c2[f"ring{k}"] = ring.astype(np.float64)
            kernels_c2["Dx"] = (dx * outer).astype(np.float64, copy=False)
            kernels_c2["Dy"] = (dy * outer).astype(np.float64, copy=False)

            kernel_ffts_c2 = {name: kernel_fft_centered(k, grid_size=grid_size) for name, k in kernels_c2.items()}
            rings_end_c2 = 1 + 2 * n_radial_bins_c2
            dip_end_c2 = rings_end_c2 + 4

            Fc2_by_field: list[np.ndarray] = []
            for field_id in range(n_fields):
                field = rho01_by_field[field_id]
                centers = centers_by_field[field_id]
                cx = centers[:, 0].astype(np.int64)
                cy = centers[:, 1].astype(np.int64)
                field_fft = np.fft.fftn(field)
                Fc2 = np.zeros((patches_per_field, dip_end_c2), dtype=np.float64)
                m0_grid = np.fft.ifftn(field_fft * kernel_ffts_c2["M0"]).real
                M0 = m0_grid[cx, cy].astype(np.float64, copy=False)
                denom = M0 + 1e-12
                Fc2[:, 0] = M0
                ring_sums = np.zeros((patches_per_field, n_radial_bins_c2), dtype=np.float64)
                for k in range(n_radial_bins_c2):
                    gk = np.fft.ifftn(field_fft * kernel_ffts_c2[f"ring{k}"]).real
                    ring_sums[:, k] = gk[cx, cy].astype(np.float64, copy=False)
                ring_fracs = ring_sums / denom[:, None]
                Fc2[:, 1 : 1 + n_radial_bins_c2] = ring_sums
                Fc2[:, 1 + n_radial_bins_c2 : 1 + 2 * n_radial_bins_c2] = ring_fracs
                dx_grid = np.fft.ifftn(field_fft * kernel_ffts_c2["Dx"]).real
                dy_grid = np.fft.ifftn(field_fft * kernel_ffts_c2["Dy"]).real
                Dx_all = dx_grid[cx, cy].astype(np.float64, copy=False)
                Dy_all = dy_grid[cx, cy].astype(np.float64, copy=False)
                Fc2[:, rings_end_c2 + 0] = Dx_all
                Fc2[:, rings_end_c2 + 1] = Dy_all
                Fc2[:, rings_end_c2 + 2] = Dx_all / denom
                Fc2[:, rings_end_c2 + 3] = Dy_all / denom
                assert_finite("Fc2", Fc2)
                Fc2_by_field.append(Fc2)

            # Pixels-only ceiling P_low(w_big).
            pixel_fold_rows: list[dict[str, Any]] = []
            for test_field in range(n_fields):
                train_fields = [i for i in range(n_fields) if i != test_field]
                regs, ymu, ysd = fit_pixel_model_vec(
                    train_fields,
                    rhoz_by_field,
                    centers_by_field,
                    int(wb),
                    y_low_by_field,
                    random_state_base=int(placebo_seed) + 9000 + 10_000 * wb + 100 * test_field,
                )
                y_pred = predict_pixel_vec(regs, ymu, ysd, rhoz_by_field[test_field], centers_by_field[test_field], int(wb))
                p = pearson_mean_2d(y_low_by_field[test_field], y_pred)
                rr = relrmse_mean_2d(y_low_by_field[test_field], y_pred)
                pixel_fold_rows.append({"field_id": int(test_field), "pearson": float(p), "relRMSE": float(rr)})
            p_mean = float(np.mean([r0["pearson"] for r0 in pixel_fold_rows]))
            p_std = float(np.std([r0["pearson"] for r0 in pixel_fold_rows], ddof=1))
            r_mean = float(np.mean([r0["relRMSE"] for r0 in pixel_fold_rows]))
            r_std = float(np.std([r0["relRMSE"] for r0 in pixel_fold_rows], ddof=1))
            pixel_rows.append(
                {
                    "w_big": int(wb),
                    "model": "P_low(pixels)",
                    "pearson_mean": p_mean,
                    "pearson_std": p_std,
                    "relRMSE_mean": r_mean,
                    "relRMSE_std": r_std,
                }
            )
            write_csv(
                paths.run_dir / f"lofo_by_field_pixels_low_wbig{wb}.csv",
                list(pixel_fold_rows[0].keys()),
                [[r0[k] for k in pixel_fold_rows[0].keys()] for r0 in pixel_fold_rows],
            )

            # Sweep over M and build feature blocks for each.
            for M in M_list:
                edges = annulus_edges(0.0, float(r_big), int(M))
                radii = 0.5 * (edges[:-1] + edges[1:])

                # Multi-annulus kernels on full disk (no core exclusion).
                kernels_dip: dict[str, np.ndarray] = {}
                kernels_quad: dict[str, np.ndarray] = {}
                for m in range(int(M)):
                    lo = edges[m]
                    hi = edges[m + 1]
                    if m == int(M) - 1:
                        ring = (rgrid >= lo) & (rgrid <= hi)
                    else:
                        ring = (rgrid >= lo) & (rgrid < hi)
                    ring_f = ring.astype(np.float64)
                    kernels_dip[f"dx{m}"] = (dx * ring_f).astype(np.float64, copy=False)
                    kernels_dip[f"dy{m}"] = (dy * ring_f).astype(np.float64, copy=False)
                    kernels_quad[f"q1{m}"] = ((dx * dx - dy * dy) * ring_f).astype(np.float64, copy=False)
                    kernels_quad[f"q2{m}"] = ((2.0 * dx * dy) * ring_f).astype(np.float64, copy=False)

                fft_dip = {name: kernel_fft_centered(k, grid_size=grid_size) for name, k in kernels_dip.items()}
                fft_quad = {name: kernel_fft_centered(k, grid_size=grid_size) for name, k in kernels_quad.items()}

                F_dip_by_field: list[np.ndarray] = []
                F_dipquad_by_field: list[np.ndarray] = []
                for field_id in range(n_fields):
                    field = rho01_by_field[field_id]
                    centers = centers_by_field[field_id]
                    cx = centers[:, 0].astype(np.int64)
                    cy = centers[:, 1].astype(np.int64)
                    field_fft = np.fft.fftn(field)

                    Fd = np.zeros((patches_per_field, 2 * int(M)), dtype=np.float64)
                    Fq = np.zeros((patches_per_field, 4 * int(M)), dtype=np.float64)
                    for m in range(int(M)):
                        dxm_grid = np.fft.ifftn(field_fft * fft_dip[f"dx{m}"]).real
                        dym_grid = np.fft.ifftn(field_fft * fft_dip[f"dy{m}"]).real
                        q1m_grid = np.fft.ifftn(field_fft * fft_quad[f"q1{m}"]).real
                        q2m_grid = np.fft.ifftn(field_fft * fft_quad[f"q2{m}"]).real
                        Fd[:, m] = dxm_grid[cx, cy].astype(np.float64, copy=False)
                        Fd[:, int(M) + m] = dym_grid[cx, cy].astype(np.float64, copy=False)
                        Fq[:, m] = dxm_grid[cx, cy].astype(np.float64, copy=False)
                        Fq[:, int(M) + m] = dym_grid[cx, cy].astype(np.float64, copy=False)
                        Fq[:, 2 * int(M) + m] = q1m_grid[cx, cy].astype(np.float64, copy=False)
                        Fq[:, 3 * int(M) + m] = q2m_grid[cx, cy].astype(np.float64, copy=False)

                    assert_finite("Fd", Fd)
                    assert_finite("Fq", Fq)

                    # Light validation by direct computation on one sample for first/last ring.
                    if field_id == 0 and validate_fft_n > 0:
                        n_check = min(int(validate_fft_n), patches_per_field)
                        for j in range(n_check):
                            cxi = int(cx[j])
                            cyi = int(cy[j])
                            patch = field[cxi - r_big : cxi + r_big + 1, cyi - r_big : cyi + r_big + 1]
                            coords_l = np.arange(int(wb), dtype=np.float64) - float(r_big)
                            dx_l = coords_l[:, None]
                            dy_l = coords_l[None, :]
                            r_l = np.sqrt(dx_l * dx_l + dy_l * dy_l, dtype=np.float64)
                            for mm in (0, int(M) - 1):
                                lo = edges[mm]
                                hi = edges[mm + 1]
                                if mm == int(M) - 1:
                                    ring = (r_l >= lo) & (r_l <= hi)
                                else:
                                    ring = (r_l >= lo) & (r_l < hi)
                                dx_d = float((patch * (dx_l * ring)).sum()) if ring.any() else 0.0
                                dy_d = float((patch * (dy_l * ring)).sum()) if ring.any() else 0.0
                                q1_d = float((patch * ((dx_l * dx_l - dy_l * dy_l) * ring)).sum()) if ring.any() else 0.0
                                q2_d = float((patch * ((2.0 * dx_l * dy_l) * ring)).sum()) if ring.any() else 0.0
                                if not np.isclose(Fd[j, mm], dx_d, rtol=validate_fft_rtol, atol=validate_fft_atol):
                                    raise RuntimeError("E33 FFT mismatch on Dx ring validation")
                                if not np.isclose(Fd[j, int(M) + mm], dy_d, rtol=validate_fft_rtol, atol=validate_fft_atol):
                                    raise RuntimeError("E33 FFT mismatch on Dy ring validation")
                                if not np.isclose(Fq[j, 2 * int(M) + mm], q1_d, rtol=validate_fft_rtol, atol=validate_fft_atol):
                                    raise RuntimeError("E33 FFT mismatch on Q1 ring validation")
                                if not np.isclose(Fq[j, 3 * int(M) + mm], q2_d, rtol=validate_fft_rtol, atol=validate_fft_atol):
                                    raise RuntimeError("E33 FFT mismatch on Q2 ring validation")

                    F_dip_by_field.append(Fd)
                    F_dipquad_by_field.append(Fq)

                # LOFO for all variants at this (w_big,M).
                variants = [
                    ("D", False, "dipole"),  # dipole-only, no B
                    ("BD", True, "dipole"),  # B + dipole
                    ("DQ", False, "dipole+quad"),
                    ("BDQ", True, "dipole+quad"),
                ]

                # Storage for coeff plots (BD and BDQ).
                coef_bd_folds: list[np.ndarray] = []  # (2M,2) raw coefs for Dx/Dy
                coef_bdq_folds: list[np.ndarray] = []  # (4M,2) raw coefs for Dx/Dy/Q1/Q2

                # Per-variant fold metrics + p-values vs C2.
                variant_fold_metrics: dict[str, list[dict[str, Any]]] = {name: [] for name, *_ in variants}
                variant_pvals_vs_c2: dict[str, list[float]] = {name: [] for name, *_ in variants}

                for test_field in range(n_fields):
                    train_fields = [i for i in range(n_fields) if i != test_field]

                    # Assemble shared train/test blocks.
                    Btr_raw = np.concatenate([B_by_field[fid] for fid in train_fields], axis=0)
                    Bte_raw = B_by_field[test_field]
                    Fc2_tr_raw = np.concatenate([Fc2_by_field[fid] for fid in train_fields], axis=0)
                    Fc2_te_raw = Fc2_by_field[test_field]
                    ytr = np.concatenate([y_low_by_field[fid] for fid in train_fields], axis=0)
                    yte = y_low_by_field[test_field]

                    # Standardize B and C2.
                    Btr, Bte, _, _ = standardize(Btr_raw, Bte_raw)
                    Fc2_tr, Fc2_te, _, _ = standardize(Fc2_tr_raw, Fc2_te_raw)
                    y_mu = ytr.mean(axis=0)
                    yc = ytr - y_mu

                    # Compute C2 baseline once (Pearson/relRMSE).
                    Xtr_c2 = np.concatenate([Btr, Fc2_tr], axis=1)
                    Xte_c2 = np.concatenate([Bte, Fc2_te], axis=1)
                    wC2 = solve_ridge(Xtr_c2.T @ Xtr_c2, Xtr_c2.T @ yc)
                    yC2_te = Xte_c2 @ wC2 + y_mu
                    pC2 = pearson_mean_2d(yte, yC2_te)
                    rrC2 = relrmse_mean_2d(yte, yC2_te)

                    # Dipole-only features (standardize per fold).
                    Fd_tr_raw = np.concatenate([F_dip_by_field[fid] for fid in train_fields], axis=0)
                    Fd_te_raw = F_dip_by_field[test_field]
                    Fd_tr, Fd_te, _, sd_fd = standardize(Fd_tr_raw, Fd_te_raw)

                    # Dipole+quad features.
                    Fq_tr_raw = np.concatenate([F_dipquad_by_field[fid] for fid in train_fields], axis=0)
                    Fq_te_raw = F_dipquad_by_field[test_field]
                    Fq_tr, Fq_te, _, sd_fq = standardize(Fq_tr_raw, Fq_te_raw)

                    # Fit each variant, compute placebo p_emp vs C2 (shuffle feature block in TRAIN only).
                    for name, use_B, mode in variants:
                        if mode == "dipole":
                            Ftr = Fd_tr
                            Fte = Fd_te
                            sd = sd_fd
                        else:
                            Ftr = Fq_tr
                            Fte = Fq_te
                            sd = sd_fq

                        if use_B:
                            Xtr = np.concatenate([Btr, Ftr], axis=1)
                            Xte = np.concatenate([Bte, Fte], axis=1)
                        else:
                            Xtr = Ftr
                            Xte = Fte

                        w = solve_ridge(Xtr.T @ Xtr, Xtr.T @ yc)
                        y_pred = Xte @ w + y_mu
                        p = pearson_mean_2d(yte, y_pred)
                        rr = relrmse_mean_2d(yte, y_pred)
                        deltaP_vs_C2 = float(p - pC2)
                        deltaR_vs_C2 = float(rr - rrC2)

                        # Placebo null: shuffle multi-annulus block rows in TRAIN only.
                        null_dP: list[float] = []
                        for _ in range(n_perms):
                            pidx = rng_null.permutation(Ftr.shape[0])
                            Fp = Ftr[pidx]
                            if use_B:
                                Xtrp = np.concatenate([Btr, Fp], axis=1)
                            else:
                                Xtrp = Fp
                            wp = solve_ridge(Xtrp.T @ Xtrp, Xtrp.T @ yc)
                            y_perm = Xte @ wp + y_mu
                            p_perm = pearson_mean_2d(yte, y_perm)
                            null_dP.append(float(p_perm - pC2))
                        null = np.asarray(null_dP, dtype=np.float64)
                        p_emp = float((np.sum(null >= deltaP_vs_C2) + 1.0) / (float(n_perms) + 1.0))
                        variant_pvals_vs_c2[name].append(p_emp)

                        variant_fold_metrics[name].append(
                            {
                                "field_id": int(test_field),
                                "pearson": float(p),
                                "relRMSE": float(rr),
                                "pearson_C2": float(pC2),
                                "relRMSE_C2": float(rrC2),
                                "deltaP_vs_C2": float(deltaP_vs_C2),
                                "deltaR_vs_C2": float(deltaR_vs_C2),
                                "p_emp_vs_C2": float(p_emp),
                            }
                        )

                        # Collect coefficients for BD / BDQ (raw units) for plotting.
                        if name == "BD" and use_B and mode == "dipole":
                            w_raw = w[-Ftr.shape[1] :, :] / sd[:, None]
                            coef_bd_folds.append(w_raw)
                        if name == "BDQ" and use_B and mode == "dipole+quad":
                            w_raw = w[-Ftr.shape[1] :, :] / sd[:, None]
                            coef_bdq_folds.append(w_raw)

                # Aggregate per-variant.
                for name, use_B, mode in variants:
                    rows = variant_fold_metrics[name]
                    pvals = variant_pvals_vs_c2[name]
                    pv = np.asarray([r0["pearson"] for r0 in rows], dtype=np.float64)
                    rv = np.asarray([r0["relRMSE"] for r0 in rows], dtype=np.float64)
                    dp = np.asarray([r0["deltaP_vs_C2"] for r0 in rows], dtype=np.float64)
                    dr = np.asarray([r0["deltaR_vs_C2"] for r0 in rows], dtype=np.float64)
                    npos = int(np.sum(dp > 0.0))
                    fp = fisher_p(pvals)
                    rows_summary.append(
                        {
                            "w_big": int(wb),
                            "M": int(M),
                            "variant": name,
                            "use_B": bool(use_B),
                            "mode": mode,
                            "pearson_mean": float(pv.mean()),
                            "pearson_std": float(pv.std(ddof=1)) if len(pv) > 1 else 0.0,
                            "relRMSE_mean": float(rv.mean()),
                            "relRMSE_std": float(rv.std(ddof=1)) if len(rv) > 1 else 0.0,
                            "pearson_C2_mean": float(np.mean([r0["pearson_C2"] for r0 in rows])),
                            "relRMSE_C2_mean": float(np.mean([r0["relRMSE_C2"] for r0 in rows])),
                            "deltaP_vs_C2_mean": float(dp.mean()),
                            "deltaP_vs_C2_std": float(dp.std(ddof=1)) if len(dp) > 1 else 0.0,
                            "deltaR_vs_C2_mean": float(dr.mean()),
                            "deltaR_vs_C2_std": float(dr.std(ddof=1)) if len(dr) > 1 else 0.0,
                            "fisher_p_vs_C2": float(fp),
                            "n_pos_vs_C2": int(npos),
                        }
                    )

                    tag = f"wbig{wb}_M{M}_{name}"
                    write_csv(
                        paths.run_dir / f"lofo_by_field_{tag}.csv",
                        list(rows[0].keys()) if rows else [],
                        [[r0[k] for k in rows[0].keys()] for r0 in rows] if rows else [],
                    )

                # Coefficient plots (mean±std across folds) for BD and BDQ.
                if coef_bd_folds:
                    coef = np.stack(coef_bd_folds, axis=0)  # (n_fields,2M,2)
                    mean = coef.mean(axis=0)
                    std = coef.std(axis=0, ddof=1) if coef.shape[0] > 1 else np.zeros_like(mean)
                    Dx_gx = mean[: int(M), 0]
                    Dx_gy = mean[: int(M), 1]
                    Dy_gx = mean[int(M) :, 0]
                    Dy_gy = mean[int(M) :, 1]
                    Dx_gx_s = std[: int(M), 0]
                    Dx_gy_s = std[: int(M), 1]
                    Dy_gx_s = std[int(M) :, 0]
                    Dy_gy_s = std[int(M) :, 1]

                    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
                    ax = axes[0]
                    ax.plot(radii, Dx_gx, "-o", label="gx <- Dx")
                    ax.fill_between(radii, Dx_gx - Dx_gx_s, Dx_gx + Dx_gx_s, alpha=0.2)
                    ax.plot(radii, Dy_gx, "-o", label="gx <- Dy")
                    ax.fill_between(radii, Dy_gx - Dy_gx_s, Dy_gx + Dy_gx_s, alpha=0.2)
                    ax.set_title("BD dipole weights (gx)")
                    ax.set_xlabel("radius")
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=8)

                    ax = axes[1]
                    ax.plot(radii, Dx_gy, "-o", label="gy <- Dx")
                    ax.fill_between(radii, Dx_gy - Dx_gy_s, Dx_gy + Dx_gy_s, alpha=0.2)
                    ax.plot(radii, Dy_gy, "-o", label="gy <- Dy")
                    ax.fill_between(radii, Dy_gy - Dy_gy_s, Dy_gy + Dy_gy_s, alpha=0.2)
                    ax.set_title("BD dipole weights (gy)")
                    ax.set_xlabel("radius")
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=8)

                    fig.suptitle(f"E33 BD dipole weights (w_big={wb}, M={M})")
                    fig.tight_layout()
                    out = paths.run_dir / f"coeffs_bd_wbig{wb}_M{M}.png"
                    fig.savefig(out, dpi=150)
                    plt.close(fig)
                    coeff_plots[f"BD_wbig{wb}_M{M}"] = str(out)

                if coef_bdq_folds:
                    coef = np.stack(coef_bdq_folds, axis=0)  # (n_fields,4M,2)
                    mean = coef.mean(axis=0)
                    std = coef.std(axis=0, ddof=1) if coef.shape[0] > 1 else np.zeros_like(mean)
                    q1 = mean[2 * int(M) : 3 * int(M), :]
                    q2 = mean[3 * int(M) : 4 * int(M), :]
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
                    ax = axes[0]
                    ax.plot(radii, q1[:, 0], "-o", label="gx <- Q1")
                    ax.fill_between(radii, q1[:, 0] - std[2 * int(M) : 3 * int(M), 0], q1[:, 0] + std[2 * int(M) : 3 * int(M), 0], alpha=0.2)
                    ax.plot(radii, q2[:, 0], "-o", label="gx <- Q2")
                    ax.fill_between(radii, q2[:, 0] - std[3 * int(M) : 4 * int(M), 0], q2[:, 0] + std[3 * int(M) : 4 * int(M), 0], alpha=0.2)
                    ax.set_title("BDQ quad weights (gx)")
                    ax.set_xlabel("radius")
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=8)

                    ax = axes[1]
                    ax.plot(radii, q1[:, 1], "-o", label="gy <- Q1")
                    ax.fill_between(radii, q1[:, 1] - std[2 * int(M) : 3 * int(M), 1], q1[:, 1] + std[2 * int(M) : 3 * int(M), 1], alpha=0.2)
                    ax.plot(radii, q2[:, 1], "-o", label="gy <- Q2")
                    ax.fill_between(radii, q2[:, 1] - std[3 * int(M) : 4 * int(M), 1], q2[:, 1] + std[3 * int(M) : 4 * int(M), 1], alpha=0.2)
                    ax.set_title("BDQ quad weights (gy)")
                    ax.set_xlabel("radius")
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=8)

                    fig.suptitle(f"E33 BDQ quad weights (w_big={wb}, M={M})")
                    fig.tight_layout()
                    out = paths.run_dir / f"coeffs_bdq_wbig{wb}_M{M}.png"
                    fig.savefig(out, dpi=150)
                    plt.close(fig)
                    coeff_plots[f"BDQ_wbig{wb}_M{M}"] = str(out)

        # Build Markdown rows.
        md_rows: list[dict[str, str]] = []
        for r0 in rows_summary:
            md_rows.append(
                {
                    "w_big": str(int(r0["w_big"])),
                    "M": str(int(r0["M"])),
                    "variant": str(r0["variant"]),
                    "Pearson mean±std": f"{float(r0['pearson_mean']):.4f} ± {float(r0['pearson_std']):.4f}",
                    "relRMSE mean±std": f"{float(r0['relRMSE_mean']):.4f} ± {float(r0['relRMSE_std']):.4f}",
                    "Pearson_C2 (mean)": f"{float(r0['pearson_C2_mean']):.4f}",
                    "ΔPearson vs C2 mean±std": f"{float(r0['deltaP_vs_C2_mean']):.4f} ± {float(r0['deltaP_vs_C2_std']):.4f}",
                    "ΔrelRMSE vs C2 mean±std": f"{float(r0['deltaR_vs_C2_mean']):.4f} ± {float(r0['deltaR_vs_C2_std']):.4f}",
                    "Fisher p (ΔP vs C2)": fmt(float(r0["fisher_p_vs_C2"])),
                    "#folds ΔP>0": f"{int(r0['n_pos_vs_C2'])}/{n_fields}",
                }
            )

        md_pixel: list[dict[str, str]] = []
        for r0 in pixel_rows:
            md_pixel.append(
                {
                    "w_big": str(int(r0["w_big"])),
                    "model": str(r0["model"]),
                    "Pearson mean±std": f"{float(r0['pearson_mean']):.4f} ± {float(r0['pearson_std']):.4f}",
                    "relRMSE mean±std": f"{float(r0['relRMSE_mean']):.4f} ± {float(r0['relRMSE_std']):.4f}",
                }
            )

        coef_lines = "\n".join([f"- `{k}`: `{v}`" for k, v in coeff_plots.items()])
        summary_md = (
            "# E33 — Low-k multi-annulus sweep (LOFO)\n\n"
            f"- run: `{paths.run_dir}`\n"
            f"- grid_size={grid_size}, alpha={alpha}, n_fields={n_fields}, patches_per_field={patches_per_field}\n"
            f"- k0_frac={k0_frac}, w_local={w_local}\n"
            f"- w_bigs={w_bigs}, M_list={M_list}, ring_bin_mode={ring_bin_mode}\n"
            f"- ridge_alpha={ridge_alpha}, perms={n_perms}\n"
            f"- pixels ceiling: SGD L2 alpha={pixel_alpha}, epochs={pixel_epochs}, batch={pixel_batch_size}\n"
            f"- band-split recon: rel_err_gx_mean={float(np.mean(rel_recon_gx)):.3e}, rel_err_gy_mean={float(np.mean(rel_recon_gy)):.3e}\n\n"
            "## Pixels ceiling (P_low)\n\n"
            + md_table(md_pixel, ["w_big", "model", "Pearson mean±std", "relRMSE mean±std"])
            + "\n\n## Feature sweep (targets: gx_low, gy_low)\n\n"
            + md_table(
                md_rows,
                [
                    "w_big",
                    "M",
                    "variant",
                    "Pearson mean±std",
                    "relRMSE mean±std",
                    "Pearson_C2 (mean)",
                    "ΔPearson vs C2 mean±std",
                    "ΔrelRMSE vs C2 mean±std",
                    "Fisher p (ΔP vs C2)",
                    "#folds ΔP>0",
                ],
            )
            + "\n\n## Coefficient plots\n\n"
            + (coef_lines if coef_lines else "- (none)\n")
            + "\n"
        )
        (paths.run_dir / "summary_e33_lowk_multiannulus_sweep.md").write_text(summary_md, encoding="utf-8")

        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "grid_size": grid_size,
                "alpha": alpha,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "k0_frac": k0_frac,
                "w_local": w_local,
                "w_bigs": [int(w) for w in w_bigs],
                "M_list": [int(m) for m in M_list],
                "ring_bin_mode": ring_bin_mode,
                "ridge_alpha": ridge_alpha,
                "n_perms": n_perms,
                "pixel": {
                    "alpha": pixel_alpha,
                    "epochs": pixel_epochs,
                    "batch_size": pixel_batch_size,
                    "eta0": pixel_eta0,
                    "power_t": pixel_power_t,
                },
                "rows_summary": rows_summary,
                "pixel_rows": pixel_rows,
                "coef_plots": coeff_plots,
                "split_recon": {
                    "rel_err_gx_mean": float(np.mean(rel_recon_gx)),
                    "rel_err_gy_mean": float(np.mean(rel_recon_gy)),
                },
            },
        )
        return paths

    if experiment == "e34":
        # E34 — Low-k “physics ceiling”: truncated impulse-response kernel vs dipole basis (and pixel debug).
        #
        # Part A: compute low-k kernels (Kx_low,Ky_low) via impulse-response through the *exact* pipeline,
        #         truncate to w_big in {129,193}, and evaluate prediction by correlation (no learning).
        # Part B: compare to the best dipole multi-annulus basis from E33 (D, M=16) for same w_big.
        # Part C: diagnose pixel underfit at w_big=193 by showing test Pearson vs epochs/eta0.
        from numpy.lib.stride_tricks import sliding_window_view
        from sklearn.linear_model import SGDRegressor

        import matplotlib.pyplot as plt

        grid_size = int(cfg.get("grid_size", 256))
        alpha = float(cfg.get("alpha", 2.0))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 10_000))
        k0_frac = float(cfg.get("k0_frac", 0.15))
        w_bigs = [_require_odd("w_big", int(w)) for w in cfg.get("w_bigs", [129, 193])]

        # Dipole basis (E33 best): D (dipole-only) with M=16.
        M = int(cfg.get("M", 16))
        ring_bin_mode = str(cfg.get("ring_bin_mode", "uniform_r")).lower()
        ridge_alpha = float(cfg.get("ridge_alpha", 1.0))

        # Pixel debug (w_big=193).
        pixel_debug_enabled = bool(cfg.get("pixel_debug_enabled", True))
        pixel_debug_wbig = _require_odd("pixel_debug_wbig", int(cfg.get("pixel_debug_wbig", 193)))
        pixel_debug_test_field = int(cfg.get("pixel_debug_test_field", 0))
        pixel_debug_epochs_max = int(cfg.get("pixel_debug_epochs_max", 8))
        pixel_debug_eta0_bases = [float(x) for x in cfg.get("pixel_debug_eta0_bases", [0.01, 0.05])]
        pixel_alpha = float(cfg.get("pixel_alpha", 1e-4))
        pixel_batch_size = int(cfg.get("pixel_batch_size", 256))
        pixel_power_t = float(cfg.get("pixel_power_t", 0.25))

        validate_fft_n = int(cfg.get("validate_fft_n", 1))
        validate_fft_rtol = float(cfg.get("validate_fft_rtol", 1e-6))
        validate_fft_atol = float(cfg.get("validate_fft_atol", 1e-6))

        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if n_fields < 2:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        if not (0.0 < float(k0_frac) < 0.5):
            raise ValueError("k0_frac must be in (0,0.5)")
        if not w_bigs:
            raise ValueError("w_bigs must be non-empty")
        if any(w > grid_size for w in w_bigs):
            raise ValueError("all w_bigs must be <= grid_size")
        if M <= 0:
            raise ValueError("M must be > 0")
        if ring_bin_mode not in {"uniform_r", "equal_area"}:
            raise ValueError("ring_bin_mode must be 'uniform_r' or 'equal_area'")
        if ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be > 0")
        if pixel_debug_epochs_max <= 0:
            raise ValueError("pixel_debug_epochs_max must be > 0")
        if any(e <= 0 for e in pixel_debug_eta0_bases):
            raise ValueError("pixel_debug_eta0_bases must be > 0")
        if pixel_alpha <= 0:
            raise ValueError("pixel_alpha must be > 0")
        if pixel_batch_size <= 0:
            raise ValueError("pixel_batch_size must be > 0")
        if validate_fft_n < 0:
            raise ValueError("validate_fft_n must be >= 0")
        if validate_fft_rtol < 0 or validate_fft_atol < 0:
            raise ValueError("validate_fft_rtol/atol must be >= 0")

        def safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            am = a - float(a.mean())
            bm = b - float(b.mean())
            denom = float(np.linalg.norm(am) * np.linalg.norm(bm)) + 1e-12
            return float((am @ bm) / denom)

        def relrmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
            e = y_pred - y_true
            rmse = float(np.sqrt(np.mean(e * e)))
            sd = float(np.std(y_true))
            return rmse / (sd + 1e-12)

        def pearson_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            px = safe_corr_1d(y_true[:, 0], y_pred[:, 0])
            py = safe_corr_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (px + py)

        def relrmse_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            rx = relrmse_1d(y_true[:, 0], y_pred[:, 0])
            ry = relrmse_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (rx + ry)

        def solve_ridge(XtX: np.ndarray, Xty: np.ndarray) -> np.ndarray:
            d = int(XtX.shape[0])
            I = np.eye(d, dtype=np.float64)
            try:
                return np.linalg.solve(XtX + ridge_alpha * I, Xty)
            except np.linalg.LinAlgError:
                w, *_ = np.linalg.lstsq(XtX + ridge_alpha * I, Xty, rcond=None)
                return w

        def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            mu = train.mean(axis=0)
            sd = train.std(axis=0)
            sd = np.where(sd > 0, sd, 1.0)
            return (train - mu) / sd, (test - mu) / sd, mu, sd

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r0 in rows:
                out.append("| " + " | ".join(r0.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        def fmt(x: float) -> str:
            if not np.isfinite(x):
                return "nan"
            if abs(float(x)) < 1e-3:
                return f"{x:.3e}"
            return f"{x:.4f}"

        def kernel_fft_centered(kernel: np.ndarray, *, grid_size: int) -> np.ndarray:
            kernel = np.asarray(kernel, dtype=np.float64)
            if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
                raise ValueError(f"kernel must be square 2D, got {kernel.shape}")
            w = int(kernel.shape[0])
            if (w % 2) == 0:
                raise ValueError(f"kernel size must be odd, got {w}")
            if w > grid_size:
                raise ValueError(f"kernel size {w} exceeds grid_size {grid_size}")
            r = w // 2
            kr = np.flip(kernel, axis=(0, 1))
            full = np.zeros((grid_size, grid_size), dtype=np.float64)
            cx = grid_size // 2
            cy = grid_size // 2
            full[cx - r : cx + r + 1, cy - r : cy + r + 1] = kr
            full0 = np.fft.ifftshift(full)
            return np.fft.fftn(full0)

        def annulus_edges(r_min: float, r_max: float, n_bins: int) -> np.ndarray:
            if ring_bin_mode == "uniform_r":
                return np.linspace(r_min, r_max + 1e-9, n_bins + 1, dtype=np.float64)
            rsq = np.linspace(r_min * r_min, r_max * r_max + 1e-9, n_bins + 1, dtype=np.float64)
            return np.sqrt(rsq, dtype=np.float64)

        # Shared fields + low-k splits.
        rho01_by_field: list[np.ndarray] = []
        rho0_by_field: list[np.ndarray] = []
        rhoz_by_field: list[np.ndarray] = []
        split_cache: list[BandSplit2D] = []
        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
            split = band_split_poisson_2d(rho01, k0_frac=float(k0_frac))
            rho01_by_field.append(rho01.astype(np.float64, copy=False))
            rho0_by_field.append((rho01 - float(rho01.mean())).astype(np.float64, copy=False))
            rhoz_by_field.append(_zscore_field(rho01).astype(np.float64, copy=False))
            split_cache.append(split)

        # Impulse-response full kernel (for images + truncation).
        rho_delta = np.zeros((grid_size, grid_size), dtype=np.float64)
        c0 = grid_size // 2
        rho_delta[c0, c0] = 1.0
        split_delta = band_split_poisson_2d(rho_delta, k0_frac=float(k0_frac))
        kfull_gx = np.asarray(split_delta.low.gx, dtype=np.float64, copy=False)
        kfull_gy = np.asarray(split_delta.low.gy, dtype=np.float64, copy=False)
        assert_finite("kernel_full_gx", kfull_gx)
        assert_finite("kernel_full_gy", kfull_gy)

        # Save full kernel images.
        def save_kernel_img(arr: np.ndarray, out: Path, title: str) -> None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            vmax = float(np.max(np.abs(arr))) + 1e-12
            im = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)

        save_kernel_img(kfull_gx, paths.run_dir / "kernel_low_full_gx.png", "Kx_low (full grid, impulse)")
        save_kernel_img(kfull_gy, paths.run_dir / "kernel_low_full_gy.png", "Ky_low (full grid, impulse)")

        # Evaluate kernel ceiling + dipole model for each w_big.
        kernel_rows: list[dict[str, Any]] = []
        dipole_rows: list[dict[str, Any]] = []
        gap_rows: list[dict[str, Any]] = []

        for wb in w_bigs:
            r_big = int(wb) // 2

            # Centers identical to E33.
            centers_by_field: list[np.ndarray] = []
            for field_id in range(n_fields):
                rng_cent = np.random.default_rng(seed + 555_555 + 10_000 * int(wb) + 1_000 * field_id)
                cx = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
                cy = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
                centers_by_field.append(np.column_stack([cx, cy]).astype(np.int64, copy=False))

            def sample_low_at_centers(split: BandSplit2D, centers: np.ndarray) -> np.ndarray:
                cx = centers[:, 0].astype(np.int64)
                cy = centers[:, 1].astype(np.int64)
                return np.column_stack([split.low.gx[cx, cy], split.low.gy[cx, cy]]).astype(np.float64, copy=False)

            y_low_by_field = [sample_low_at_centers(split_cache[fid], centers_by_field[fid]) for fid in range(n_fields)]

            # Part A: truncated kernel ceiling.
            g_patch_gx = kfull_gx[c0 - r_big : c0 + r_big + 1, c0 - r_big : c0 + r_big + 1]
            g_patch_gy = kfull_gy[c0 - r_big : c0 + r_big + 1, c0 - r_big : c0 + r_big + 1]
            # Use correlation-kernel form (flip) as in E31, since kernel_fft_centered flips internally.
            kcorr_gx = g_patch_gx[::-1, ::-1].astype(np.float64, copy=False)
            kcorr_gy = g_patch_gy[::-1, ::-1].astype(np.float64, copy=False)
            kfft_gx = kernel_fft_centered(kcorr_gx, grid_size=grid_size)
            kfft_gy = kernel_fft_centered(kcorr_gy, grid_size=grid_size)

            save_kernel_img(g_patch_gx, paths.run_dir / f"kernel_low_gx_wbig{wb}.png", f"Kx_low truncated (w={wb})")
            save_kernel_img(g_patch_gy, paths.run_dir / f"kernel_low_gy_wbig{wb}.png", f"Ky_low truncated (w={wb})")

            per_field_kernel: list[dict[str, Any]] = []
            for fid in range(n_fields):
                field_fft = np.fft.fftn(rho0_by_field[fid])
                gx_pred = np.fft.ifftn(field_fft * kfft_gx).real
                gy_pred = np.fft.ifftn(field_fft * kfft_gy).real
                centers = centers_by_field[fid]
                cx = centers[:, 0].astype(np.int64)
                cy = centers[:, 1].astype(np.int64)
                y_pred = np.column_stack([gx_pred[cx, cy], gy_pred[cx, cy]]).astype(np.float64, copy=False)
                y_true = y_low_by_field[fid]
                p = pearson_mean_2d(y_true, y_pred)
                rr = relrmse_mean_2d(y_true, y_pred)
                per_field_kernel.append({"field_id": int(fid), "pearson": float(p), "relRMSE": float(rr)})

            kernel_p = np.asarray([r0["pearson"] for r0 in per_field_kernel], dtype=np.float64)
            kernel_r = np.asarray([r0["relRMSE"] for r0 in per_field_kernel], dtype=np.float64)
            kernel_rows.append(
                {
                    "w_big": int(wb),
                    "pearson_mean": float(kernel_p.mean()),
                    "pearson_std": float(kernel_p.std(ddof=1)) if len(kernel_p) > 1 else 0.0,
                    "relRMSE_mean": float(kernel_r.mean()),
                    "relRMSE_std": float(kernel_r.std(ddof=1)) if len(kernel_r) > 1 else 0.0,
                }
            )
            write_csv(
                paths.run_dir / f"kernel_ceiling_by_field_wbig{wb}.csv",
                list(per_field_kernel[0].keys()),
                [[r0[k] for k in per_field_kernel[0].keys()] for r0 in per_field_kernel],
            )

            # Part B: dipole multi-annulus model (E33 D, M=16) on same centers.
            edges = annulus_edges(0.0, float(r_big), int(M))
            radii = 0.5 * (edges[:-1] + edges[1:])
            coords = np.arange(int(wb), dtype=np.float64) - float(r_big)
            dx = coords[:, None]
            dy = coords[None, :]
            rgrid = np.sqrt(dx * dx + dy * dy, dtype=np.float64)

            kernels: dict[str, np.ndarray] = {}
            for m in range(int(M)):
                lo = edges[m]
                hi = edges[m + 1]
                if m == int(M) - 1:
                    ring = (rgrid >= lo) & (rgrid <= hi)
                else:
                    ring = (rgrid >= lo) & (rgrid < hi)
                ring_f = ring.astype(np.float64)
                kernels[f"dx{m}"] = (dx * ring_f).astype(np.float64, copy=False)
                kernels[f"dy{m}"] = (dy * ring_f).astype(np.float64, copy=False)
            kernel_ffts = {name: kernel_fft_centered(k, grid_size=grid_size) for name, k in kernels.items()}

            F_by_field: list[np.ndarray] = []
            for fid in range(n_fields):
                field_fft = np.fft.fftn(rho01_by_field[fid])
                centers = centers_by_field[fid]
                cx = centers[:, 0].astype(np.int64)
                cy = centers[:, 1].astype(np.int64)
                F = np.zeros((patches_per_field, 2 * int(M)), dtype=np.float64)
                for m in range(int(M)):
                    dxm_grid = np.fft.ifftn(field_fft * kernel_ffts[f"dx{m}"]).real
                    dym_grid = np.fft.ifftn(field_fft * kernel_ffts[f"dy{m}"]).real
                    F[:, m] = dxm_grid[cx, cy].astype(np.float64, copy=False)
                    F[:, int(M) + m] = dym_grid[cx, cy].astype(np.float64, copy=False)
                assert_finite("F_dipole", F)
                F_by_field.append(F)

                if fid == 0 and validate_fft_n > 0:
                    n_check = min(int(validate_fft_n), patches_per_field)
                    for j in range(n_check):
                        cxi = int(cx[j])
                        cyi = int(cy[j])
                        patch = rho01_by_field[fid][cxi - r_big : cxi + r_big + 1, cyi - r_big : cyi + r_big + 1]
                        coords_l = np.arange(int(wb), dtype=np.float64) - float(r_big)
                        dx_l = coords_l[:, None]
                        dy_l = coords_l[None, :]
                        r_l = np.sqrt(dx_l * dx_l + dy_l * dy_l, dtype=np.float64)
                        for mm in (0, int(M) - 1):
                            lo = edges[mm]
                            hi = edges[mm + 1]
                            if mm == int(M) - 1:
                                ring = (r_l >= lo) & (r_l <= hi)
                            else:
                                ring = (r_l >= lo) & (r_l < hi)
                            dx_d = float((patch * (dx_l * ring)).sum()) if ring.any() else 0.0
                            dy_d = float((patch * (dy_l * ring)).sum()) if ring.any() else 0.0
                            if not np.isclose(F[j, mm], dx_d, rtol=validate_fft_rtol, atol=validate_fft_atol):
                                raise RuntimeError("E34 FFT mismatch on dipole Dx validation")
                            if not np.isclose(F[j, int(M) + mm], dy_d, rtol=validate_fft_rtol, atol=validate_fft_atol):
                                raise RuntimeError("E34 FFT mismatch on dipole Dy validation")

            per_field_dip: list[dict[str, Any]] = []
            for test_field in range(n_fields):
                train_fields = [i for i in range(n_fields) if i != test_field]
                Ftr_raw = np.concatenate([F_by_field[fid] for fid in train_fields], axis=0)
                Fte_raw = F_by_field[test_field]
                ytr = np.concatenate([y_low_by_field[fid] for fid in train_fields], axis=0)
                yte = y_low_by_field[test_field]
                Ftr, Fte, _, _ = standardize(Ftr_raw, Fte_raw)
                y_mu = ytr.mean(axis=0)
                yc = ytr - y_mu
                w = solve_ridge(Ftr.T @ Ftr, Ftr.T @ yc)
                y_pred = Fte @ w + y_mu
                p = pearson_mean_2d(yte, y_pred)
                rr = relrmse_mean_2d(yte, y_pred)
                per_field_dip.append({"field_id": int(test_field), "pearson": float(p), "relRMSE": float(rr)})

            dip_p = np.asarray([r0["pearson"] for r0 in per_field_dip], dtype=np.float64)
            dip_r = np.asarray([r0["relRMSE"] for r0 in per_field_dip], dtype=np.float64)
            dipole_rows.append(
                {
                    "w_big": int(wb),
                    "M": int(M),
                    "pearson_mean": float(dip_p.mean()),
                    "pearson_std": float(dip_p.std(ddof=1)) if len(dip_p) > 1 else 0.0,
                    "relRMSE_mean": float(dip_r.mean()),
                    "relRMSE_std": float(dip_r.std(ddof=1)) if len(dip_r) > 1 else 0.0,
                }
            )
            write_csv(
                paths.run_dir / f"dipole_by_field_wbig{wb}_M{M}.csv",
                list(per_field_dip[0].keys()),
                [[r0[k] for k in per_field_dip[0].keys()] for r0 in per_field_dip],
            )

            # Gap table.
            gap_rows.append(
                {
                    "w_big": int(wb),
                    "Pearson_ceiling_mean": float(kernel_p.mean()),
                    "Pearson_dipole_mean": float(dip_p.mean()),
                    "gap_Pearson": float(kernel_p.mean() - dip_p.mean()),
                    "relRMSE_ceiling_mean": float(kernel_r.mean()),
                    "relRMSE_dipole_mean": float(dip_r.mean()),
                    "gap_relRMSE": float(kernel_r.mean() - dip_r.mean()),
                }
            )

        # Part C: pixel debug (single fold) to show convergence for w_big=193.
        pixel_debug_rows: list[dict[str, Any]] = []
        if pixel_debug_enabled:
            wb = int(pixel_debug_wbig)
            if wb not in [int(x) for x in w_bigs]:
                # Still allowed, but must fit in grid.
                if wb > grid_size:
                    raise ValueError("pixel_debug_wbig must be <= grid_size")
            r_big = wb // 2
            if not (0 <= pixel_debug_test_field < n_fields):
                raise ValueError("pixel_debug_test_field out of range")

            # Centers for this w_big (E33-style).
            centers_by_field: list[np.ndarray] = []
            for field_id in range(n_fields):
                rng_cent = np.random.default_rng(seed + 555_555 + 10_000 * wb + 1_000 * field_id)
                cx = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
                cy = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
                centers_by_field.append(np.column_stack([cx, cy]).astype(np.int64, copy=False))

            y_low_by_field = [sample_low_at_centers(split_cache[fid], centers_by_field[fid]) for fid in range(n_fields)]
            test_field = int(pixel_debug_test_field)
            train_fields = [i for i in range(n_fields) if i != test_field]

            # Precompute windows for speed.
            wins = [sliding_window_view(rhoz_by_field[fid], (wb, wb)) for fid in range(n_fields)]

            def eval_model(regs: tuple[SGDRegressor, SGDRegressor]) -> tuple[float, float]:
                reg_gx, reg_gy = regs
                centers = centers_by_field[test_field]
                y_true = y_low_by_field[test_field]
                out = np.empty_like(y_true)
                for start in range(0, centers.shape[0], pixel_batch_size):
                    idx = slice(start, min(centers.shape[0], start + pixel_batch_size))
                    cx = centers[idx, 0].astype(np.int64)
                    cy = centers[idx, 1].astype(np.int64)
                    Xb = wins[test_field][cx - r_big, cy - r_big].reshape(len(cx), -1).astype(np.float32, copy=False)
                    out[idx, 0] = reg_gx.predict(Xb)
                    out[idx, 1] = reg_gy.predict(Xb)
                p = pearson_mean_2d(y_true, out)
                rr = relrmse_mean_2d(y_true, out)
                return p, rr

            # Train per eta0_base up to max epochs, recording test metrics each epoch.
            for eta0_base in pixel_debug_eta0_bases:
                eta0_eff = float(eta0_base) / float(wb)
                reg_gx = SGDRegressor(
                    loss="squared_error",
                    penalty="l2",
                    alpha=float(pixel_alpha),
                    learning_rate="invscaling",
                    eta0=float(eta0_eff),
                    power_t=float(pixel_power_t),
                    max_iter=1,
                    tol=None,
                    fit_intercept=True,
                    average=True,
                    random_state=int(placebo_seed) + 7000,
                )
                reg_gy = SGDRegressor(
                    loss="squared_error",
                    penalty="l2",
                    alpha=float(pixel_alpha),
                    learning_rate="invscaling",
                    eta0=float(eta0_eff),
                    power_t=float(pixel_power_t),
                    max_iter=1,
                    tol=None,
                    fit_intercept=True,
                    average=True,
                    random_state=int(placebo_seed) + 7001,
                )

                for epoch in range(1, int(pixel_debug_epochs_max) + 1):
                    for fid in train_fields:
                        centers = centers_by_field[fid]
                        y = y_low_by_field[fid]
                        perm = np.random.default_rng(int(placebo_seed) + 8000 + 100 * epoch + fid).permutation(centers.shape[0])
                        for start in range(0, centers.shape[0], pixel_batch_size):
                            idx = perm[start : start + pixel_batch_size]
                            cx = centers[idx, 0].astype(np.int64)
                            cy = centers[idx, 1].astype(np.int64)
                            Xb = wins[fid][cx - r_big, cy - r_big].reshape(len(idx), -1).astype(np.float32, copy=False)
                            reg_gx.partial_fit(Xb, y[idx, 0])
                            reg_gy.partial_fit(Xb, y[idx, 1])

                    p, rr = eval_model((reg_gx, reg_gy))
                    pixel_debug_rows.append(
                        {
                            "eta0_base": float(eta0_base),
                            "epoch": int(epoch),
                            "pearson_test": float(p),
                            "relRMSE_test": float(rr),
                        }
                    )

        # Write summary markdown.
        md_kernel = [
            {
                "w_big": str(r0["w_big"]),
                "Pearson mean±std": f"{r0['pearson_mean']:.4f} ± {r0['pearson_std']:.4f}",
                "relRMSE mean±std": f"{r0['relRMSE_mean']:.4f} ± {r0['relRMSE_std']:.4f}",
            }
            for r0 in kernel_rows
        ]
        md_dip = [
            {
                "w_big": str(r0["w_big"]),
                "M": str(r0["M"]),
                "Pearson mean±std": f"{r0['pearson_mean']:.4f} ± {r0['pearson_std']:.4f}",
                "relRMSE mean±std": f"{r0['relRMSE_mean']:.4f} ± {r0['relRMSE_std']:.4f}",
            }
            for r0 in dipole_rows
        ]
        md_gap = [
            {
                "w_big": str(r0["w_big"]),
                "Pearson_ceiling": f"{r0['Pearson_ceiling_mean']:.4f}",
                "Pearson_dipole": f"{r0['Pearson_dipole_mean']:.4f}",
                "gap_Pearson": f"{r0['gap_Pearson']:.4f}",
                "relRMSE_ceiling": f"{r0['relRMSE_ceiling_mean']:.4f}",
                "relRMSE_dipole": f"{r0['relRMSE_dipole_mean']:.4f}",
                "gap_relRMSE": f"{r0['gap_relRMSE']:.4f}",
            }
            for r0 in gap_rows
        ]
        md_pix = [
            {
                "eta0_base": f"{r0['eta0_base']:.3g}",
                "epoch": str(r0["epoch"]),
                "Pearson_test": f"{r0['pearson_test']:.4f}",
                "relRMSE_test": f"{r0['relRMSE_test']:.4f}",
            }
            for r0 in pixel_debug_rows
        ]

        summary_md = (
            "# E34 — Low-k kernel ceiling vs dipole basis\n\n"
            f"- run: `{paths.run_dir}`\n"
            f"- grid_size={grid_size}, alpha={alpha}, k0_frac={k0_frac}, n_fields={n_fields}, patches_per_field={patches_per_field}\n"
            f"- w_bigs={w_bigs}\n"
            f"- dipole: D (M={M}, ring_bin_mode={ring_bin_mode}), ridge_alpha={ridge_alpha}\n\n"
            "## A) Kernel ceiling (truncated impulse kernel)\n\n"
            + md_table(md_kernel, ["w_big", "Pearson mean±std", "relRMSE mean±std"])
            + "\n\n## B) Dipole basis (LOFO)\n\n"
            + md_table(md_dip, ["w_big", "M", "Pearson mean±std", "relRMSE mean±std"])
            + "\n\n## Gap (ceiling - dipole)\n\n"
            + md_table(md_gap, ["w_big", "Pearson_ceiling", "Pearson_dipole", "gap_Pearson", "relRMSE_ceiling", "relRMSE_dipole", "gap_relRMSE"])
            + "\n\n## C) Pixel debug (w_big=193)\n\n"
            + (md_table(md_pix, ["eta0_base", "epoch", "Pearson_test", "relRMSE_test"]) if md_pix else "- (disabled)\n")
            + "\n\n## Kernel images\n\n"
            + "- `kernel_low_full_gx.png`\n"
            + "- `kernel_low_full_gy.png`\n"
            + "\n"
            + "\n".join([f"- `kernel_low_gx_wbig{w}.png` / `kernel_low_gy_wbig{w}.png`" for w in w_bigs])
            + "\n"
        )
        (paths.run_dir / "summary_e34_lowk_kernel_ceiling.md").write_text(summary_md, encoding="utf-8")

        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "grid_size": grid_size,
                "alpha": alpha,
                "k0_frac": k0_frac,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "w_bigs": [int(w) for w in w_bigs],
                "kernel_rows": kernel_rows,
                "dipole_rows": dipole_rows,
                "gap_rows": gap_rows,
                "pixel_debug_rows": pixel_debug_rows,
            },
        )
        return paths

    if experiment == "e35":
        # E35 — Dipole multi-annulus convergence vs kernel ceiling (and best-in-basis projection).
        #
        # For each w_big in {129,193} and M in {4,8,16,32,64}:
        #  A) Learned (LOFO): ridge on dipole annulus features (Dx_m, Dy_m).
        #  B) Projected: approximate the *mapping kernel* (kcorr = flipped impulse response) in L2 on the same basis,
        #     then predict using the resulting coefficients (no training).
        # Compare both to the truncated impulse-kernel ceiling.
        import matplotlib.pyplot as plt

        grid_size = int(cfg.get("grid_size", 256))
        alpha = float(cfg.get("alpha", 2.0))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 10_000))
        k0_frac = float(cfg.get("k0_frac", 0.15))
        w_bigs = [_require_odd("w_big", int(w)) for w in cfg.get("w_bigs", [129, 193])]

        M_list = [int(m) for m in cfg.get("M_list", [4, 8, 16, 32, 64])]
        if any(m <= 0 for m in M_list):
            raise ValueError("All M_list entries must be > 0")
        ring_bin_mode = str(cfg.get("ring_bin_mode", "uniform_r")).lower()
        ridge_alpha = float(cfg.get("ridge_alpha", 1.0))

        gap_target_pearson = float(cfg.get("gap_target_pearson", 0.002))
        gap_target_relrmse = float(cfg.get("gap_target_relrmse", 0.02))

        plot_wbig = _require_odd("plot_wbig", int(cfg.get("plot_wbig", 193)))
        plot_M = int(cfg.get("plot_M", max(M_list)))

        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if n_fields < 2:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        if not (0.0 < float(k0_frac) < 0.5):
            raise ValueError("k0_frac must be in (0,0.5)")
        if any(w > grid_size for w in w_bigs):
            raise ValueError("all w_bigs must be <= grid_size")
        if ring_bin_mode not in {"uniform_r", "equal_area"}:
            raise ValueError("ring_bin_mode must be 'uniform_r' or 'equal_area'")
        if ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be > 0")
        if gap_target_pearson < 0 or gap_target_relrmse < 0:
            raise ValueError("gap_target_* must be >= 0")

        def safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            am = a - float(a.mean())
            bm = b - float(b.mean())
            denom = float(np.linalg.norm(am) * np.linalg.norm(bm)) + 1e-12
            return float((am @ bm) / denom)

        def relrmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
            e = y_pred - y_true
            rmse = float(np.sqrt(np.mean(e * e)))
            sd = float(np.std(y_true))
            return rmse / (sd + 1e-12)

        def pearson_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            px = safe_corr_1d(y_true[:, 0], y_pred[:, 0])
            py = safe_corr_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (px + py)

        def relrmse_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            rx = relrmse_1d(y_true[:, 0], y_pred[:, 0])
            ry = relrmse_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (rx + ry)

        def solve_ridge(XtX: np.ndarray, Xty: np.ndarray) -> np.ndarray:
            d = int(XtX.shape[0])
            I = np.eye(d, dtype=np.float64)
            try:
                return np.linalg.solve(XtX + ridge_alpha * I, Xty)
            except np.linalg.LinAlgError:
                w, *_ = np.linalg.lstsq(XtX + ridge_alpha * I, Xty, rcond=None)
                return w

        def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            mu = train.mean(axis=0)
            sd = train.std(axis=0)
            sd = np.where(sd > 0, sd, 1.0)
            return (train - mu) / sd, (test - mu) / sd, mu, sd

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r0 in rows:
                out.append("| " + " | ".join(r0.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        def annulus_edges(r_min: float, r_max: float, n_bins: int) -> np.ndarray:
            if ring_bin_mode == "uniform_r":
                return np.linspace(r_min, r_max + 1e-9, n_bins + 1, dtype=np.float64)
            rsq = np.linspace(r_min * r_min, r_max * r_max + 1e-9, n_bins + 1, dtype=np.float64)
            return np.sqrt(rsq, dtype=np.float64)

        def kernel_fft_centered(kernel: np.ndarray, *, grid_size: int) -> np.ndarray:
            kernel = np.asarray(kernel, dtype=np.float64)
            if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
                raise ValueError(f"kernel must be square 2D, got {kernel.shape}")
            w = int(kernel.shape[0])
            if (w % 2) == 0:
                raise ValueError(f"kernel size must be odd, got {w}")
            if w > grid_size:
                raise ValueError(f"kernel size {w} exceeds grid_size {grid_size}")
            r = w // 2
            kr = np.flip(kernel, axis=(0, 1))
            full = np.zeros((grid_size, grid_size), dtype=np.float64)
            cx = grid_size // 2
            cy = grid_size // 2
            full[cx - r : cx + r + 1, cy - r : cy + r + 1] = kr
            full0 = np.fft.ifftshift(full)
            return np.fft.fftn(full0)

        def save_kernel_img(arr: np.ndarray, out: Path, title: str) -> None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            vmax = float(np.max(np.abs(arr))) + 1e-12
            im = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)

        # Generate fields + band splits.
        rho01_by_field: list[np.ndarray] = []
        rho0_by_field: list[np.ndarray] = []
        split_cache: list[BandSplit2D] = []
        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
            split = band_split_poisson_2d(rho01, k0_frac=float(k0_frac))
            rho01_by_field.append(rho01.astype(np.float64, copy=False))
            rho0_by_field.append((rho01 - float(rho01.mean())).astype(np.float64, copy=False))
            split_cache.append(split)

        # Impulse-response low-k kernel (full grid), used to extract truncated kernels for each w_big.
        rho_delta = np.zeros((grid_size, grid_size), dtype=np.float64)
        c0 = grid_size // 2
        rho_delta[c0, c0] = 1.0
        split_delta = band_split_poisson_2d(rho_delta, k0_frac=float(k0_frac))
        kfull_gx = np.asarray(split_delta.low.gx, dtype=np.float64, copy=False)
        kfull_gy = np.asarray(split_delta.low.gy, dtype=np.float64, copy=False)
        assert_finite("kernel_full_gx", kfull_gx)
        assert_finite("kernel_full_gy", kfull_gy)

        save_kernel_img(kfull_gx, paths.run_dir / "kernel_low_full_gx.png", "Kx_low (full grid, impulse)")
        save_kernel_img(kfull_gy, paths.run_dir / "kernel_low_full_gy.png", "Ky_low (full grid, impulse)")

        def sample_low_at_centers(split: BandSplit2D, centers: np.ndarray) -> np.ndarray:
            cx = centers[:, 0].astype(np.int64)
            cy = centers[:, 1].astype(np.int64)
            return np.column_stack([split.low.gx[cx, cy], split.low.gy[cx, cy]]).astype(np.float64, copy=False)

        rows_all: list[dict[str, Any]] = []
        kernel_approx_stats: list[dict[str, Any]] = []

        # For coefficient plots at a representative setting (w_big=plot_wbig, M=plot_M).
        learned_coefs_by_fold: list[np.ndarray] = []  # raw-scale coefs (2M,2)
        learned_bin_edges: np.ndarray | None = None
        proj_coefs_x: np.ndarray | None = None
        proj_coefs_y: np.ndarray | None = None
        proj_kernel_gx: np.ndarray | None = None
        proj_kernel_gy: np.ndarray | None = None
        true_kernel_gx: np.ndarray | None = None
        true_kernel_gy: np.ndarray | None = None
        plot_r_mid: np.ndarray | None = None

        # Compute ceiling once per w_big.
        ceiling_cache: dict[int, dict[str, Any]] = {}

        for wb in w_bigs:
            r_big = int(wb) // 2

            # Patch centers (E33/E34-style).
            centers_by_field: list[np.ndarray] = []
            for field_id in range(n_fields):
                rng_cent = np.random.default_rng(seed + 555_555 + 10_000 * int(wb) + 1_000 * field_id)
                cx = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
                cy = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
                centers_by_field.append(np.column_stack([cx, cy]).astype(np.int64, copy=False))

            y_low_by_field = [sample_low_at_centers(split_cache[fid], centers_by_field[fid]) for fid in range(n_fields)]

            # Truncated impulse kernels (offset form), then the mapping kernel kcorr = flipped patch.
            g_patch_gx = kfull_gx[c0 - r_big : c0 + r_big + 1, c0 - r_big : c0 + r_big + 1]
            g_patch_gy = kfull_gy[c0 - r_big : c0 + r_big + 1, c0 - r_big : c0 + r_big + 1]
            kcorr_gx = g_patch_gx[::-1, ::-1].astype(np.float64, copy=False)
            kcorr_gy = g_patch_gy[::-1, ::-1].astype(np.float64, copy=False)

            save_kernel_img(g_patch_gx, paths.run_dir / f"kernel_low_gx_wbig{wb}.png", f"Kx_low truncated (w={wb})")
            save_kernel_img(g_patch_gy, paths.run_dir / f"kernel_low_gy_wbig{wb}.png", f"Ky_low truncated (w={wb})")

            # Kernel ceiling (no learning): convolve mean-subtracted rho with offset-form kernel.
            kfft_gx = kernel_fft_centered(kcorr_gx, grid_size=grid_size)  # flips internally -> uses g_patch_gx
            kfft_gy = kernel_fft_centered(kcorr_gy, grid_size=grid_size)
            per_field_ceiling: list[dict[str, Any]] = []
            for fid in range(n_fields):
                field_fft = np.fft.fftn(rho0_by_field[fid])
                gx_pred = np.fft.ifftn(field_fft * kfft_gx).real
                gy_pred = np.fft.ifftn(field_fft * kfft_gy).real
                centers = centers_by_field[fid]
                cx = centers[:, 0].astype(np.int64)
                cy = centers[:, 1].astype(np.int64)
                y_pred = np.column_stack([gx_pred[cx, cy], gy_pred[cx, cy]]).astype(np.float64, copy=False)
                y_true = y_low_by_field[fid]
                p = pearson_mean_2d(y_true, y_pred)
                rr = relrmse_mean_2d(y_true, y_pred)
                per_field_ceiling.append({"field_id": int(fid), "pearson": float(p), "relRMSE": float(rr)})

            ceiling_p = np.asarray([r0["pearson"] for r0 in per_field_ceiling], dtype=np.float64)
            ceiling_r = np.asarray([r0["relRMSE"] for r0 in per_field_ceiling], dtype=np.float64)
            ceiling_cache[int(wb)] = {
                "pearson_mean": float(ceiling_p.mean()),
                "pearson_std": float(ceiling_p.std(ddof=1)) if len(ceiling_p) > 1 else 0.0,
                "relRMSE_mean": float(ceiling_r.mean()),
                "relRMSE_std": float(ceiling_r.std(ddof=1)) if len(ceiling_r) > 1 else 0.0,
            }

            # Sweep M.
            for M in M_list:
                edges = annulus_edges(0.0, float(r_big), int(M))
                coords = np.arange(int(wb), dtype=np.float64) - float(r_big)
                dx = coords[:, None]
                dy = coords[None, :]
                rgrid = np.sqrt(dx * dx + dy * dy, dtype=np.float64)

                # Basis kernels for dipole features.
                kernels: dict[str, np.ndarray] = {}
                coeff_x = np.zeros((int(M),), dtype=np.float64)
                coeff_y = np.zeros((int(M),), dtype=np.float64)
                Kx_approx = np.zeros((int(wb), int(wb)), dtype=np.float64)
                Ky_approx = np.zeros((int(wb), int(wb)), dtype=np.float64)

                for m in range(int(M)):
                    lo = edges[m]
                    hi = edges[m + 1]
                    if m == int(M) - 1:
                        ring = (rgrid >= lo) & (rgrid <= hi)
                    else:
                        ring = (rgrid >= lo) & (rgrid < hi)
                    ring_f = ring.astype(np.float64)
                    bx = (dx * ring_f).astype(np.float64, copy=False)
                    by = (dy * ring_f).astype(np.float64, copy=False)

                    denom_x = float(np.sum(bx * bx))
                    denom_y = float(np.sum(by * by))
                    if denom_x > 0:
                        coeff_x[m] = float(np.sum(kcorr_gx * bx)) / (denom_x + 1e-12)
                    else:
                        coeff_x[m] = 0.0
                    if denom_y > 0:
                        coeff_y[m] = float(np.sum(kcorr_gy * by)) / (denom_y + 1e-12)
                    else:
                        coeff_y[m] = 0.0
                    Kx_approx += coeff_x[m] * bx
                    Ky_approx += coeff_y[m] * by

                    kernels[f"dx{m}"] = bx
                    kernels[f"dy{m}"] = by

                # Kernel approximation stats (map-kernel space).
                corr_kx = safe_corr_1d(kcorr_gx, Kx_approx)
                corr_ky = safe_corr_1d(kcorr_gy, Ky_approx)
                rel_kx = float(np.linalg.norm((Kx_approx - kcorr_gx).ravel()) / (np.linalg.norm(kcorr_gx.ravel()) + 1e-12))
                rel_ky = float(np.linalg.norm((Ky_approx - kcorr_gy).ravel()) / (np.linalg.norm(kcorr_gy.ravel()) + 1e-12))
                kernel_approx_stats.append(
                    {
                        "w_big": int(wb),
                        "M": int(M),
                        "corr_kernel_mean": float(0.5 * (corr_kx + corr_ky)),
                        "relL2_kernel_mean": float(0.5 * (rel_kx + rel_ky)),
                    }
                )

                # Save kernel approximation visuals for the representative plot setting.
                if int(wb) == int(plot_wbig) and int(M) == int(plot_M):
                    true_kernel_gx = kcorr_gx.copy()
                    true_kernel_gy = kcorr_gy.copy()
                    proj_kernel_gx = Kx_approx.copy()
                    proj_kernel_gy = Ky_approx.copy()
                    proj_coefs_x = coeff_x.copy()
                    proj_coefs_y = coeff_y.copy()
                    learned_bin_edges = edges.copy()
                    plot_r_mid = 0.5 * (edges[:-1] + edges[1:])

                kernel_ffts = {name: kernel_fft_centered(k, grid_size=grid_size) for name, k in kernels.items()}

                # Build dipole features for all fields at this (w_big,M).
                F_by_field: list[np.ndarray] = []
                for fid in range(n_fields):
                    field_fft = np.fft.fftn(rho01_by_field[fid])
                    centers = centers_by_field[fid]
                    cx = centers[:, 0].astype(np.int64)
                    cy = centers[:, 1].astype(np.int64)
                    F = np.zeros((patches_per_field, 2 * int(M)), dtype=np.float64)
                    for m in range(int(M)):
                        dxm_grid = np.fft.ifftn(field_fft * kernel_ffts[f"dx{m}"]).real
                        dym_grid = np.fft.ifftn(field_fft * kernel_ffts[f"dy{m}"]).real
                        F[:, m] = dxm_grid[cx, cy].astype(np.float64, copy=False)
                        F[:, int(M) + m] = dym_grid[cx, cy].astype(np.float64, copy=False)
                    assert_finite("F_dipole", F)
                    F_by_field.append(F)

                # Method B: projected mapping (no training), evaluated per field.
                per_field_proj: list[dict[str, Any]] = []
                for fid in range(n_fields):
                    F = F_by_field[fid]
                    y_pred = np.column_stack(
                        [
                            (F[:, : int(M)] @ coeff_x).astype(np.float64, copy=False),
                            (F[:, int(M) :] @ coeff_y).astype(np.float64, copy=False),
                        ]
                    )
                    y_true = y_low_by_field[fid]
                    p = pearson_mean_2d(y_true, y_pred)
                    rr = relrmse_mean_2d(y_true, y_pred)
                    per_field_proj.append({"field_id": int(fid), "pearson": float(p), "relRMSE": float(rr)})
                proj_p = np.asarray([r0["pearson"] for r0 in per_field_proj], dtype=np.float64)
                proj_r = np.asarray([r0["relRMSE"] for r0 in per_field_proj], dtype=np.float64)

                # Method A: learned ridge in LOFO.
                per_field_learn: list[dict[str, Any]] = []
                fold_raw_coefs: list[np.ndarray] = []
                for test_field in range(n_fields):
                    train_fields = [i for i in range(n_fields) if i != test_field]
                    Ftr_raw = np.concatenate([F_by_field[fid] for fid in train_fields], axis=0)
                    Fte_raw = F_by_field[test_field]
                    ytr = np.concatenate([y_low_by_field[fid] for fid in train_fields], axis=0)
                    yte = y_low_by_field[test_field]

                    Ftr, Fte, mu, sd = standardize(Ftr_raw, Fte_raw)
                    y_mu = ytr.mean(axis=0)
                    yc = ytr - y_mu
                    w = solve_ridge(Ftr.T @ Ftr, Ftr.T @ yc)
                    y_pred = Fte @ w + y_mu

                    p = pearson_mean_2d(yte, y_pred)
                    rr = relrmse_mean_2d(yte, y_pred)
                    per_field_learn.append({"field_id": int(test_field), "pearson": float(p), "relRMSE": float(rr)})

                    # Raw-scale coefficients for plotting (convert from standardized space).
                    w_raw = w / sd[:, None]
                    fold_raw_coefs.append(w_raw.astype(np.float64, copy=False))

                learn_p = np.asarray([r0["pearson"] for r0 in per_field_learn], dtype=np.float64)
                learn_r = np.asarray([r0["relRMSE"] for r0 in per_field_learn], dtype=np.float64)

                # Capture fold coefficients for representative plot setting.
                if int(wb) == int(plot_wbig) and int(M) == int(plot_M):
                    learned_coefs_by_fold = fold_raw_coefs

                # Append summary row.
                c = ceiling_cache[int(wb)]
                rows_all.append(
                    {
                        "w_big": int(wb),
                        "M": int(M),
                        "ceiling_pearson_mean": float(c["pearson_mean"]),
                        "ceiling_pearson_std": float(c["pearson_std"]),
                        "ceiling_relRMSE_mean": float(c["relRMSE_mean"]),
                        "ceiling_relRMSE_std": float(c["relRMSE_std"]),
                        "learn_pearson_mean": float(learn_p.mean()),
                        "learn_pearson_std": float(learn_p.std(ddof=1)) if len(learn_p) > 1 else 0.0,
                        "learn_relRMSE_mean": float(learn_r.mean()),
                        "learn_relRMSE_std": float(learn_r.std(ddof=1)) if len(learn_r) > 1 else 0.0,
                        "proj_pearson_mean": float(proj_p.mean()),
                        "proj_pearson_std": float(proj_p.std(ddof=1)) if len(proj_p) > 1 else 0.0,
                        "proj_relRMSE_mean": float(proj_r.mean()),
                        "proj_relRMSE_std": float(proj_r.std(ddof=1)) if len(proj_r) > 1 else 0.0,
                    }
                )

        # Coefficient plots + kernel approximation plots for representative setting.
        plot_lines: list[str] = []
        if (
            true_kernel_gx is not None
            and true_kernel_gy is not None
            and proj_kernel_gx is not None
            and proj_kernel_gy is not None
            and proj_coefs_x is not None
            and proj_coefs_y is not None
            and learned_bin_edges is not None
            and plot_r_mid is not None
        ):
            # Kernel compare images.
            for comp, true_k, approx_k in [
                ("gx", true_kernel_gx, proj_kernel_gx),
                ("gy", true_kernel_gy, proj_kernel_gy),
            ]:
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                vmax = float(np.max(np.abs(true_k))) + 1e-12
                axes[0].imshow(true_k, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
                axes[0].set_title(f"true kcorr_{comp}")
                axes[1].imshow(approx_k, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
                axes[1].set_title(f"proj approx_{comp}")
                diff = approx_k - true_k
                vmax_d = float(np.max(np.abs(diff))) + 1e-12
                axes[2].imshow(diff, cmap="RdBu_r", vmin=-vmax_d, vmax=vmax_d, interpolation="nearest")
                axes[2].set_title("diff")
                for ax in axes:
                    ax.set_xticks([])
                    ax.set_yticks([])
                fig.tight_layout()
                out = paths.run_dir / f"kernel_approx_vs_true_wbig{plot_wbig}_M{plot_M}_{comp}.png"
                fig.savefig(out, dpi=150)
                plt.close(fig)
                plot_lines.append(f"- `{out.name}`")

            # Coeff vs radius plot.
            fig, ax = plt.subplots(1, 1, figsize=(7, 4))
            ax.plot(plot_r_mid, proj_coefs_x, label="proj: Dx->gx", lw=2)
            ax.plot(plot_r_mid, proj_coefs_y, label="proj: Dy->gy", lw=2)
            if learned_coefs_by_fold:
                W = np.stack(learned_coefs_by_fold, axis=0)  # (folds, 2M, 2)
                M0 = int(plot_M)
                # main blocks (expected dominant)
                dx_gx = W[:, :M0, 0].mean(axis=0)
                dy_gy = W[:, M0 : 2 * M0, 1].mean(axis=0)
                ax.plot(plot_r_mid, dx_gx, "--", label="learned mean: Dx->gx", lw=2)
                ax.plot(plot_r_mid, dy_gy, "--", label="learned mean: Dy->gy", lw=2)
            ax.axhline(0.0, color="k", lw=1, alpha=0.3)
            ax.set_xlabel("annulus radius (midpoint)")
            ax.set_ylabel("coefficient (raw units)")
            ax.set_title(f"Coefficients vs radius (w_big={plot_wbig}, M={plot_M})")
            ax.legend(loc="best", fontsize=9)
            fig.tight_layout()
            out = paths.run_dir / f"coeffs_vs_radius_wbig{plot_wbig}_M{plot_M}.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            plot_lines.append(f"- `{out.name}`")

        # Derive "M needed" statements.
        def meets_target(row: dict[str, Any], *, which: str) -> bool:
            if which not in {"learn", "proj"}:
                raise ValueError("which must be 'learn' or 'proj'")
            gp = float(row[f"{which}_pearson_mean"]) - float(row["ceiling_pearson_mean"])
            gr = float(row[f"{which}_relRMSE_mean"]) - float(row["ceiling_relRMSE_mean"])
            return (abs(gp) <= gap_target_pearson) and (abs(gr) <= gap_target_relrmse)

        m_needed_lines: list[str] = []
        for wb in w_bigs:
            rows_w = [r0 for r0 in rows_all if int(r0["w_big"]) == int(wb)]
            rows_w = sorted(rows_w, key=lambda r0: int(r0["M"]))
            for which in ("learn", "proj"):
                ms_ok = [int(r0["M"]) for r0 in rows_w if meets_target(r0, which=which)]
                if ms_ok:
                    m_needed_lines.append(f"- w_big={int(wb)}: {which} reaches targets at M={min(ms_ok)}")
                else:
                    m_needed_lines.append(f"- w_big={int(wb)}: {which} does not reach targets up to M={max(M_list)}")

        # Build markdown tables.
        md_rows: list[dict[str, str]] = []
        for r0 in sorted(rows_all, key=lambda x: (int(x["w_big"]), int(x["M"]))):
            cp = float(r0["ceiling_pearson_mean"])
            cr = float(r0["ceiling_relRMSE_mean"])
            lp = float(r0["learn_pearson_mean"])
            lr = float(r0["learn_relRMSE_mean"])
            pp = float(r0["proj_pearson_mean"])
            pr = float(r0["proj_relRMSE_mean"])
            md_rows.append(
                {
                    "w_big": str(r0["w_big"]),
                    "M": str(r0["M"]),
                    "Pearson_ceiling": f"{cp:.4f}",
                    "relRMSE_ceiling": f"{cr:.4f}",
                    "Pearson_learned": f"{lp:.4f} ± {float(r0['learn_pearson_std']):.4f}",
                    "relRMSE_learned": f"{lr:.4f} ± {float(r0['learn_relRMSE_std']):.4f}",
                    "Pearson_proj": f"{pp:.4f} ± {float(r0['proj_pearson_std']):.4f}",
                    "relRMSE_proj": f"{pr:.4f} ± {float(r0['proj_relRMSE_std']):.4f}",
                    "gapP_learn": f"{(lp - cp):+.4f}",
                    "gapR_learn": f"{(lr - cr):+.4f}",
                    "gapP_proj": f"{(pp - cp):+.4f}",
                    "gapR_proj": f"{(pr - cr):+.4f}",
                }
            )

        md_kstats: list[dict[str, str]] = []
        for r0 in sorted(kernel_approx_stats, key=lambda x: (int(x["w_big"]), int(x["M"]))):
            md_kstats.append(
                {
                    "w_big": str(r0["w_big"]),
                    "M": str(r0["M"]),
                    "corr(Kapprox, Ktrue)": f"{float(r0['corr_kernel_mean']):.4f}",
                    "relL2(Kapprox, Ktrue)": f"{float(r0['relL2_kernel_mean']):.4f}",
                }
            )

        summary_md = (
            "# E35 — Dipole multi-annulus convergence vs kernel ceiling\n\n"
            f"- run: `{paths.run_dir}`\n"
            f"- grid_size={grid_size}, alpha={alpha}, k0_frac={k0_frac}, n_fields={n_fields}, patches_per_field={patches_per_field}\n"
            f"- w_bigs={w_bigs}, M_list={M_list}, ring_bin_mode={ring_bin_mode}, ridge_alpha={ridge_alpha}\n"
            f"- targets: gapP<= {gap_target_pearson}, gapRelRMSE<= {gap_target_relrmse}\n\n"
            "## Metrics vs kernel ceiling\n\n"
            + md_table(
                md_rows,
                [
                    "w_big",
                    "M",
                    "Pearson_ceiling",
                    "relRMSE_ceiling",
                    "Pearson_learned",
                    "relRMSE_learned",
                    "Pearson_proj",
                    "relRMSE_proj",
                    "gapP_learn",
                    "gapR_learn",
                    "gapP_proj",
                    "gapR_proj",
                ],
            )
            + "\n\n## Kernel approximation quality (projection only)\n\n"
            + md_table(md_kstats, ["w_big", "M", "corr(Kapprox, Ktrue)", "relL2(Kapprox, Ktrue)"])
            + "\n\n## M needed to reach targets\n\n"
            + ("\n".join(m_needed_lines) + "\n")
            + ("\n## Plots\n\n" + ("\n".join(plot_lines) + "\n") if plot_lines else "\n## Plots\n\n- (none)\n")
        )

        (paths.run_dir / "summary_e35_dipole_convergence.md").write_text(summary_md, encoding="utf-8")
        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "grid_size": grid_size,
                "alpha": alpha,
                "k0_frac": k0_frac,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "w_bigs": [int(w) for w in w_bigs],
                "M_list": [int(m) for m in M_list],
                "ring_bin_mode": ring_bin_mode,
                "ridge_alpha": ridge_alpha,
                "gap_targets": {"pearson": gap_target_pearson, "relRMSE": gap_target_relrmse},
                "rows": rows_all,
                "kernel_approx_stats": kernel_approx_stats,
            },
        )
        return paths

    if experiment == "e36":
        # E36 — Sectorized annulus dipole basis to test lattice anisotropy and close the low-k relRMSE gap.
        #
        # For each w_big and sector count S, compare:
        #  - Ceiling: truncated impulse-response low-k kernel (exact pipeline).
        #  - Learned: LOFO ridge on sectorized dipole features (Dx_{m,s}, Dy_{m,s}).
        #  - Projection: best-in-basis L2 projection of the true correlation-kernel onto the sectorized basis.
        import matplotlib.pyplot as plt

        grid_size = int(cfg.get("grid_size", 256))
        alpha = float(cfg.get("alpha", 2.0))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 10_000))
        k0_frac = float(cfg.get("k0_frac", 0.15))
        w_bigs = [_require_odd("w_big", int(w)) for w in cfg.get("w_bigs", [129, 193])]

        # Basis resolution: fixed M (radial bins), sweep S (angular sectors).
        M = int(cfg.get("M", 32))
        S_list = [int(s) for s in cfg.get("S_list", [1, 4, 8, 16])]
        if any(s <= 0 for s in S_list):
            raise ValueError("All S_list entries must be > 0")
        ring_bin_mode = str(cfg.get("ring_bin_mode", "uniform_r")).lower()
        ridge_alpha = float(cfg.get("ridge_alpha", 1.0))

        # Conclusion threshold (relative to ceiling): gapP > pearson_min and gapRelRMSE < relrmse_max.
        gap_pearson_min = float(cfg.get("gap_pearson_min", -0.003))
        gap_relrmse_max = float(cfg.get("gap_relrmse_max", 0.03))

        # Diagnostics: mask visualization and coefficient plots.
        diag_mask_wbig = _require_odd("diag_mask_wbig", int(cfg.get("diag_mask_wbig", 193)))
        diag_mask_M = int(cfg.get("diag_mask_M", 8))
        diag_mask_S = int(cfg.get("diag_mask_S", 8))
        coef_plot_wbig = _require_odd("coef_plot_wbig", int(cfg.get("coef_plot_wbig", 193)))
        coef_plot_S = int(cfg.get("coef_plot_S", max(S_list)))

        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if n_fields < 2:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        if not (0.0 < float(k0_frac) < 0.5):
            raise ValueError("k0_frac must be in (0,0.5)")
        if not w_bigs:
            raise ValueError("w_bigs must be non-empty")
        if any(w > grid_size for w in w_bigs):
            raise ValueError("all w_bigs must be <= grid_size")
        if M <= 0:
            raise ValueError("M must be > 0")
        if ring_bin_mode not in {"uniform_r", "equal_area"}:
            raise ValueError("ring_bin_mode must be 'uniform_r' or 'equal_area'")
        if ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be > 0")
        if diag_mask_M <= 0 or diag_mask_S <= 0:
            raise ValueError("diag_mask_M/diag_mask_S must be > 0")

        def safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            am = a - float(a.mean())
            bm = b - float(b.mean())
            denom = float(np.linalg.norm(am) * np.linalg.norm(bm)) + 1e-12
            return float((am @ bm) / denom)

        def relrmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
            e = y_pred - y_true
            rmse = float(np.sqrt(np.mean(e * e)))
            sd = float(np.std(y_true))
            return rmse / (sd + 1e-12)

        def pearson_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            px = safe_corr_1d(y_true[:, 0], y_pred[:, 0])
            py = safe_corr_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (px + py)

        def relrmse_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            rx = relrmse_1d(y_true[:, 0], y_pred[:, 0])
            ry = relrmse_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (rx + ry)

        def solve_ridge(XtX: np.ndarray, Xty: np.ndarray) -> np.ndarray:
            d = int(XtX.shape[0])
            I = np.eye(d, dtype=np.float64)
            try:
                return np.linalg.solve(XtX + ridge_alpha * I, Xty)
            except np.linalg.LinAlgError:
                w, *_ = np.linalg.lstsq(XtX + ridge_alpha * I, Xty, rcond=None)
                return w

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r0 in rows:
                out.append("| " + " | ".join(r0.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        def annulus_edges(r_min: float, r_max: float, n_bins: int) -> np.ndarray:
            if ring_bin_mode == "uniform_r":
                return np.linspace(r_min, r_max + 1e-9, n_bins + 1, dtype=np.float64)
            rsq = np.linspace(r_min * r_min, r_max * r_max + 1e-9, n_bins + 1, dtype=np.float64)
            return np.sqrt(rsq, dtype=np.float64)

        def kernel_fft_centered(kernel: np.ndarray, *, grid_size: int) -> np.ndarray:
            kernel = np.asarray(kernel, dtype=np.float64)
            if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
                raise ValueError(f"kernel must be square 2D, got {kernel.shape}")
            w = int(kernel.shape[0])
            if (w % 2) == 0:
                raise ValueError(f"kernel size must be odd, got {w}")
            if w > grid_size:
                raise ValueError(f"kernel size {w} exceeds grid_size {grid_size}")
            r = w // 2
            kr = np.flip(kernel, axis=(0, 1))  # convolution kernel so that output = correlation with `kernel`
            full = np.zeros((grid_size, grid_size), dtype=np.float64)
            cx = grid_size // 2
            cy = grid_size // 2
            full[cx - r : cx + r + 1, cy - r : cy + r + 1] = kr
            full0 = np.fft.ifftshift(full)
            return np.fft.fftn(full0)

        def save_kernel_img(arr: np.ndarray, out: Path, title: str) -> None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            vmax = float(np.max(np.abs(arr))) + 1e-12
            im = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)

        def build_ring_sector_index(
            wb: int, *, M: int, S: int
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            r_big = wb // 2
            coords = np.arange(int(wb), dtype=np.float64) - float(r_big)
            dx, dy = np.meshgrid(coords, coords, indexing="ij")
            rgrid = np.sqrt(dx * dx + dy * dy, dtype=np.float64)
            theta = np.arctan2(dy, dx)  # [-pi, pi]

            edges_r = annulus_edges(0.0, float(r_big), int(M))
            edges_th = np.linspace(-np.pi, np.pi, int(S) + 1, dtype=np.float64)

            ring_idx = np.digitize(rgrid, edges_r, right=False) - 1  # -1..M-1 (M means > r_big)
            ring_idx = ring_idx.astype(np.int32, copy=False)
            ring_idx = np.where((ring_idx >= 0) & (ring_idx < int(M)), ring_idx, -1).astype(np.int32, copy=False)

            sec_idx = np.digitize(theta, edges_th, right=False) - 1
            sec_idx = sec_idx.astype(np.int32, copy=False)
            sec_idx = np.where(sec_idx == int(S), int(S) - 1, sec_idx).astype(np.int32, copy=False)
            sec_idx = np.clip(sec_idx, 0, int(S) - 1)

            bin_idx = ring_idx * int(S) + sec_idx  # invalid pixels -> negative
            bin_idx = np.where(ring_idx >= 0, bin_idx, -1).astype(np.int32, copy=False)
            return dx.astype(np.float64), dy.astype(np.float64), rgrid, bin_idx, edges_r

        # Save a mask visualization to verify annulus+sector partitioning.
        if diag_mask_wbig in [int(w) for w in w_bigs]:
            dx_d, dy_d, rgrid_d, bin_idx_d, edges_r_d = build_ring_sector_index(
                int(diag_mask_wbig), M=int(diag_mask_M), S=int(diag_mask_S)
            )
            label = bin_idx_d.astype(np.float64)
            label[label < 0] = np.nan
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            im = ax.imshow(label, cmap="turbo", interpolation="nearest")
            ax.set_title(f"Annulus+sector bins (w_big={diag_mask_wbig}, M={diag_mask_M}, S={diag_mask_S})")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            out = paths.run_dir / f"mask_annulus_sector_wbig{diag_mask_wbig}_M{diag_mask_M}_S{diag_mask_S}.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)

        # Generate fields + band splits (shared across settings).
        rho01_by_field: list[np.ndarray] = []
        rho0_by_field: list[np.ndarray] = []
        field_fft_by_field: list[np.ndarray] = []
        split_cache: list[BandSplit2D] = []
        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
            split = band_split_poisson_2d(rho01, k0_frac=float(k0_frac))
            rho01_by_field.append(rho01.astype(np.float64, copy=False))
            rho0 = (rho01 - float(rho01.mean())).astype(np.float64, copy=False)
            rho0_by_field.append(rho0)
            field_fft_by_field.append(np.fft.fftn(rho01))
            split_cache.append(split)

        def sample_low_at_centers(split: BandSplit2D, centers: np.ndarray) -> np.ndarray:
            cx = centers[:, 0].astype(np.int64)
            cy = centers[:, 1].astype(np.int64)
            return np.column_stack([split.low.gx[cx, cy], split.low.gy[cx, cy]]).astype(np.float64, copy=False)

        # Impulse-response low-k kernel (full grid), used to compute ceiling + projection kernels.
        rho_delta = np.zeros((grid_size, grid_size), dtype=np.float64)
        c0 = grid_size // 2
        rho_delta[c0, c0] = 1.0
        split_delta = band_split_poisson_2d(rho_delta, k0_frac=float(k0_frac))
        kfull_gx = np.asarray(split_delta.low.gx, dtype=np.float64, copy=False)
        kfull_gy = np.asarray(split_delta.low.gy, dtype=np.float64, copy=False)
        assert_finite("kernel_full_gx", kfull_gx)
        assert_finite("kernel_full_gy", kfull_gy)
        save_kernel_img(kfull_gx, paths.run_dir / "kernel_low_full_gx.png", "Kx_low (full grid, impulse)")
        save_kernel_img(kfull_gy, paths.run_dir / "kernel_low_full_gy.png", "Ky_low (full grid, impulse)")

        rows_summary: list[dict[str, Any]] = []
        kernel_approx_rows: list[dict[str, Any]] = []
        coef_plot_lines: list[str] = []
        best_S_lines: list[str] = []

        for wb in w_bigs:
            r_big = int(wb) // 2
            B = int(M)  # radial bins

            # Patch centers (deterministic, per w_big).
            centers_by_field: list[np.ndarray] = []
            for field_id in range(n_fields):
                rng_cent = np.random.default_rng(seed + 555_555 + 10_000 * int(wb) + 1_000 * field_id)
                cx = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
                cy = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
                centers_by_field.append(np.column_stack([cx, cy]).astype(np.int64, copy=False))

            y_low_by_field = [sample_low_at_centers(split_cache[fid], centers_by_field[fid]) for fid in range(n_fields)]

            # True truncated low-k correlation-kernels (kcorr = flip(convolution-kernel patch)).
            g_patch_gx = kfull_gx[c0 - r_big : c0 + r_big + 1, c0 - r_big : c0 + r_big + 1]
            g_patch_gy = kfull_gy[c0 - r_big : c0 + r_big + 1, c0 - r_big : c0 + r_big + 1]
            kcorr_true_gx = g_patch_gx[::-1, ::-1].astype(np.float64, copy=False)
            kcorr_true_gy = g_patch_gy[::-1, ::-1].astype(np.float64, copy=False)
            save_kernel_img(g_patch_gx, paths.run_dir / f"kernel_low_gx_wbig{wb}.png", f"Kx_low truncated (w={wb})")
            save_kernel_img(g_patch_gy, paths.run_dir / f"kernel_low_gy_wbig{wb}.png", f"Ky_low truncated (w={wb})")

            # Ceiling metrics (no learning): correlation with kcorr_true on mean-subtracted rho.
            kfft_gx = kernel_fft_centered(kcorr_true_gx, grid_size=grid_size)
            kfft_gy = kernel_fft_centered(kcorr_true_gy, grid_size=grid_size)
            per_field_ceiling: list[dict[str, Any]] = []
            for fid in range(n_fields):
                field_fft = np.fft.fftn(rho0_by_field[fid])
                gx_pred = np.fft.ifftn(field_fft * kfft_gx).real
                gy_pred = np.fft.ifftn(field_fft * kfft_gy).real
                centers = centers_by_field[fid]
                cx = centers[:, 0].astype(np.int64)
                cy = centers[:, 1].astype(np.int64)
                y_pred = np.column_stack([gx_pred[cx, cy], gy_pred[cx, cy]]).astype(np.float64, copy=False)
                y_true = y_low_by_field[fid]
                per_field_ceiling.append(
                    {"field_id": int(fid), "pearson": pearson_mean_2d(y_true, y_pred), "relRMSE": relrmse_mean_2d(y_true, y_pred)}
                )
            ceiling_p = np.asarray([r0["pearson"] for r0 in per_field_ceiling], dtype=np.float64)
            ceiling_r = np.asarray([r0["relRMSE"] for r0 in per_field_ceiling], dtype=np.float64)
            ceiling_mean_p = float(ceiling_p.mean())
            ceiling_std_p = float(ceiling_p.std(ddof=1)) if len(ceiling_p) > 1 else 0.0
            ceiling_mean_r = float(ceiling_r.mean())
            ceiling_std_r = float(ceiling_r.std(ddof=1)) if len(ceiling_r) > 1 else 0.0

            for S in S_list:
                # Build bin index for this (w_big,M,S).
                dx, dy, rgrid, bin_idx, edges_r = build_ring_sector_index(int(wb), M=int(M), S=int(S))
                n_bins = int(M) * int(S)
                d = 2 * n_bins

                # Projection coefficients via disjoint-support closed form.
                bin_flat = bin_idx.reshape(-1)
                valid = bin_flat >= 0
                bin_valid = bin_flat[valid].astype(np.int64, copy=False)
                dx_flat = dx.reshape(-1)[valid].astype(np.float64, copy=False)
                dy_flat = dy.reshape(-1)[valid].astype(np.float64, copy=False)
                kx_flat = kcorr_true_gx.reshape(-1)[valid].astype(np.float64, copy=False)
                ky_flat = kcorr_true_gy.reshape(-1)[valid].astype(np.float64, copy=False)

                num_x = np.bincount(bin_valid, weights=kx_flat * dx_flat, minlength=n_bins).astype(np.float64, copy=False)
                den_x = np.bincount(bin_valid, weights=dx_flat * dx_flat, minlength=n_bins).astype(np.float64, copy=False)
                coef_x = np.where(den_x > 0, num_x / (den_x + 1e-12), 0.0).astype(np.float64, copy=False)

                num_y = np.bincount(bin_valid, weights=ky_flat * dy_flat, minlength=n_bins).astype(np.float64, copy=False)
                den_y = np.bincount(bin_valid, weights=dy_flat * dy_flat, minlength=n_bins).astype(np.float64, copy=False)
                coef_y = np.where(den_y > 0, num_y / (den_y + 1e-12), 0.0).astype(np.float64, copy=False)

                # Build projected kernels (kcorr space) for quality metrics/visuals.
                Kx_approx = np.zeros((int(wb), int(wb)), dtype=np.float64)
                Ky_approx = np.zeros((int(wb), int(wb)), dtype=np.float64)
                if valid.any():
                    Kx_approx.reshape(-1)[valid] = coef_x[bin_valid] * dx_flat
                    Ky_approx.reshape(-1)[valid] = coef_y[bin_valid] * dy_flat

                corr_kx = safe_corr_1d(kcorr_true_gx, Kx_approx)
                corr_ky = safe_corr_1d(kcorr_true_gy, Ky_approx)
                rel_kx = float(np.linalg.norm((Kx_approx - kcorr_true_gx).ravel()) / (np.linalg.norm(kcorr_true_gx.ravel()) + 1e-12))
                rel_ky = float(np.linalg.norm((Ky_approx - kcorr_true_gy).ravel()) / (np.linalg.norm(kcorr_true_gy.ravel()) + 1e-12))
                kernel_approx_rows.append(
                    {
                        "w_big": int(wb),
                        "M": int(M),
                        "S": int(S),
                        "corr_kernel_mean": float(0.5 * (corr_kx + corr_ky)),
                        "relL2_kernel_mean": float(0.5 * (rel_kx + rel_ky)),
                    }
                )

                # Build sectorized dipole features via FFT (correlations), storing float32 to limit memory.
                F_by_field: list[np.ndarray] = [np.zeros((patches_per_field, d), dtype=np.float32) for _ in range(n_fields)]

                # Precompute row/col indices per field for quick sampling.
                centers_idx = []
                for fid in range(n_fields):
                    centers = centers_by_field[fid]
                    centers_idx.append((centers[:, 0].astype(np.int64), centers[:, 1].astype(np.int64)))

                # Iterate over bins, compute Dx and Dy features.
                # Feature layout: [Dx bins..., Dy bins...].
                for b in range(n_bins):
                    mask = (bin_idx == int(b)).astype(np.float64, copy=False)
                    if not mask.any():
                        continue
                    kdx = (dx * mask).astype(np.float64, copy=False)
                    kdy = (dy * mask).astype(np.float64, copy=False)
                    kfft_dx = kernel_fft_centered(kdx, grid_size=grid_size)
                    kfft_dy = kernel_fft_centered(kdy, grid_size=grid_size)

                    for fid in range(n_fields):
                        field_fft = field_fft_by_field[fid]
                        cx, cy = centers_idx[fid]
                        dx_grid = np.fft.ifftn(field_fft * kfft_dx).real
                        dy_grid = np.fft.ifftn(field_fft * kfft_dy).real
                        F_by_field[fid][:, b] = dx_grid[cx, cy].astype(np.float32, copy=False)
                        F_by_field[fid][:, n_bins + b] = dy_grid[cx, cy].astype(np.float32, copy=False)

                # Projection performance (no learning): per-field metrics.
                per_field_proj: list[dict[str, Any]] = []
                for fid in range(n_fields):
                    F = F_by_field[fid].astype(np.float64, copy=False)
                    y_pred = np.column_stack([F[:, :n_bins] @ coef_x, F[:, n_bins:] @ coef_y]).astype(np.float64, copy=False)
                    y_true = y_low_by_field[fid]
                    per_field_proj.append(
                        {"field_id": int(fid), "pearson": pearson_mean_2d(y_true, y_pred), "relRMSE": relrmse_mean_2d(y_true, y_pred)}
                    )
                proj_p = np.asarray([r0["pearson"] for r0 in per_field_proj], dtype=np.float64)
                proj_r = np.asarray([r0["relRMSE"] for r0 in per_field_proj], dtype=np.float64)
                proj_mean_p = float(proj_p.mean())
                proj_std_p = float(proj_p.std(ddof=1)) if len(proj_p) > 1 else 0.0
                proj_mean_r = float(proj_r.mean())
                proj_std_r = float(proj_r.std(ddof=1)) if len(proj_r) > 1 else 0.0

                # Learned ridge (LOFO), using per-field sufficient statistics to avoid huge X_train matrices.
                sums_F: list[np.ndarray] = []
                sums_y: list[np.ndarray] = []
                FtF: list[np.ndarray] = []
                FtY: list[np.ndarray] = []
                for fid in range(n_fields):
                    F = F_by_field[fid].astype(np.float64, copy=False)
                    y = y_low_by_field[fid].astype(np.float64, copy=False)
                    sums_F.append(F.sum(axis=0))
                    sums_y.append(y.sum(axis=0))
                    FtF.append((F.T @ F).astype(np.float64, copy=False))
                    FtY.append((F.T @ y).astype(np.float64, copy=False))

                total_sum_F = np.sum(np.stack(sums_F, axis=0), axis=0)
                total_sum_y = np.sum(np.stack(sums_y, axis=0), axis=0)
                total_FtF = np.sum(np.stack(FtF, axis=0), axis=0)
                total_FtY = np.sum(np.stack(FtY, axis=0), axis=0)

                learned_fold: list[dict[str, Any]] = []
                learned_raw_coefs: list[np.ndarray] = []
                n_total = int(n_fields) * int(patches_per_field)

                for test_field in range(n_fields):
                    n_test = int(patches_per_field)
                    n_train = n_total - n_test
                    sum_F_tr = total_sum_F - sums_F[test_field]
                    sum_y_tr = total_sum_y - sums_y[test_field]
                    FtF_tr = total_FtF - FtF[test_field]
                    FtY_tr = total_FtY - FtY[test_field]

                    mu = sum_F_tr / float(n_train)
                    y_mu = sum_y_tr / float(n_train)

                    var = np.diag(FtF_tr) / float(n_train) - mu * mu
                    var = np.where(var > 1e-12, var, 1e-12)
                    sd = np.sqrt(var, dtype=np.float64)

                    FtF_center = FtF_tr - float(n_train) * np.outer(mu, mu)
                    FtY_center = FtY_tr - float(n_train) * (mu[:, None] * y_mu[None, :])

                    XtX = FtF_center / (sd[:, None] * sd[None, :])
                    Xty = FtY_center / sd[:, None]
                    w = solve_ridge(XtX, Xty)

                    # Predict on test field.
                    Fte = F_by_field[test_field].astype(np.float64, copy=False)
                    Xte = (Fte - mu) / sd
                    y_pred = Xte @ w + y_mu
                    y_true = y_low_by_field[test_field]
                    learned_fold.append(
                        {"field_id": int(test_field), "pearson": pearson_mean_2d(y_true, y_pred), "relRMSE": relrmse_mean_2d(y_true, y_pred)}
                    )

                    # Raw-scale coefficients for plotting (w_raw = w / sd).
                    if int(wb) == int(coef_plot_wbig) and int(S) == int(coef_plot_S):
                        learned_raw_coefs.append((w / sd[:, None]).astype(np.float64, copy=False))

                learn_p = np.asarray([r0["pearson"] for r0 in learned_fold], dtype=np.float64)
                learn_r = np.asarray([r0["relRMSE"] for r0 in learned_fold], dtype=np.float64)
                learn_mean_p = float(learn_p.mean())
                learn_std_p = float(learn_p.std(ddof=1)) if len(learn_p) > 1 else 0.0
                learn_mean_r = float(learn_r.mean())
                learn_std_r = float(learn_r.std(ddof=1)) if len(learn_r) > 1 else 0.0

                rows_summary.append(
                    {
                        "w_big": int(wb),
                        "M": int(M),
                        "S": int(S),
                        "ceiling_pearson_mean": float(ceiling_mean_p),
                        "ceiling_pearson_std": float(ceiling_std_p),
                        "ceiling_relRMSE_mean": float(ceiling_mean_r),
                        "ceiling_relRMSE_std": float(ceiling_std_r),
                        "learn_pearson_mean": float(learn_mean_p),
                        "learn_pearson_std": float(learn_std_p),
                        "learn_relRMSE_mean": float(learn_mean_r),
                        "learn_relRMSE_std": float(learn_std_r),
                        "proj_pearson_mean": float(proj_mean_p),
                        "proj_pearson_std": float(proj_std_p),
                        "proj_relRMSE_mean": float(proj_mean_r),
                        "proj_relRMSE_std": float(proj_std_r),
                    }
                )

                # Optional kernel visuals for the representative coefficient plot setting.
                if int(wb) == int(coef_plot_wbig) and int(S) == int(coef_plot_S):
                    save_kernel_img(
                        Kx_approx,
                        paths.run_dir / f"kernel_proj_approx_gx_wbig{wb}_M{M}_S{S}.png",
                        f"Projected Kcorr gx (w={wb}, M={M}, S={S})",
                    )
                    save_kernel_img(
                        Ky_approx,
                        paths.run_dir / f"kernel_proj_approx_gy_wbig{wb}_M{M}_S{S}.png",
                        f"Projected Kcorr gy (w={wb}, M={M}, S={S})",
                    )

                    # Coefficients vs radius for a subset of sectors.
                    r_mid = 0.5 * (edges_r[:-1] + edges_r[1:])
                    sectors_to_plot = [0, int(S) // 4, int(S) // 2, (3 * int(S)) // 4]
                    sectors_to_plot = sorted({int(s0) for s0 in sectors_to_plot if 0 <= int(s0) < int(S)})
                    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
                    for s0 in sectors_to_plot:
                        idx = np.arange(int(M), dtype=np.int64) * int(S) + int(s0)
                        ax.plot(r_mid, coef_x[idx], label=f"proj Dx->gx, sector {s0}")
                    if learned_raw_coefs:
                        W = np.stack(learned_raw_coefs, axis=0)  # (folds, d, 2)
                        w_dx_gx = W[:, :n_bins, 0].mean(axis=0)
                        for s0 in sectors_to_plot:
                            idx = np.arange(int(M), dtype=np.int64) * int(S) + int(s0)
                            ax.plot(r_mid, w_dx_gx[idx], "--", label=f"learn Dx->gx, sector {s0}")
                    ax.axhline(0.0, color="k", lw=1, alpha=0.3)
                    ax.set_xlabel("radius (annulus midpoint)")
                    ax.set_ylabel("coefficient (raw units)")
                    ax.set_title(f"Coefficients vs radius (w_big={wb}, M={M}, S={S})")
                    ax.legend(loc="best", fontsize=8, ncol=2)
                    fig.tight_layout()
                    out = paths.run_dir / f"coeffs_vs_radius_wbig{wb}_M{M}_S{S}.png"
                    fig.savefig(out, dpi=150)
                    plt.close(fig)
                    coef_plot_lines.append(f"- `{out.name}`")

            # Choose best S (smallest meeting threshold) for w_big, for learned+projection.
            rows_w = [r0 for r0 in rows_summary if int(r0["w_big"]) == int(wb)]
            rows_w = sorted(rows_w, key=lambda r0: int(r0["S"]))
            for which in ("learn", "proj"):
                ok = []
                for r0 in rows_w:
                    gp = float(r0[f"{which}_pearson_mean"]) - float(r0["ceiling_pearson_mean"])
                    gr = float(r0[f"{which}_relRMSE_mean"]) - float(r0["ceiling_relRMSE_mean"])
                    if (gp > gap_pearson_min) and (gr < gap_relrmse_max):
                        ok.append(int(r0["S"]))
                if ok:
                    best_S_lines.append(f"- w_big={int(wb)}: {which} meets targets at S={min(ok)}")
                else:
                    best_S_lines.append(f"- w_big={int(wb)}: {which} does not meet targets up to S={max(S_list)}")

        md_rows: list[dict[str, str]] = []
        for r0 in sorted(rows_summary, key=lambda x: (int(x["w_big"]), int(x["S"]))):
            cp = float(r0["ceiling_pearson_mean"])
            cr = float(r0["ceiling_relRMSE_mean"])
            lp = float(r0["learn_pearson_mean"])
            lr = float(r0["learn_relRMSE_mean"])
            pp = float(r0["proj_pearson_mean"])
            pr = float(r0["proj_relRMSE_mean"])
            md_rows.append(
                {
                    "w_big": str(r0["w_big"]),
                    "S": str(r0["S"]),
                    "Pearson_ceiling": f"{cp:.4f}",
                    "relRMSE_ceiling": f"{cr:.4f}",
                    "Pearson_learned": f"{lp:.4f} ± {float(r0['learn_pearson_std']):.4f}",
                    "relRMSE_learned": f"{lr:.4f} ± {float(r0['learn_relRMSE_std']):.4f}",
                    "Pearson_proj": f"{pp:.4f} ± {float(r0['proj_pearson_std']):.4f}",
                    "relRMSE_proj": f"{pr:.4f} ± {float(r0['proj_relRMSE_std']):.4f}",
                    "gapP_learn": f"{(lp - cp):+.4f}",
                    "gapR_learn": f"{(lr - cr):+.4f}",
                    "gapP_proj": f"{(pp - cp):+.4f}",
                    "gapR_proj": f"{(pr - cr):+.4f}",
                }
            )

        md_krows: list[dict[str, str]] = []
        for r0 in sorted(kernel_approx_rows, key=lambda x: (int(x["w_big"]), int(x["S"]))):
            md_krows.append(
                {
                    "w_big": str(r0["w_big"]),
                    "S": str(r0["S"]),
                    "corr(Kapprox,Ktrue)": f"{float(r0['corr_kernel_mean']):.4f}",
                    "relL2(Kapprox,Ktrue)": f"{float(r0['relL2_kernel_mean']):.4f}",
                }
            )

        summary_md = (
            "# E36 — Sectorized dipole basis (low-k)\n\n"
            f"- run: `{paths.run_dir}`\n"
            f"- grid_size={grid_size}, alpha={alpha}, k0_frac={k0_frac}, n_fields={n_fields}, patches_per_field={patches_per_field}\n"
            f"- w_bigs={w_bigs}, M={M}, S_list={S_list}, ring_bin_mode={ring_bin_mode}, ridge_alpha={ridge_alpha}\n"
            f"- target thresholds (relative to ceiling): gapP > {gap_pearson_min}, gapRelRMSE < {gap_relrmse_max}\n\n"
            "## Metrics vs ceiling\n\n"
            + md_table(
                md_rows,
                [
                    "w_big",
                    "S",
                    "Pearson_ceiling",
                    "relRMSE_ceiling",
                    "Pearson_learned",
                    "relRMSE_learned",
                    "Pearson_proj",
                    "relRMSE_proj",
                    "gapP_learn",
                    "gapR_learn",
                    "gapP_proj",
                    "gapR_proj",
                ],
            )
            + "\n\n## Kernel approximation quality (projection)\n\n"
            + md_table(md_krows, ["w_big", "S", "corr(Kapprox,Ktrue)", "relL2(Kapprox,Ktrue)"])
            + "\n\n## Best S under thresholds\n\n"
            + ("\n".join(best_S_lines) + "\n")
            + "\n## Diagnostics\n\n"
            + f"- `mask_annulus_sector_wbig{diag_mask_wbig}_M{diag_mask_M}_S{diag_mask_S}.png`\n"
            + "- `kernel_low_full_gx.png` / `kernel_low_full_gy.png`\n"
            + "\n"
            + ("\n".join(coef_plot_lines) + "\n" if coef_plot_lines else "")
        )

        (paths.run_dir / "summary_e36_sectorized_dipole_lowk.md").write_text(summary_md, encoding="utf-8")
        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "grid_size": grid_size,
                "alpha": alpha,
                "k0_frac": k0_frac,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "w_bigs": [int(w) for w in w_bigs],
                "M": int(M),
                "S_list": [int(s) for s in S_list],
                "ring_bin_mode": ring_bin_mode,
                "ridge_alpha": ridge_alpha,
                "thresholds": {"gap_pearson_min": gap_pearson_min, "gap_relrmse_max": gap_relrmse_max},
                "rows_summary": rows_summary,
                "kernel_approx_rows": kernel_approx_rows,
                "best_S_lines": best_S_lines,
            },
        )
        return paths

    if experiment == "e37":
        # E37 — Rectangular dipole basis dx * f(|dx|,|dy|) to close the low-k relRMSE gap.
        #
        # Compare (per w_big):
        #  - Ceiling: truncated impulse-response low-k kernel (exact pipeline).
        #  - Annulus dipole baseline: dx*f(r) with M=32 (learned + projection).
        #  - Rectangular dipoles: dx*f(|dx|,|dy|) (learned + projection), with a small bin sweep.
        import matplotlib.pyplot as plt

        grid_size = int(cfg.get("grid_size", 256))
        alpha = float(cfg.get("alpha", 2.0))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 10_000))
        k0_frac = float(cfg.get("k0_frac", 0.15))
        w_bigs = [_require_odd("w_big", int(w)) for w in cfg.get("w_bigs", [129, 193])]

        bin_settings = cfg.get("bin_settings", [(8, 4), (12, 6), (16, 8)])
        bin_settings = [(int(a), int(b)) for a, b in bin_settings]
        if any(nx <= 0 or ny <= 0 for nx, ny in bin_settings):
            raise ValueError("bin_settings must contain positive (Nx,Ny) pairs")

        # Baseline: annulus dipole (no sectors).
        annulus_M = int(cfg.get("annulus_M", 32))

        ring_bin_mode = str(cfg.get("ring_bin_mode", "uniform_r")).lower()
        ridge_alpha = float(cfg.get("ridge_alpha", 1.0))

        # Success criteria (projection, w_big=193).
        gap_pearson_min = float(cfg.get("gap_pearson_min", -0.003))
        gap_relrmse_max = float(cfg.get("gap_relrmse_max", 0.03))

        # Diagnostics.
        diag_wbig = _require_odd("diag_wbig", int(cfg.get("diag_wbig", 193)))
        diag_setting = cfg.get("diag_setting", (8, 4))
        diag_setting = (int(diag_setting[0]), int(diag_setting[1]))

        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if n_fields < 2:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        if not (0.0 < float(k0_frac) < 0.5):
            raise ValueError("k0_frac must be in (0,0.5)")
        if any(w > grid_size for w in w_bigs):
            raise ValueError("all w_bigs must be <= grid_size")
        if annulus_M <= 0:
            raise ValueError("annulus_M must be > 0")
        if ring_bin_mode not in {"uniform_r", "equal_area"}:
            raise ValueError("ring_bin_mode must be 'uniform_r' or 'equal_area'")
        if ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be > 0")

        def safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            am = a - float(a.mean())
            bm = b - float(b.mean())
            denom = float(np.linalg.norm(am) * np.linalg.norm(bm)) + 1e-12
            return float((am @ bm) / denom)

        def relrmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
            e = y_pred - y_true
            rmse = float(np.sqrt(np.mean(e * e)))
            sd = float(np.std(y_true))
            return rmse / (sd + 1e-12)

        def pearson_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            px = safe_corr_1d(y_true[:, 0], y_pred[:, 0])
            py = safe_corr_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (px + py)

        def relrmse_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            rx = relrmse_1d(y_true[:, 0], y_pred[:, 0])
            ry = relrmse_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (rx + ry)

        def solve_ridge(XtX: np.ndarray, Xty: np.ndarray) -> np.ndarray:
            d = int(XtX.shape[0])
            I = np.eye(d, dtype=np.float64)
            try:
                return np.linalg.solve(XtX + ridge_alpha * I, Xty)
            except np.linalg.LinAlgError:
                w, *_ = np.linalg.lstsq(XtX + ridge_alpha * I, Xty, rcond=None)
                return w

        def annulus_edges(r_min: float, r_max: float, n_bins: int) -> np.ndarray:
            if ring_bin_mode == "uniform_r":
                return np.linspace(r_min, r_max + 1e-9, n_bins + 1, dtype=np.float64)
            rsq = np.linspace(r_min * r_min, r_max * r_max + 1e-9, n_bins + 1, dtype=np.float64)
            return np.sqrt(rsq, dtype=np.float64)

        def kernel_fft_centered(kernel: np.ndarray, *, grid_size: int) -> np.ndarray:
            kernel = np.asarray(kernel, dtype=np.float64)
            if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
                raise ValueError(f"kernel must be square 2D, got {kernel.shape}")
            w = int(kernel.shape[0])
            if (w % 2) == 0:
                raise ValueError(f"kernel size must be odd, got {w}")
            if w > grid_size:
                raise ValueError(f"kernel size {w} exceeds grid_size {grid_size}")
            r = w // 2
            kr = np.flip(kernel, axis=(0, 1))  # makes convolution(field, kr) == correlation(field, kernel)
            full = np.zeros((grid_size, grid_size), dtype=np.float64)
            cx = grid_size // 2
            cy = grid_size // 2
            full[cx - r : cx + r + 1, cy - r : cy + r + 1] = kr
            full0 = np.fft.ifftshift(full)
            return np.fft.fftn(full0)

        def save_kernel_img(arr: np.ndarray, out: Path, title: str) -> None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            vmax = float(np.max(np.abs(arr))) + 1e-12
            im = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r0 in rows:
                out.append("| " + " | ".join(r0.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        def candidate_edges(r: int) -> np.ndarray:
            # Deterministic, dense-near-0 edge candidates, always includes 0 and r+1.
            base = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, r + 1]
            base = [int(x) for x in base if 0 <= int(x) <= r + 1]
            dense = list(range(0, min(16, r + 1) + 1))
            geom = np.unique(np.rint(np.geomspace(1, r + 1, num=64)).astype(int)).tolist()
            vals = sorted(set(base + dense + geom + [0, r + 1]))
            return np.asarray(vals, dtype=np.int64)

        def pick_edges(cand: np.ndarray, n_bins: int) -> np.ndarray:
            cand = np.asarray(cand, dtype=np.int64)
            if cand.ndim != 1:
                raise ValueError("cand must be 1D")
            if len(cand) < n_bins + 1:
                # Fallback: use all integer edges.
                cand = np.arange(int(cand.min()), int(cand.max()) + 1, dtype=np.int64)
            idx = np.round(np.linspace(0, len(cand) - 1, n_bins + 1)).astype(int)
            idx[0] = 0
            idx[-1] = len(cand) - 1
            for k in range(1, len(idx)):
                if idx[k] <= idx[k - 1]:
                    idx[k] = idx[k - 1] + 1
            for k in range(len(idx) - 2, -1, -1):
                if idx[k] >= idx[k + 1]:
                    idx[k] = idx[k + 1] - 1
            if idx[0] != 0 or idx[-1] != len(cand) - 1 or np.any(np.diff(idx) <= 0):
                raise RuntimeError("Failed to build strictly increasing edge indices")
            edges = cand[idx].astype(np.int64, copy=False)
            return edges

        def build_rect_bins(
            wb: int, *, nx_bins: int, ny_bins: int
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            r_big = wb // 2
            coords = np.arange(int(wb), dtype=np.float64) - float(r_big)
            dx, dy = np.meshgrid(coords, coords, indexing="ij")
            ax = np.abs(dx)
            ay = np.abs(dy)

            cand = candidate_edges(r_big)
            edges_x = pick_edges(cand, int(nx_bins)).astype(np.float64, copy=False)
            edges_y = pick_edges(cand, int(ny_bins)).astype(np.float64, copy=False)

            # ix in [0..nx_bins-1], iy in [0..ny_bins-1]
            ix = np.digitize(ax, edges_x, right=False) - 1
            iy = np.digitize(ay, edges_y, right=False) - 1
            ix = np.clip(ix, 0, int(nx_bins) - 1).astype(np.int32, copy=False)
            iy = np.clip(iy, 0, int(ny_bins) - 1).astype(np.int32, copy=False)

            bin_idx = (ix * int(ny_bins) + iy).astype(np.int32, copy=False)
            return dx.astype(np.float64), dy.astype(np.float64), bin_idx, edges_x.astype(np.float64), edges_y.astype(np.float64)

        def lofo_ridge_from_fields(F_by_field: list[np.ndarray], y_by_field: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            # Learned ridge with StandardScaler-style normalization, using per-field sufficient statistics.
            sums_F: list[np.ndarray] = []
            sums_y: list[np.ndarray] = []
            FtF: list[np.ndarray] = []
            FtY: list[np.ndarray] = []
            for fid in range(n_fields):
                F = np.asarray(F_by_field[fid], dtype=np.float64)
                y = np.asarray(y_by_field[fid], dtype=np.float64)
                sums_F.append(F.sum(axis=0))
                sums_y.append(y.sum(axis=0))
                FtF.append(F.T @ F)
                FtY.append(F.T @ y)

            total_sum_F = np.sum(np.stack(sums_F, axis=0), axis=0)
            total_sum_y = np.sum(np.stack(sums_y, axis=0), axis=0)
            total_FtF = np.sum(np.stack(FtF, axis=0), axis=0)
            total_FtY = np.sum(np.stack(FtY, axis=0), axis=0)

            n_total = int(n_fields) * int(patches_per_field)
            fold_p = np.zeros((n_fields,), dtype=np.float64)
            fold_r = np.zeros((n_fields,), dtype=np.float64)

            for test_field in range(n_fields):
                n_test = int(patches_per_field)
                n_train = n_total - n_test
                sum_F_tr = total_sum_F - sums_F[test_field]
                sum_y_tr = total_sum_y - sums_y[test_field]
                FtF_tr = total_FtF - FtF[test_field]
                FtY_tr = total_FtY - FtY[test_field]

                mu = sum_F_tr / float(n_train)
                y_mu = sum_y_tr / float(n_train)

                var = np.diag(FtF_tr) / float(n_train) - mu * mu
                var = np.where(var > 1e-12, var, 1e-12)
                sd = np.sqrt(var, dtype=np.float64)

                FtF_center = FtF_tr - float(n_train) * np.outer(mu, mu)
                FtY_center = FtY_tr - float(n_train) * (mu[:, None] * y_mu[None, :])

                XtX = FtF_center / (sd[:, None] * sd[None, :])
                Xty = FtY_center / sd[:, None]
                w = solve_ridge(XtX, Xty)

                Fte = np.asarray(F_by_field[test_field], dtype=np.float64)
                Xte = (Fte - mu) / sd
                y_pred = Xte @ w + y_mu
                y_true = np.asarray(y_by_field[test_field], dtype=np.float64)
                fold_p[test_field] = pearson_mean_2d(y_true, y_pred)
                fold_r[test_field] = relrmse_mean_2d(y_true, y_pred)
            return fold_p, fold_r

        # Generate fields + band split targets.
        rho01_by_field: list[np.ndarray] = []
        rho0_by_field: list[np.ndarray] = []
        field_fft_by_field: list[np.ndarray] = []
        split_cache: list[BandSplit2D] = []
        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
            split = band_split_poisson_2d(rho01, k0_frac=float(k0_frac))
            rho01_by_field.append(rho01.astype(np.float64, copy=False))
            rho0 = (rho01 - float(rho01.mean())).astype(np.float64, copy=False)
            rho0_by_field.append(rho0)
            field_fft_by_field.append(np.fft.fftn(rho01))
            split_cache.append(split)

        def sample_low_at_centers(split: BandSplit2D, centers: np.ndarray) -> np.ndarray:
            cx = centers[:, 0].astype(np.int64)
            cy = centers[:, 1].astype(np.int64)
            return np.column_stack([split.low.gx[cx, cy], split.low.gy[cx, cy]]).astype(np.float64, copy=False)

        # Impulse-response low-k kernel (full grid).
        rho_delta = np.zeros((grid_size, grid_size), dtype=np.float64)
        c0 = grid_size // 2
        rho_delta[c0, c0] = 1.0
        split_delta = band_split_poisson_2d(rho_delta, k0_frac=float(k0_frac))
        kfull_gx = np.asarray(split_delta.low.gx, dtype=np.float64, copy=False)
        kfull_gy = np.asarray(split_delta.low.gy, dtype=np.float64, copy=False)
        save_kernel_img(kfull_gx, paths.run_dir / "kernel_low_full_gx.png", "Kx_low (full grid, impulse)")
        save_kernel_img(kfull_gy, paths.run_dir / "kernel_low_full_gy.png", "Ky_low (full grid, impulse)")

        # Summary storage.
        rows_metrics: list[dict[str, Any]] = []
        rows_kapprox: list[dict[str, Any]] = []

        # Diagnostics outputs (only once).
        diag_done_mask = False
        diag_done_kernels = False

        for wb in w_bigs:
            r_big = int(wb) // 2

            # Deterministic centers per w_big.
            centers_by_field: list[np.ndarray] = []
            centers_idx: list[tuple[np.ndarray, np.ndarray]] = []
            for field_id in range(n_fields):
                rng_cent = np.random.default_rng(seed + 555_555 + 10_000 * int(wb) + 1_000 * field_id)
                cx = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
                cy = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
                centers = np.column_stack([cx, cy]).astype(np.int64, copy=False)
                centers_by_field.append(centers)
                centers_idx.append((cx.astype(np.int64, copy=False), cy.astype(np.int64, copy=False)))

            y_low_by_field = [sample_low_at_centers(split_cache[fid], centers_by_field[fid]) for fid in range(n_fields)]

            # Truncated impulse kernels and their correlation-kernel form.
            g_patch_gx = kfull_gx[c0 - r_big : c0 + r_big + 1, c0 - r_big : c0 + r_big + 1]
            g_patch_gy = kfull_gy[c0 - r_big : c0 + r_big + 1, c0 - r_big : c0 + r_big + 1]
            kcorr_true_gx = g_patch_gx[::-1, ::-1].astype(np.float64, copy=False)
            kcorr_true_gy = g_patch_gy[::-1, ::-1].astype(np.float64, copy=False)
            save_kernel_img(g_patch_gx, paths.run_dir / f"kernel_low_gx_wbig{wb}.png", f"Kx_low truncated (w={wb})")
            save_kernel_img(g_patch_gy, paths.run_dir / f"kernel_low_gy_wbig{wb}.png", f"Ky_low truncated (w={wb})")

            # Ceiling (no learning): convolve mean-subtracted rho with kernel.
            kfft_gx = kernel_fft_centered(kcorr_true_gx, grid_size=grid_size)
            kfft_gy = kernel_fft_centered(kcorr_true_gy, grid_size=grid_size)
            ceil_fold_p = np.zeros((n_fields,), dtype=np.float64)
            ceil_fold_r = np.zeros((n_fields,), dtype=np.float64)
            for fid in range(n_fields):
                field_fft = np.fft.fftn(rho0_by_field[fid])
                gx_pred = np.fft.ifftn(field_fft * kfft_gx).real
                gy_pred = np.fft.ifftn(field_fft * kfft_gy).real
                cx, cy = centers_idx[fid]
                y_pred = np.column_stack([gx_pred[cx, cy], gy_pred[cx, cy]]).astype(np.float64, copy=False)
                y_true = y_low_by_field[fid]
                ceil_fold_p[fid] = pearson_mean_2d(y_true, y_pred)
                ceil_fold_r[fid] = relrmse_mean_2d(y_true, y_pred)
            ceil_mean_p = float(ceil_fold_p.mean())
            ceil_std_p = float(ceil_fold_p.std(ddof=1)) if len(ceil_fold_p) > 1 else 0.0
            ceil_mean_r = float(ceil_fold_r.mean())
            ceil_std_r = float(ceil_fold_r.std(ddof=1)) if len(ceil_fold_r) > 1 else 0.0

            # Baseline annulus dipole (M=32): projection + learned.
            coords = np.arange(int(wb), dtype=np.float64) - float(r_big)
            dx, dy = np.meshgrid(coords, coords, indexing="ij")
            rgrid = np.sqrt(dx * dx + dy * dy, dtype=np.float64)
            edges_r = annulus_edges(0.0, float(r_big), int(annulus_M))
            n_bins_base = int(annulus_M)

            coef_base_x = np.zeros((n_bins_base,), dtype=np.float64)
            coef_base_y = np.zeros((n_bins_base,), dtype=np.float64)
            Fbase_by_field: list[np.ndarray] = [np.zeros((patches_per_field, 2 * n_bins_base), dtype=np.float32) for _ in range(n_fields)]

            for m in range(n_bins_base):
                lo = edges_r[m]
                hi = edges_r[m + 1]
                if m == n_bins_base - 1:
                    ring = (rgrid >= lo) & (rgrid <= hi)
                else:
                    ring = (rgrid >= lo) & (rgrid < hi)
                ring_f = ring.astype(np.float64)
                bx = (dx * ring_f).astype(np.float64, copy=False)
                by = (dy * ring_f).astype(np.float64, copy=False)
                denom_x = float(np.sum(bx * bx))
                denom_y = float(np.sum(by * by))
                coef_base_x[m] = float(np.sum(kcorr_true_gx * bx)) / (denom_x + 1e-12) if denom_x > 0 else 0.0
                coef_base_y[m] = float(np.sum(kcorr_true_gy * by)) / (denom_y + 1e-12) if denom_y > 0 else 0.0

                kfft_dx = kernel_fft_centered(bx, grid_size=grid_size)
                kfft_dy = kernel_fft_centered(by, grid_size=grid_size)
                for fid in range(n_fields):
                    field_fft = field_fft_by_field[fid]
                    cx, cy = centers_idx[fid]
                    dx_grid = np.fft.ifftn(field_fft * kfft_dx).real
                    dy_grid = np.fft.ifftn(field_fft * kfft_dy).real
                    Fbase_by_field[fid][:, m] = dx_grid[cx, cy].astype(np.float32, copy=False)
                    Fbase_by_field[fid][:, n_bins_base + m] = dy_grid[cx, cy].astype(np.float32, copy=False)

            # Baseline projection metrics (no learning).
            base_proj_p = np.zeros((n_fields,), dtype=np.float64)
            base_proj_r = np.zeros((n_fields,), dtype=np.float64)
            for fid in range(n_fields):
                F = Fbase_by_field[fid].astype(np.float64, copy=False)
                y_pred = np.column_stack([F[:, :n_bins_base] @ coef_base_x, F[:, n_bins_base:] @ coef_base_y]).astype(np.float64, copy=False)
                y_true = y_low_by_field[fid]
                base_proj_p[fid] = pearson_mean_2d(y_true, y_pred)
                base_proj_r[fid] = relrmse_mean_2d(y_true, y_pred)
            base_proj_mean_p = float(base_proj_p.mean())
            base_proj_std_p = float(base_proj_p.std(ddof=1)) if len(base_proj_p) > 1 else 0.0
            base_proj_mean_r = float(base_proj_r.mean())
            base_proj_std_r = float(base_proj_r.std(ddof=1)) if len(base_proj_r) > 1 else 0.0

            # Baseline learned (LOFO ridge).
            base_learn_p, base_learn_r = lofo_ridge_from_fields(Fbase_by_field, y_low_by_field)
            base_learn_mean_p = float(base_learn_p.mean())
            base_learn_std_p = float(base_learn_p.std(ddof=1)) if len(base_learn_p) > 1 else 0.0
            base_learn_mean_r = float(base_learn_r.mean())
            base_learn_std_r = float(base_learn_r.std(ddof=1)) if len(base_learn_r) > 1 else 0.0

            # Record baseline rows.
            rows_metrics.append(
                {
                    "w_big": int(wb),
                    "method": "ceiling",
                    "setting": "-",
                    "d": 0,
                    "pearson_mean": ceil_mean_p,
                    "pearson_std": ceil_std_p,
                    "relRMSE_mean": ceil_mean_r,
                    "relRMSE_std": ceil_std_r,
                    "gapP": 0.0,
                    "gapR": 0.0,
                }
            )
            rows_metrics.append(
                {
                    "w_big": int(wb),
                    "method": "annulus_learn",
                    "setting": f"M={annulus_M}",
                    "d": int(2 * n_bins_base),
                    "pearson_mean": base_learn_mean_p,
                    "pearson_std": base_learn_std_p,
                    "relRMSE_mean": base_learn_mean_r,
                    "relRMSE_std": base_learn_std_r,
                    "gapP": float(base_learn_mean_p - ceil_mean_p),
                    "gapR": float(base_learn_mean_r - ceil_mean_r),
                }
            )
            rows_metrics.append(
                {
                    "w_big": int(wb),
                    "method": "annulus_proj",
                    "setting": f"M={annulus_M}",
                    "d": int(2 * n_bins_base),
                    "pearson_mean": base_proj_mean_p,
                    "pearson_std": base_proj_std_p,
                    "relRMSE_mean": base_proj_mean_r,
                    "relRMSE_std": base_proj_std_r,
                    "gapP": float(base_proj_mean_p - ceil_mean_p),
                    "gapR": float(base_proj_mean_r - ceil_mean_r),
                }
            )

            # Rectangular bin settings.
            for nx_bins, ny_bins in bin_settings:
                dxr, dyr, bin_idx, edges_x, edges_y = build_rect_bins(int(wb), nx_bins=int(nx_bins), ny_bins=int(ny_bins))
                n_bins = int(nx_bins) * int(ny_bins)
                d = 2 * n_bins

                # Diagnostics: show bin regions (smallest setting).
                if (int(wb) == int(diag_wbig)) and ((int(nx_bins), int(ny_bins)) == diag_setting) and not diag_done_mask:
                    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                    ax.imshow(bin_idx.astype(np.float64), cmap="turbo", interpolation="nearest")
                    ax.set_title(f"Rect bins (w_big={wb}, Nx={nx_bins}, Ny={ny_bins})")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    fig.tight_layout()
                    out = paths.run_dir / f"mask_rectbins_wbig{wb}_Nx{nx_bins}_Ny{ny_bins}.png"
                    fig.savefig(out, dpi=150)
                    plt.close(fig)
                    diag_done_mask = True

                # Projection coefficients (disjoint masks).
                bin_flat = bin_idx.reshape(-1).astype(np.int64, copy=False)
                dx_flat = dxr.reshape(-1).astype(np.float64, copy=False)
                dy_flat = dyr.reshape(-1).astype(np.float64, copy=False)
                kx_flat = kcorr_true_gx.reshape(-1).astype(np.float64, copy=False)
                ky_flat = kcorr_true_gy.reshape(-1).astype(np.float64, copy=False)

                num_x = np.bincount(bin_flat, weights=kx_flat * dx_flat, minlength=n_bins).astype(np.float64, copy=False)
                den_x = np.bincount(bin_flat, weights=dx_flat * dx_flat, minlength=n_bins).astype(np.float64, copy=False)
                coef_x = np.where(den_x > 0, num_x / (den_x + 1e-12), 0.0).astype(np.float64, copy=False)

                num_y = np.bincount(bin_flat, weights=ky_flat * dy_flat, minlength=n_bins).astype(np.float64, copy=False)
                den_y = np.bincount(bin_flat, weights=dy_flat * dy_flat, minlength=n_bins).astype(np.float64, copy=False)
                coef_y = np.where(den_y > 0, num_y / (den_y + 1e-12), 0.0).astype(np.float64, copy=False)

                Kx_approx = coef_x[bin_idx] * dxr
                Ky_approx = coef_y[bin_idx] * dyr

                corr_kx = safe_corr_1d(kcorr_true_gx, Kx_approx)
                corr_ky = safe_corr_1d(kcorr_true_gy, Ky_approx)
                rel_kx = float(np.linalg.norm((Kx_approx - kcorr_true_gx).ravel()) / (np.linalg.norm(kcorr_true_gx.ravel()) + 1e-12))
                rel_ky = float(np.linalg.norm((Ky_approx - kcorr_true_gy).ravel()) / (np.linalg.norm(kcorr_true_gy.ravel()) + 1e-12))
                rows_kapprox.append(
                    {
                        "w_big": int(wb),
                        "Nx": int(nx_bins),
                        "Ny": int(ny_bins),
                        "d": int(d),
                        "corr_kernel_mean": float(0.5 * (corr_kx + corr_ky)),
                        "relL2_kernel_mean": float(0.5 * (rel_kx + rel_ky)),
                    }
                )

                # Diagnostics: Kx_true vs Kx_approx (smallest setting at w_big=193).
                if (int(wb) == int(diag_wbig)) and ((int(nx_bins), int(ny_bins)) == diag_setting) and not diag_done_kernels:
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    vmax = float(np.max(np.abs(kcorr_true_gx))) + 1e-12
                    axes[0].imshow(kcorr_true_gx, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
                    axes[0].set_title("Kx_true (kcorr)")
                    axes[1].imshow(Kx_approx, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
                    axes[1].set_title("Kx_approx (proj)")
                    diff = Kx_approx - kcorr_true_gx
                    vmax_d = float(np.max(np.abs(diff))) + 1e-12
                    axes[2].imshow(diff, cmap="RdBu_r", vmin=-vmax_d, vmax=vmax_d, interpolation="nearest")
                    axes[2].set_title("diff")
                    for ax in axes:
                        ax.set_xticks([])
                        ax.set_yticks([])
                    fig.tight_layout()
                    out = paths.run_dir / f"kernel_rectproj_compare_wbig{wb}_Nx{nx_bins}_Ny{ny_bins}_Kx.png"
                    fig.savefig(out, dpi=150)
                    plt.close(fig)
                    diag_done_kernels = True

                # Build features for learned/proj evaluation: Dx_{i,j}, Dy_{i,j}.
                F_by_field: list[np.ndarray] = [np.zeros((patches_per_field, d), dtype=np.float32) for _ in range(n_fields)]
                for b in range(n_bins):
                    mask = (bin_idx == int(b)).astype(np.float64, copy=False)
                    kdx = (dxr * mask).astype(np.float64, copy=False)
                    kdy = (dyr * mask).astype(np.float64, copy=False)
                    kfft_dx = kernel_fft_centered(kdx, grid_size=grid_size)
                    kfft_dy = kernel_fft_centered(kdy, grid_size=grid_size)
                    for fid in range(n_fields):
                        field_fft = field_fft_by_field[fid]
                        cx, cy = centers_idx[fid]
                        dx_grid = np.fft.ifftn(field_fft * kfft_dx).real
                        dy_grid = np.fft.ifftn(field_fft * kfft_dy).real
                        F_by_field[fid][:, b] = dx_grid[cx, cy].astype(np.float32, copy=False)
                        F_by_field[fid][:, n_bins + b] = dy_grid[cx, cy].astype(np.float32, copy=False)

                # Projection performance (no learning).
                proj_p = np.zeros((n_fields,), dtype=np.float64)
                proj_r = np.zeros((n_fields,), dtype=np.float64)
                for fid in range(n_fields):
                    F = F_by_field[fid].astype(np.float64, copy=False)
                    y_pred = np.column_stack([F[:, :n_bins] @ coef_x, F[:, n_bins:] @ coef_y]).astype(np.float64, copy=False)
                    y_true = y_low_by_field[fid]
                    proj_p[fid] = pearson_mean_2d(y_true, y_pred)
                    proj_r[fid] = relrmse_mean_2d(y_true, y_pred)
                proj_mean_p = float(proj_p.mean())
                proj_std_p = float(proj_p.std(ddof=1)) if len(proj_p) > 1 else 0.0
                proj_mean_r = float(proj_r.mean())
                proj_std_r = float(proj_r.std(ddof=1)) if len(proj_r) > 1 else 0.0

                # Learned performance (LOFO ridge).
                learn_p, learn_r = lofo_ridge_from_fields(F_by_field, y_low_by_field)
                learn_mean_p = float(learn_p.mean())
                learn_std_p = float(learn_p.std(ddof=1)) if len(learn_p) > 1 else 0.0
                learn_mean_r = float(learn_r.mean())
                learn_std_r = float(learn_r.std(ddof=1)) if len(learn_r) > 1 else 0.0

                rows_metrics.append(
                    {
                        "w_big": int(wb),
                        "method": "rect_learn",
                        "setting": f"Nx={nx_bins},Ny={ny_bins}",
                        "d": int(d),
                        "pearson_mean": learn_mean_p,
                        "pearson_std": learn_std_p,
                        "relRMSE_mean": learn_mean_r,
                        "relRMSE_std": learn_std_r,
                        "gapP": float(learn_mean_p - ceil_mean_p),
                        "gapR": float(learn_mean_r - ceil_mean_r),
                        "edges_x": edges_x.tolist(),
                        "edges_y": edges_y.tolist(),
                    }
                )
                rows_metrics.append(
                    {
                        "w_big": int(wb),
                        "method": "rect_proj",
                        "setting": f"Nx={nx_bins},Ny={ny_bins}",
                        "d": int(d),
                        "pearson_mean": proj_mean_p,
                        "pearson_std": proj_std_p,
                        "relRMSE_mean": proj_mean_r,
                        "relRMSE_std": proj_std_r,
                        "gapP": float(proj_mean_p - ceil_mean_p),
                        "gapR": float(proj_mean_r - ceil_mean_r),
                        "edges_x": edges_x.tolist(),
                        "edges_y": edges_y.tolist(),
                    }
                )

        # Render markdown summary.
        md_by_wbig: dict[int, list[dict[str, str]]] = {}
        for r0 in rows_metrics:
            wb = int(r0["w_big"])
            if wb not in md_by_wbig:
                md_by_wbig[wb] = []
            md_by_wbig[wb].append(
                {
                    "method": str(r0["method"]),
                    "setting": str(r0["setting"]),
                    "d": str(int(r0.get("d", 0))),
                    "Pearson mean±std": (
                        f"{float(r0['pearson_mean']):.4f} ± {float(r0['pearson_std']):.4f}"
                        if r0["method"] != "ceiling"
                        else f"{float(r0['pearson_mean']):.4f} ± {float(r0['pearson_std']):.4f}"
                    ),
                    "relRMSE mean±std": f"{float(r0['relRMSE_mean']):.4f} ± {float(r0['relRMSE_std']):.4f}",
                    "gapP vs ceiling": f"{float(r0['gapP']):+.4f}",
                    "gapR vs ceiling": f"{float(r0['gapR']):+.4f}",
                }
            )

        md_krows: list[dict[str, str]] = []
        for r0 in sorted(rows_kapprox, key=lambda x: (int(x["w_big"]), int(x["d"]))):
            md_krows.append(
                {
                    "w_big": str(r0["w_big"]),
                    "Nx": str(r0["Nx"]),
                    "Ny": str(r0["Ny"]),
                    "d": str(r0["d"]),
                    "corr(Kapprox,Ktrue)": f"{float(r0['corr_kernel_mean']):.4f}",
                    "relL2(Kapprox,Ktrue)": f"{float(r0['relL2_kernel_mean']):.4f}",
                }
            )

        # Success check for w_big=193 (projection).
        success_lines: list[str] = []
        for wb in [w for w in w_bigs if int(w) == 193]:
            wb = int(wb)
            proj_rows = [r0 for r0 in rows_metrics if int(r0["w_big"]) == wb and r0["method"] == "rect_proj"]
            ok = []
            for r0 in proj_rows:
                if (float(r0["gapP"]) > gap_pearson_min) and (float(r0["gapR"]) < gap_relrmse_max):
                    ok.append(r0["setting"])
            if ok:
                success_lines.append(f"- w_big=193: projection PASS at {ok[0]}")
            else:
                success_lines.append("- w_big=193: projection PASS at (none)")

        summary_parts = [
            "# E37 — Rectangular dipole basis (low-k)\n",
            f"- run: `{paths.run_dir}`\n",
            f"- grid_size={grid_size}, alpha={alpha}, k0_frac={k0_frac}, n_fields={n_fields}, patches_per_field={patches_per_field}\n",
            f"- w_bigs={w_bigs}\n",
            f"- annulus baseline: M={annulus_M}\n",
            f"- rect bin_settings={bin_settings}\n",
            f"- success criteria (w_big=193 proj): gapP > {gap_pearson_min}, gapR < {gap_relrmse_max}\n\n",
        ]
        for wb in sorted(md_by_wbig.keys()):
            rows = md_by_wbig[wb]
            order = {"ceiling": 0, "annulus_learn": 1, "annulus_proj": 2, "rect_learn": 3, "rect_proj": 4}
            rows = sorted(rows, key=lambda r: (order.get(r["method"], 99), r["setting"]))
            summary_parts.append(f"## w_big={wb}\n\n")
            summary_parts.append(
                md_table(rows, ["method", "setting", "d", "Pearson mean±std", "relRMSE mean±std", "gapP vs ceiling", "gapR vs ceiling"])
            )
            summary_parts.append("\n\n")

        summary_parts.append("## Kernel approximation quality (projection)\n\n")
        summary_parts.append(md_table(md_krows, ["w_big", "Nx", "Ny", "d", "corr(Kapprox,Ktrue)", "relL2(Kapprox,Ktrue)"]))
        summary_parts.append("\n\n## PASS check\n\n")
        summary_parts.append("\n".join(success_lines) + "\n")
        summary_parts.append("\n## Diagnostics\n\n")
        summary_parts.append(f"- `mask_rectbins_wbig{diag_wbig}_Nx{diag_setting[0]}_Ny{diag_setting[1]}.png`\n")
        summary_parts.append(f"- `kernel_rectproj_compare_wbig{diag_wbig}_Nx{diag_setting[0]}_Ny{diag_setting[1]}_Kx.png`\n")

        summary_md = "".join(summary_parts)
        (paths.run_dir / "summary_e37_rect_dipole_lowk.md").write_text(summary_md, encoding="utf-8")

        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "grid_size": grid_size,
                "alpha": alpha,
                "k0_frac": k0_frac,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "w_bigs": [int(w) for w in w_bigs],
                "annulus_M": int(annulus_M),
                "ring_bin_mode": ring_bin_mode,
                "ridge_alpha": ridge_alpha,
                "bin_settings": [(int(a), int(b)) for a, b in bin_settings],
                "success_criteria": {"gap_pearson_min": gap_pearson_min, "gap_relrmse_max": gap_relrmse_max},
                "rows_metrics": rows_metrics,
                "rows_kapprox": rows_kapprox,
            },
        )
        return paths

    if experiment == "e38":
        # E38 — Two-channel LOFO full g: improved low-k (rectangular dipoles) + fixed high-k local kernel.
        #
        # Decompose full g into low/high using the same band_split_poisson_2d(k0_frac), then predict:
        #  - low-k: learned rectangular dipoles (Nx=16,Ny=8) vs annulus dipole baseline (M=32) vs low-k ceiling (kernel)
        #  - high-k: fixed impulse-response kernel truncated to w_local=33
        # Recompose: g_full_pred = g_low_pred + g_high_pred, evaluate in LOFO.
        import matplotlib.pyplot as plt
        from scipy.stats import chi2

        grid_size = int(cfg.get("grid_size", 256))
        alpha = float(cfg.get("alpha", 2.0))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 10_000))
        k0_frac = float(cfg.get("k0_frac", 0.15))
        w_big = _require_odd("w_big", int(cfg.get("w_big", 193)))
        w_local = _require_odd("w_local", int(cfg.get("w_local", 33)))

        # Low-k learned models.
        rect_nx = int(cfg.get("rect_nx", 16))
        rect_ny = int(cfg.get("rect_ny", 8))
        annulus_M = int(cfg.get("annulus_M", 32))
        ring_bin_mode = str(cfg.get("ring_bin_mode", "uniform_r")).lower()

        ridge_alpha = float(cfg.get("ridge_alpha", 1.0))

        # Placebo: permutations of low-rect mapping vs baseline (annulus).
        n_perms = int(cfg.get("n_perms", 100))
        placebo_seed = int(cfg.get("placebo_seed", seed))

        # PASS thresholds (relative to baseline full_two_channel_annulus).
        pass_fisher_p = float(cfg.get("pass_fisher_p", 0.05))
        pass_min_pos_folds = int(cfg.get("pass_min_pos_folds", 7))

        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if n_fields < 2:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        if not (0.0 < float(k0_frac) < 0.5):
            raise ValueError("k0_frac must be in (0,0.5)")
        if w_big > grid_size or w_local > grid_size:
            raise ValueError("w_big/w_local must be <= grid_size")
        if w_local > w_big:
            raise ValueError("w_local must be <= w_big (centers sampled with w_big margin)")
        if rect_nx <= 0 or rect_ny <= 0:
            raise ValueError("rect_nx/rect_ny must be > 0")
        if annulus_M <= 0:
            raise ValueError("annulus_M must be > 0")
        if ring_bin_mode not in {"uniform_r", "equal_area"}:
            raise ValueError("ring_bin_mode must be 'uniform_r' or 'equal_area'")
        if ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be > 0")
        if n_perms <= 0:
            raise ValueError("n_perms must be > 0")
        if pass_min_pos_folds <= 0 or pass_min_pos_folds > n_fields:
            raise ValueError("pass_min_pos_folds out of range")

        def safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            am = a - float(a.mean())
            bm = b - float(b.mean())
            denom = float(np.linalg.norm(am) * np.linalg.norm(bm)) + 1e-12
            return float((am @ bm) / denom)

        def relrmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
            e = y_pred - y_true
            rmse = float(np.sqrt(np.mean(e * e)))
            sd = float(np.std(y_true))
            return rmse / (sd + 1e-12)

        def pearson_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            px = safe_corr_1d(y_true[:, 0], y_pred[:, 0])
            py = safe_corr_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (px + py)

        def relrmse_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            rx = relrmse_1d(y_true[:, 0], y_pred[:, 0])
            ry = relrmse_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (rx + ry)

        def solve_ridge(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            d = int(A.shape[0])
            I = np.eye(d, dtype=np.float64)
            try:
                return np.linalg.solve(A + ridge_alpha * I, B)
            except np.linalg.LinAlgError:
                w, *_ = np.linalg.lstsq(A + ridge_alpha * I, B, rcond=None)
                return w

        def annulus_edges(r_min: float, r_max: float, n_bins: int) -> np.ndarray:
            if ring_bin_mode == "uniform_r":
                return np.linspace(r_min, r_max + 1e-9, n_bins + 1, dtype=np.float64)
            rsq = np.linspace(r_min * r_min, r_max * r_max + 1e-9, n_bins + 1, dtype=np.float64)
            return np.sqrt(rsq, dtype=np.float64)

        def kernel_fft_centered(kernel: np.ndarray, *, grid_size: int) -> np.ndarray:
            kernel = np.asarray(kernel, dtype=np.float64)
            if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
                raise ValueError(f"kernel must be square 2D, got {kernel.shape}")
            w = int(kernel.shape[0])
            if (w % 2) == 0:
                raise ValueError(f"kernel size must be odd, got {w}")
            if w > grid_size:
                raise ValueError(f"kernel size {w} exceeds grid_size {grid_size}")
            r = w // 2
            kr = np.flip(kernel, axis=(0, 1))  # convolution kernel so that output = correlation with `kernel`
            full = np.zeros((grid_size, grid_size), dtype=np.float64)
            cx = grid_size // 2
            cy = grid_size // 2
            full[cx - r : cx + r + 1, cy - r : cy + r + 1] = kr
            full0 = np.fft.ifftshift(full)
            return np.fft.fftn(full0)

        def save_img(arr: np.ndarray, out: Path, title: str) -> None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            vmax = float(np.max(np.abs(arr))) + 1e-12
            im = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r0 in rows:
                out.append("| " + " | ".join(r0.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        def candidate_edges(r: int) -> np.ndarray:
            base = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, r + 1]
            base = [int(x) for x in base if 0 <= int(x) <= r + 1]
            dense = list(range(0, min(16, r + 1) + 1))
            geom = np.unique(np.rint(np.geomspace(1, r + 1, num=64)).astype(int)).tolist()
            vals = sorted(set(base + dense + geom + [0, r + 1]))
            return np.asarray(vals, dtype=np.int64)

        def pick_edges(cand: np.ndarray, n_bins: int) -> np.ndarray:
            cand = np.asarray(cand, dtype=np.int64)
            if len(cand) < n_bins + 1:
                cand = np.arange(int(cand.min()), int(cand.max()) + 1, dtype=np.int64)
            idx = np.round(np.linspace(0, len(cand) - 1, n_bins + 1)).astype(int)
            idx[0] = 0
            idx[-1] = len(cand) - 1
            for k in range(1, len(idx)):
                if idx[k] <= idx[k - 1]:
                    idx[k] = idx[k - 1] + 1
            for k in range(len(idx) - 2, -1, -1):
                if idx[k] >= idx[k + 1]:
                    idx[k] = idx[k + 1] - 1
            if idx[0] != 0 or idx[-1] != len(cand) - 1 or np.any(np.diff(idx) <= 0):
                raise RuntimeError("Failed to build strictly increasing edge indices")
            return cand[idx].astype(np.float64, copy=False)

        def build_rect_bins(wb: int, *, nx_bins: int, ny_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            r_big = wb // 2
            coords = np.arange(int(wb), dtype=np.float64) - float(r_big)
            dx, dy = np.meshgrid(coords, coords, indexing="ij")
            ax = np.abs(dx)
            ay = np.abs(dy)
            cand = candidate_edges(r_big)
            edges_x = pick_edges(cand, int(nx_bins))
            edges_y = pick_edges(cand, int(ny_bins))
            ix = np.digitize(ax, edges_x, right=False) - 1
            iy = np.digitize(ay, edges_y, right=False) - 1
            ix = np.clip(ix, 0, int(nx_bins) - 1).astype(np.int32, copy=False)
            iy = np.clip(iy, 0, int(ny_bins) - 1).astype(np.int32, copy=False)
            bin_idx = (ix * int(ny_bins) + iy).astype(np.int32, copy=False)
            return dx.astype(np.float64), dy.astype(np.float64), bin_idx

        # 1) Generate fields + band splits.
        rho01_by_field: list[np.ndarray] = []
        rho0_by_field: list[np.ndarray] = []
        split_cache: list[BandSplit2D] = []
        var_frac_rows: list[dict[str, Any]] = []
        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
            split = band_split_poisson_2d(rho01, k0_frac=float(k0_frac))
            rho01_by_field.append(rho01.astype(np.float64, copy=False))
            rho0_by_field.append((rho01 - float(rho01.mean())).astype(np.float64, copy=False))
            split_cache.append(split)

            vfx = float(np.var(split.low.gx) / (np.var(split.full.gx) + 1e-12))
            vfy = float(np.var(split.low.gy) / (np.var(split.full.gy) + 1e-12))
            var_frac_rows.append({"field_id": int(field_id), "var_frac_gx": vfx, "var_frac_gy": vfy, "var_frac_mean": 0.5 * (vfx + vfy)})

        var_frac_mean = float(np.mean([r0["var_frac_mean"] for r0 in var_frac_rows]))
        var_frac_std = float(np.std([r0["var_frac_mean"] for r0 in var_frac_rows], ddof=1)) if n_fields > 1 else 0.0

        # 2) Centers (shared across all channels).
        r_big = w_big // 2
        centers_by_field: list[np.ndarray] = []
        centers_idx: list[tuple[np.ndarray, np.ndarray]] = []
        for field_id in range(n_fields):
            rng_cent = np.random.default_rng(seed + 555_555 + 10_000 * int(w_big) + 1_000 * field_id)
            cx = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
            cy = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
            centers = np.column_stack([cx, cy]).astype(np.int64, copy=False)
            centers_by_field.append(centers)
            centers_idx.append((cx.astype(np.int64, copy=False), cy.astype(np.int64, copy=False)))

        def sample_vec_at_centers(gx: np.ndarray, gy: np.ndarray, centers: np.ndarray) -> np.ndarray:
            cx = centers[:, 0].astype(np.int64)
            cy = centers[:, 1].astype(np.int64)
            return np.column_stack([gx[cx, cy], gy[cx, cy]]).astype(np.float64, copy=False)

        y_low_by_field = [sample_vec_at_centers(split_cache[fid].low.gx, split_cache[fid].low.gy, centers_by_field[fid]) for fid in range(n_fields)]
        y_full_by_field = [
            sample_vec_at_centers(split_cache[fid].full.gx, split_cache[fid].full.gy, centers_by_field[fid]) for fid in range(n_fields)
        ]

        # 3) Impulse kernels for low/high.
        rho_delta = np.zeros((grid_size, grid_size), dtype=np.float64)
        c0 = grid_size // 2
        rho_delta[c0, c0] = 1.0
        split_delta = band_split_poisson_2d(rho_delta, k0_frac=float(k0_frac))

        # Low-k (w_big) correlation kernels (kcorr = flip(convolution-kernel patch)).
        g_patch_low_gx = split_delta.low.gx[c0 - r_big : c0 + r_big + 1, c0 - r_big : c0 + r_big + 1]
        g_patch_low_gy = split_delta.low.gy[c0 - r_big : c0 + r_big + 1, c0 - r_big : c0 + r_big + 1]
        kcorr_low_gx = g_patch_low_gx[::-1, ::-1].astype(np.float64, copy=False)
        kcorr_low_gy = g_patch_low_gy[::-1, ::-1].astype(np.float64, copy=False)
        kfft_low_gx = kernel_fft_centered(kcorr_low_gx, grid_size=grid_size)
        kfft_low_gy = kernel_fft_centered(kcorr_low_gy, grid_size=grid_size)

        # High-k (w_local) correlation kernels.
        r_loc = w_local // 2
        g_patch_high_gx = split_delta.high.gx[c0 - r_loc : c0 + r_loc + 1, c0 - r_loc : c0 + r_loc + 1]
        g_patch_high_gy = split_delta.high.gy[c0 - r_loc : c0 + r_loc + 1, c0 - r_loc : c0 + r_loc + 1]
        kcorr_high_gx = g_patch_high_gx[::-1, ::-1].astype(np.float64, copy=False)
        kcorr_high_gy = g_patch_high_gy[::-1, ::-1].astype(np.float64, copy=False)
        kfft_high_gx = kernel_fft_centered(kcorr_high_gx, grid_size=grid_size)
        kfft_high_gy = kernel_fft_centered(kcorr_high_gy, grid_size=grid_size)

        # Save kernel images (sanity).
        save_img(g_patch_low_gx, paths.run_dir / f"kernel_low_gx_wbig{w_big}.png", f"Kx_low truncated (w={w_big})")
        save_img(g_patch_low_gy, paths.run_dir / f"kernel_low_gy_wbig{w_big}.png", f"Ky_low truncated (w={w_big})")
        save_img(g_patch_high_gx, paths.run_dir / f"kernel_high_gx_wlocal{w_local}.png", f"Kx_high truncated (w={w_local})")
        save_img(g_patch_high_gy, paths.run_dir / f"kernel_high_gy_wlocal{w_local}.png", f"Ky_high truncated (w={w_local})")

        # 4) Precompute channel predictions that do not require learning (low ceiling + high kernel).
        y_low_ceiling_pred_by_field: list[np.ndarray] = []
        y_high_kernel_pred_by_field: list[np.ndarray] = []
        for fid in range(n_fields):
            field_fft = np.fft.fftn(rho0_by_field[fid])
            gx_low_pred = np.fft.ifftn(field_fft * kfft_low_gx).real
            gy_low_pred = np.fft.ifftn(field_fft * kfft_low_gy).real
            y_low_ceiling_pred_by_field.append(sample_vec_at_centers(gx_low_pred, gy_low_pred, centers_by_field[fid]))

            gx_high_pred = np.fft.ifftn(field_fft * kfft_high_gx).real
            gy_high_pred = np.fft.ifftn(field_fft * kfft_high_gy).real
            y_high_kernel_pred_by_field.append(sample_vec_at_centers(gx_high_pred, gy_high_pred, centers_by_field[fid]))

        # 5) Build low-k feature blocks on w_big.
        field_fft_rho01_by_field = [np.fft.fftn(rho01_by_field[fid]) for fid in range(n_fields)]

        # Annulus dipole features (M=annulus_M).
        coords = np.arange(int(w_big), dtype=np.float64) - float(r_big)
        dx, dy = np.meshgrid(coords, coords, indexing="ij")
        rgrid = np.sqrt(dx * dx + dy * dy, dtype=np.float64)
        edges_r = annulus_edges(0.0, float(r_big), int(annulus_M))
        n_bins_ann = int(annulus_M)
        F_ann_by_field: list[np.ndarray] = [np.zeros((patches_per_field, 2 * n_bins_ann), dtype=np.float32) for _ in range(n_fields)]
        for m in range(n_bins_ann):
            lo = edges_r[m]
            hi = edges_r[m + 1]
            if m == n_bins_ann - 1:
                ring = (rgrid >= lo) & (rgrid <= hi)
            else:
                ring = (rgrid >= lo) & (rgrid < hi)
            ring_f = ring.astype(np.float64)
            kdx = (dx * ring_f).astype(np.float64, copy=False)
            kdy = (dy * ring_f).astype(np.float64, copy=False)
            kfft_dx = kernel_fft_centered(kdx, grid_size=grid_size)
            kfft_dy = kernel_fft_centered(kdy, grid_size=grid_size)
            for fid in range(n_fields):
                field_fft = field_fft_rho01_by_field[fid]
                cx, cy = centers_idx[fid]
                dx_grid = np.fft.ifftn(field_fft * kfft_dx).real
                dy_grid = np.fft.ifftn(field_fft * kfft_dy).real
                F_ann_by_field[fid][:, m] = dx_grid[cx, cy].astype(np.float32, copy=False)
                F_ann_by_field[fid][:, n_bins_ann + m] = dy_grid[cx, cy].astype(np.float32, copy=False)

        # Rectangular dipoles (Nx,Ny).
        dxr, dyr, bin_idx = build_rect_bins(w_big, nx_bins=int(rect_nx), ny_bins=int(rect_ny))
        n_bins_rect = int(rect_nx) * int(rect_ny)
        F_rect_by_field: list[np.ndarray] = [np.zeros((patches_per_field, 2 * n_bins_rect), dtype=np.float32) for _ in range(n_fields)]
        for b in range(n_bins_rect):
            mask = (bin_idx == int(b)).astype(np.float64, copy=False)
            kdx = (dxr * mask).astype(np.float64, copy=False)
            kdy = (dyr * mask).astype(np.float64, copy=False)
            kfft_dx = kernel_fft_centered(kdx, grid_size=grid_size)
            kfft_dy = kernel_fft_centered(kdy, grid_size=grid_size)
            for fid in range(n_fields):
                field_fft = field_fft_rho01_by_field[fid]
                cx, cy = centers_idx[fid]
                dx_grid = np.fft.ifftn(field_fft * kfft_dx).real
                dy_grid = np.fft.ifftn(field_fft * kfft_dy).real
                F_rect_by_field[fid][:, b] = dx_grid[cx, cy].astype(np.float32, copy=False)
                F_rect_by_field[fid][:, n_bins_rect + b] = dy_grid[cx, cy].astype(np.float32, copy=False)

        # 6) LOFO: fit low-k models (annulus/rect), recombine with high-k kernel, evaluate on full g.
        def fit_predict_fold(
            F_by_field: list[np.ndarray],
            y_by_field: list[np.ndarray],
            *,
            test_field: int,
            permute_train_y: bool = False,
            perm_rng: np.random.Generator | None = None,
        ) -> np.ndarray:
            # Fit ridge with StandardScaler and intercept using sufficient statistics.
            d = int(F_by_field[0].shape[1])
            sums_F = [np.asarray(F_by_field[fid], dtype=np.float64).sum(axis=0) for fid in range(n_fields)]
            sums_y = [np.asarray(y_by_field[fid], dtype=np.float64).sum(axis=0) for fid in range(n_fields)]
            FtF = [(np.asarray(F_by_field[fid], dtype=np.float64).T @ np.asarray(F_by_field[fid], dtype=np.float64)) for fid in range(n_fields)]

            # FtY may be permuted.
            if permute_train_y:
                if perm_rng is None:
                    raise ValueError("perm_rng required when permute_train_y=True")
                FtY = []
                for fid in range(n_fields):
                    F = np.asarray(F_by_field[fid], dtype=np.float64)
                    y = np.asarray(y_by_field[fid], dtype=np.float64)
                    if fid == test_field:
                        FtY.append(F.T @ y)
                    else:
                        perm = perm_rng.permutation(y.shape[0])
                        FtY.append(F.T @ y[perm])
            else:
                FtY = [(np.asarray(F_by_field[fid], dtype=np.float64).T @ np.asarray(y_by_field[fid], dtype=np.float64)) for fid in range(n_fields)]

            total_sum_F = np.sum(np.stack(sums_F, axis=0), axis=0)
            total_sum_y = np.sum(np.stack(sums_y, axis=0), axis=0)
            total_FtF = np.sum(np.stack(FtF, axis=0), axis=0)
            total_FtY = np.sum(np.stack(FtY, axis=0), axis=0)

            n_total = int(n_fields) * int(patches_per_field)
            n_test = int(patches_per_field)
            n_train = n_total - n_test

            sum_F_tr = total_sum_F - sums_F[test_field]
            sum_y_tr = total_sum_y - sums_y[test_field]
            FtF_tr = total_FtF - FtF[test_field]
            FtY_tr = total_FtY - FtY[test_field]

            mu = sum_F_tr / float(n_train)
            y_mu = sum_y_tr / float(n_train)
            var = np.diag(FtF_tr) / float(n_train) - mu * mu
            var = np.where(var > 1e-12, var, 1e-12)
            sd = np.sqrt(var, dtype=np.float64)

            FtF_center = FtF_tr - float(n_train) * np.outer(mu, mu)
            FtY_center = FtY_tr - float(n_train) * (mu[:, None] * y_mu[None, :])

            XtX = FtF_center / (sd[:, None] * sd[None, :])
            Xty = FtY_center / sd[:, None]
            w = solve_ridge(XtX, Xty)

            Fte = np.asarray(F_by_field[test_field], dtype=np.float64)
            Xte = (Fte - mu) / sd
            y_pred = Xte @ w + y_mu
            return y_pred.astype(np.float64, copy=False)

        fold_rows: list[dict[str, Any]] = []
        delta_real: list[float] = []
        delta_real_rel: list[float] = []
        placebo_pvals: list[float] = []

        # Metrics storage per model across folds.
        model_fold_metrics: dict[str, list[dict[str, Any]]] = {k: [] for k in ["full_annulus", "full_rect", "full_ceiling"]}
        low_fold_metrics: dict[str, list[dict[str, Any]]] = {k: [] for k in ["low_annulus", "low_rect", "low_ceiling"]}

        for test_field in range(n_fields):
            y_full_true = y_full_by_field[test_field]
            y_low_true = y_low_by_field[test_field]

            # Low-k predictions.
            y_low_ceiling = y_low_ceiling_pred_by_field[test_field]
            y_low_ann = fit_predict_fold(F_ann_by_field, y_low_by_field, test_field=test_field)
            y_low_rect = fit_predict_fold(F_rect_by_field, y_low_by_field, test_field=test_field)

            # High-k prediction.
            y_high = y_high_kernel_pred_by_field[test_field]

            # Full predictions.
            y_full_ceiling = y_low_ceiling + y_high
            y_full_ann = y_low_ann + y_high
            y_full_rect = y_low_rect + y_high

            # Metrics.
            p_full_ceiling = pearson_mean_2d(y_full_true, y_full_ceiling)
            rr_full_ceiling = relrmse_mean_2d(y_full_true, y_full_ceiling)
            p_full_ann = pearson_mean_2d(y_full_true, y_full_ann)
            rr_full_ann = relrmse_mean_2d(y_full_true, y_full_ann)
            p_full_rect = pearson_mean_2d(y_full_true, y_full_rect)
            rr_full_rect = relrmse_mean_2d(y_full_true, y_full_rect)

            p_low_ceiling = pearson_mean_2d(y_low_true, y_low_ceiling)
            rr_low_ceiling = relrmse_mean_2d(y_low_true, y_low_ceiling)
            p_low_ann = pearson_mean_2d(y_low_true, y_low_ann)
            rr_low_ann = relrmse_mean_2d(y_low_true, y_low_ann)
            p_low_rect = pearson_mean_2d(y_low_true, y_low_rect)
            rr_low_rect = relrmse_mean_2d(y_low_true, y_low_rect)

            model_fold_metrics["full_ceiling"].append({"field_id": int(test_field), "pearson": p_full_ceiling, "relRMSE": rr_full_ceiling})
            model_fold_metrics["full_annulus"].append({"field_id": int(test_field), "pearson": p_full_ann, "relRMSE": rr_full_ann})
            model_fold_metrics["full_rect"].append({"field_id": int(test_field), "pearson": p_full_rect, "relRMSE": rr_full_rect})

            low_fold_metrics["low_ceiling"].append({"field_id": int(test_field), "pearson": p_low_ceiling, "relRMSE": rr_low_ceiling})
            low_fold_metrics["low_annulus"].append({"field_id": int(test_field), "pearson": p_low_ann, "relRMSE": rr_low_ann})
            low_fold_metrics["low_rect"].append({"field_id": int(test_field), "pearson": p_low_rect, "relRMSE": rr_low_rect})

            dP = float(p_full_rect - p_full_ann)
            dR = float(rr_full_rect - rr_full_ann)
            delta_real.append(dP)
            delta_real_rel.append(dR)

            # Placebo for rect features: permute y_low in TRAIN only, refit, evaluate ΔPearson vs annulus baseline.
            rng_perm = np.random.default_rng(int(placebo_seed) + 10_000 * test_field)
            deltas_perm = np.zeros((n_perms,), dtype=np.float64)
            for pidx in range(n_perms):
                rng_pi = np.random.default_rng(rng_perm.integers(0, 2**32 - 1, dtype=np.uint64))
                y_low_rect_perm = fit_predict_fold(
                    F_rect_by_field,
                    y_low_by_field,
                    test_field=test_field,
                    permute_train_y=True,
                    perm_rng=rng_pi,
                )
                y_full_rect_perm = y_low_rect_perm + y_high
                p_perm = pearson_mean_2d(y_full_true, y_full_rect_perm)
                deltas_perm[pidx] = float(p_perm - p_full_ann)

            # Empirical p-value (one-sided): fraction of perm deltas >= real delta.
            denom = float(n_perms + 1)
            p_emp = (float(np.sum(deltas_perm >= dP)) + 1.0) / denom
            placebo_pvals.append(p_emp)

            fold_rows.append(
                {
                    "field_id": int(test_field),
                    "pearson_full_ceiling": float(p_full_ceiling),
                    "pearson_full_annulus": float(p_full_ann),
                    "pearson_full_rect": float(p_full_rect),
                    "deltaP_rect_minus_annulus": float(dP),
                    "relRMSE_full_ceiling": float(rr_full_ceiling),
                    "relRMSE_full_annulus": float(rr_full_ann),
                    "relRMSE_full_rect": float(rr_full_rect),
                    "deltaR_rect_minus_annulus": float(dR),
                    "p_emp_placebo": float(p_emp),
                }
            )

        # Aggregates.
        def mean_std(vals: list[float]) -> tuple[float, float]:
            arr = np.asarray(vals, dtype=np.float64)
            return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

        def agg_metrics(rows: list[dict[str, Any]]) -> tuple[float, float, float, float]:
            p = np.asarray([r0["pearson"] for r0 in rows], dtype=np.float64)
            r = np.asarray([r0["relRMSE"] for r0 in rows], dtype=np.float64)
            return float(p.mean()), float(p.std(ddof=1)) if len(p) > 1 else 0.0, float(r.mean()), float(r.std(ddof=1)) if len(r) > 1 else 0.0

        full_ceiling_m = agg_metrics(model_fold_metrics["full_ceiling"])
        full_ann_m = agg_metrics(model_fold_metrics["full_annulus"])
        full_rect_m = agg_metrics(model_fold_metrics["full_rect"])

        low_ceiling_m = agg_metrics(low_fold_metrics["low_ceiling"])
        low_ann_m = agg_metrics(low_fold_metrics["low_annulus"])
        low_rect_m = agg_metrics(low_fold_metrics["low_rect"])

        dP_mean, dP_std = mean_std(delta_real)
        dR_mean, dR_std = mean_std(delta_real_rel)

        # Fisher combine placebo p-values across folds.
        pvals = np.clip(np.asarray(placebo_pvals, dtype=np.float64), 1e-300, 1.0)
        fisher_stat = float(-2.0 * np.sum(np.log(pvals)))
        fisher_p = float(chi2.sf(fisher_stat, df=2 * len(pvals)))

        n_pos = int(np.sum(np.asarray(delta_real, dtype=np.float64) > 0))
        passes = (dP_mean > 0) and (dR_mean < 0) and (fisher_p < pass_fisher_p) and (n_pos >= pass_min_pos_folds)

        # Summary markdown.
        md_var = [
            {
                "field_id": str(r0["field_id"]),
                "var_frac_gx": f"{float(r0['var_frac_gx']):.4f}",
                "var_frac_gy": f"{float(r0['var_frac_gy']):.4f}",
                "var_frac_mean": f"{float(r0['var_frac_mean']):.4f}",
            }
            for r0 in var_frac_rows
        ]

        def fmt_ms(m: tuple[float, float, float, float]) -> tuple[str, str]:
            return f"{m[0]:.4f} ± {m[1]:.4f}", f"{m[2]:.4f} ± {m[3]:.4f}"

        full_rows = []
        for name, m in [
            ("full_two_channel_annulus", full_ann_m),
            ("full_two_channel_rect", full_rect_m),
            ("full_two_channel_ceiling", full_ceiling_m),
        ]:
            p_s, r_s = fmt_ms(m)
            full_rows.append({"model": name, "Pearson mean±std": p_s, "relRMSE mean±std": r_s})

        low_rows = []
        for name, m in [
            ("low_annulus_learn", low_ann_m),
            ("low_rect_learn", low_rect_m),
            ("low_ceiling", low_ceiling_m),
        ]:
            p_s, r_s = fmt_ms(m)
            low_rows.append({"model": name, "Pearson mean±std": p_s, "relRMSE mean±std": r_s})

        md_folds = [
            {
                "field_id": str(r0["field_id"]),
                "Pearson_baseline(annulus)": f"{r0['pearson_full_annulus']:.4f}",
                "Pearson_rect": f"{r0['pearson_full_rect']:.4f}",
                "ΔPearson": f"{r0['deltaP_rect_minus_annulus']:+.4f}",
                "relRMSE_baseline(annulus)": f"{r0['relRMSE_full_annulus']:.4f}",
                "relRMSE_rect": f"{r0['relRMSE_full_rect']:.4f}",
                "ΔrelRMSE": f"{r0['deltaR_rect_minus_annulus']:+.4f}",
                "p_emp(placebo)": f"{r0['p_emp_placebo']:.4f}",
            }
            for r0 in fold_rows
        ]

        summary_md = (
            "# E38 — Two-channel LOFO full g (rect low-k + kernel high-k)\n\n"
            f"- run: `{paths.run_dir}`\n"
            f"- grid_size={grid_size}, alpha={alpha}, k0_frac={k0_frac}, n_fields={n_fields}, patches_per_field={patches_per_field}\n"
            f"- w_big={w_big} (low-k), w_local={w_local} (high-k kernel)\n"
            f"- low_rect: Nx={rect_nx}, Ny={rect_ny} (d={2*rect_nx*rect_ny}) ; low_annulus: M={annulus_M} (d={2*annulus_M})\n"
            f"- placebo: n_perms={n_perms} per fold, placebo_seed={placebo_seed}\n\n"
            "## Low-k variance fraction (per field)\n\n"
            + md_table(md_var, ["field_id", "var_frac_gx", "var_frac_gy", "var_frac_mean"])
            + f"\n\n- var_frac_mean across fields: {var_frac_mean:.4f} ± {var_frac_std:.4f}\n\n"
            "## Low-k channel metrics (gx_low, gy_low)\n\n"
            + md_table(low_rows, ["model", "Pearson mean±std", "relRMSE mean±std"])
            + "\n\n## Full g metrics (gx_full, gy_full)\n\n"
            + md_table(full_rows, ["model", "Pearson mean±std", "relRMSE mean±std"])
            + "\n\n## Improvement: rect vs annulus (full g)\n\n"
            + f"- ΔPearson mean±std: {dP_mean:+.4f} ± {dP_std:.4f}\n"
            + f"- ΔrelRMSE mean±std: {dR_mean:+.4f} ± {dR_std:.4f}\n"
            + f"- #folds ΔPearson>0: {n_pos}/{n_fields}\n"
            + f"- Fisher p-value (placebo ΔPearson): {fisher_p:.4g}\n"
            + f"- VERDICT: {'PASS' if passes else 'FAIL'}\n\n"
            "## Per-fold table (baseline=annulus)\n\n"
            + md_table(
                md_folds,
                [
                    "field_id",
                    "Pearson_baseline(annulus)",
                    "Pearson_rect",
                    "ΔPearson",
                    "relRMSE_baseline(annulus)",
                    "relRMSE_rect",
                    "ΔrelRMSE",
                    "p_emp(placebo)",
                ],
            )
            + "\n"
        )

        (paths.run_dir / "summary_e38_two_channel_fullg_rectlowk.md").write_text(summary_md, encoding="utf-8")
        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "grid_size": grid_size,
                "alpha": alpha,
                "k0_frac": k0_frac,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "w_big": int(w_big),
                "w_local": int(w_local),
                "rect": {"nx": int(rect_nx), "ny": int(rect_ny)},
                "annulus_M": int(annulus_M),
                "ring_bin_mode": ring_bin_mode,
                "ridge_alpha": ridge_alpha,
                "placebo": {"n_perms": int(n_perms), "seed": int(placebo_seed)},
                "var_frac_rows": var_frac_rows,
                "fold_rows": fold_rows,
                "full_metrics": {"annulus": full_ann_m, "rect": full_rect_m, "ceiling": full_ceiling_m},
                "low_metrics": {"annulus": low_ann_m, "rect": low_rect_m, "ceiling": low_ceiling_m},
                "delta": {
                    "dPearson_mean": dP_mean,
                    "dPearson_std": dP_std,
                    "drelRMSE_mean": dR_mean,
                    "drelRMSE_std": dR_std,
                    "n_pos": n_pos,
                    "fisher_p": fisher_p,
                    "pass": bool(passes),
                },
            },
        )
        return paths

    if experiment == "e39":
        # E39 — Full-g LOFO: compare two-channel rect vs pixels-full baseline + rect projection ablation.
        #
        # Models (full g):
        #  A) P_full_pixels: pixels-only linear model on rho (w_big), trained with SGDRegressor (streamed).
        #  B) two_channel_rect_learn: low_rect_learn (learned) + high_kernel
        #  C) two_channel_rect_proj: low_rect_proj (projection-only) + high_kernel
        #  D) two_channel_ceiling: low_ceiling (truncated impulse kernel) + high_kernel
        #
        # Also report deltas vs P_full_pixels and vs ceiling; optional placebo for rect_learn vs P_full.
        import matplotlib.pyplot as plt
        from numpy.lib.stride_tricks import sliding_window_view
        from scipy.stats import chi2
        from sklearn.linear_model import SGDRegressor

        grid_size = int(cfg.get("grid_size", 256))
        alpha = float(cfg.get("alpha", 2.0))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 10_000))
        k0_frac = float(cfg.get("k0_frac", 0.15))
        w_big = _require_odd("w_big", int(cfg.get("w_big", 193)))
        w_local = _require_odd("w_local", int(cfg.get("w_local", 33)))

        # Low-k models.
        rect_nx = int(cfg.get("rect_nx", 16))
        rect_ny = int(cfg.get("rect_ny", 8))
        annulus_M = int(cfg.get("annulus_M", 32))
        ring_bin_mode = str(cfg.get("ring_bin_mode", "uniform_r")).lower()
        ridge_alpha = float(cfg.get("ridge_alpha", 1.0))

        # Pixels baseline (SGD).
        pixel_epochs = int(cfg.get("pixel_epochs", 4))
        pixel_batch_size = int(cfg.get("pixel_batch_size", 256))
        pixel_alpha = float(cfg.get("pixel_alpha", 1e-4))
        pixel_eta0_base = float(cfg.get("pixel_eta0_base", 0.01))
        pixel_power_t = float(cfg.get("pixel_power_t", 0.25))

        # Placebo for rect_learn vs P_full.
        rect_placebo_enabled = bool(cfg.get("rect_placebo_enabled", True))
        rect_placebo_perms = int(cfg.get("rect_placebo_perms", 50))
        placebo_seed = int(cfg.get("placebo_seed", seed))

        # PASS criteria (rect_learn vs P_full).
        pass_fisher_p = float(cfg.get("pass_fisher_p", 0.05))
        pass_min_pos_folds = int(cfg.get("pass_min_pos_folds", 7))

        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if n_fields < 2:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        if not (0.0 < float(k0_frac) < 0.5):
            raise ValueError("k0_frac must be in (0,0.5)")
        if w_big > grid_size or w_local > grid_size:
            raise ValueError("w_big/w_local must be <= grid_size")
        if w_local > w_big:
            raise ValueError("w_local must be <= w_big")
        if rect_nx <= 0 or rect_ny <= 0:
            raise ValueError("rect_nx/rect_ny must be > 0")
        if annulus_M <= 0:
            raise ValueError("annulus_M must be > 0")
        if ring_bin_mode not in {"uniform_r", "equal_area"}:
            raise ValueError("ring_bin_mode must be 'uniform_r' or 'equal_area'")
        if ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be > 0")
        if pixel_epochs <= 0:
            raise ValueError("pixel_epochs must be > 0")
        if pixel_batch_size <= 0:
            raise ValueError("pixel_batch_size must be > 0")
        if pixel_alpha <= 0 or pixel_eta0_base <= 0:
            raise ValueError("pixel_alpha/pixel_eta0_base must be > 0")
        if rect_placebo_perms <= 0:
            raise ValueError("rect_placebo_perms must be > 0")
        if pass_min_pos_folds <= 0 or pass_min_pos_folds > n_fields:
            raise ValueError("pass_min_pos_folds out of range")

        def safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            am = a - float(a.mean())
            bm = b - float(b.mean())
            denom = float(np.linalg.norm(am) * np.linalg.norm(bm)) + 1e-12
            return float((am @ bm) / denom)

        def relrmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
            e = y_pred - y_true
            rmse = float(np.sqrt(np.mean(e * e)))
            sd = float(np.std(y_true))
            return rmse / (sd + 1e-12)

        def pearson_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            px = safe_corr_1d(y_true[:, 0], y_pred[:, 0])
            py = safe_corr_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (px + py)

        def relrmse_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            rx = relrmse_1d(y_true[:, 0], y_pred[:, 0])
            ry = relrmse_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (rx + ry)

        def solve_ridge(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            d = int(A.shape[0])
            I = np.eye(d, dtype=np.float64)
            try:
                return np.linalg.solve(A + ridge_alpha * I, B)
            except np.linalg.LinAlgError:
                w, *_ = np.linalg.lstsq(A + ridge_alpha * I, B, rcond=None)
                return w

        def annulus_edges(r_min: float, r_max: float, n_bins: int) -> np.ndarray:
            if ring_bin_mode == "uniform_r":
                return np.linspace(r_min, r_max + 1e-9, n_bins + 1, dtype=np.float64)
            rsq = np.linspace(r_min * r_min, r_max * r_max + 1e-9, n_bins + 1, dtype=np.float64)
            return np.sqrt(rsq, dtype=np.float64)

        def kernel_fft_centered(kernel: np.ndarray, *, grid_size: int) -> np.ndarray:
            kernel = np.asarray(kernel, dtype=np.float64)
            if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
                raise ValueError(f"kernel must be square 2D, got {kernel.shape}")
            w = int(kernel.shape[0])
            if (w % 2) == 0:
                raise ValueError(f"kernel size must be odd, got {w}")
            if w > grid_size:
                raise ValueError(f"kernel size {w} exceeds grid_size {grid_size}")
            r = w // 2
            kr = np.flip(kernel, axis=(0, 1))
            full = np.zeros((grid_size, grid_size), dtype=np.float64)
            cx = grid_size // 2
            cy = grid_size // 2
            full[cx - r : cx + r + 1, cy - r : cy + r + 1] = kr
            full0 = np.fft.ifftshift(full)
            return np.fft.fftn(full0)

        def save_img(arr: np.ndarray, out: Path, title: str) -> None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            vmax = float(np.max(np.abs(arr))) + 1e-12
            im = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r0 in rows:
                out.append("| " + " | ".join(r0.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        def candidate_edges(r: int) -> np.ndarray:
            base = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, r + 1]
            base = [int(x) for x in base if 0 <= int(x) <= r + 1]
            dense = list(range(0, min(16, r + 1) + 1))
            geom = np.unique(np.rint(np.geomspace(1, r + 1, num=64)).astype(int)).tolist()
            vals = sorted(set(base + dense + geom + [0, r + 1]))
            return np.asarray(vals, dtype=np.int64)

        def pick_edges(cand: np.ndarray, n_bins: int) -> np.ndarray:
            cand = np.asarray(cand, dtype=np.int64)
            if len(cand) < n_bins + 1:
                cand = np.arange(int(cand.min()), int(cand.max()) + 1, dtype=np.int64)
            idx = np.round(np.linspace(0, len(cand) - 1, n_bins + 1)).astype(int)
            idx[0] = 0
            idx[-1] = len(cand) - 1
            for k in range(1, len(idx)):
                if idx[k] <= idx[k - 1]:
                    idx[k] = idx[k - 1] + 1
            for k in range(len(idx) - 2, -1, -1):
                if idx[k] >= idx[k + 1]:
                    idx[k] = idx[k + 1] - 1
            if idx[0] != 0 or idx[-1] != len(cand) - 1 or np.any(np.diff(idx) <= 0):
                raise RuntimeError("Failed to build strictly increasing edge indices")
            return cand[idx].astype(np.float64, copy=False)

        def build_rect_bins(wb: int, *, nx_bins: int, ny_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            r = wb // 2
            coords = np.arange(int(wb), dtype=np.float64) - float(r)
            dx, dy = np.meshgrid(coords, coords, indexing="ij")
            ax = np.abs(dx)
            ay = np.abs(dy)
            cand = candidate_edges(r)
            edges_x = pick_edges(cand, int(nx_bins))
            edges_y = pick_edges(cand, int(ny_bins))
            ix = np.digitize(ax, edges_x, right=False) - 1
            iy = np.digitize(ay, edges_y, right=False) - 1
            ix = np.clip(ix, 0, int(nx_bins) - 1).astype(np.int32, copy=False)
            iy = np.clip(iy, 0, int(ny_bins) - 1).astype(np.int32, copy=False)
            bin_idx = (ix * int(ny_bins) + iy).astype(np.int32, copy=False)
            return dx.astype(np.float64), dy.astype(np.float64), bin_idx

        # Generate fields + band split.
        rho01_by_field: list[np.ndarray] = []
        rho0_by_field: list[np.ndarray] = []
        rhoz_by_field: list[np.ndarray] = []
        field_fft_rho01_by_field: list[np.ndarray] = []
        split_cache: list[BandSplit2D] = []
        var_frac_rows: list[dict[str, Any]] = []

        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
            split = band_split_poisson_2d(rho01, k0_frac=float(k0_frac))
            rho01_by_field.append(rho01.astype(np.float64, copy=False))
            rho0 = (rho01 - float(rho01.mean())).astype(np.float64, copy=False)
            rho0_by_field.append(rho0)
            mu = float(rho01.mean())
            sd = float(rho01.std())
            rhoz_by_field.append(((rho01 - mu) / (sd + 1e-12)).astype(np.float32, copy=False))
            field_fft_rho01_by_field.append(np.fft.fftn(rho01))
            split_cache.append(split)

            vfx = float(np.var(split.low.gx) / (np.var(split.full.gx) + 1e-12))
            vfy = float(np.var(split.low.gy) / (np.var(split.full.gy) + 1e-12))
            var_frac_rows.append({"field_id": int(field_id), "var_frac_gx": vfx, "var_frac_gy": vfy, "var_frac_mean": 0.5 * (vfx + vfy)})

        var_frac_mean = float(np.mean([r0["var_frac_mean"] for r0 in var_frac_rows]))
        var_frac_std = float(np.std([r0["var_frac_mean"] for r0 in var_frac_rows], ddof=1)) if n_fields > 1 else 0.0

        # Centers (margin w_big).
        r_big = w_big // 2
        centers_by_field: list[np.ndarray] = []
        centers_idx: list[tuple[np.ndarray, np.ndarray]] = []
        for field_id in range(n_fields):
            rng_cent = np.random.default_rng(seed + 555_555 + 10_000 * int(w_big) + 1_000 * field_id)
            cx = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
            cy = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
            centers = np.column_stack([cx, cy]).astype(np.int64, copy=False)
            centers_by_field.append(centers)
            centers_idx.append((cx.astype(np.int64, copy=False), cy.astype(np.int64, copy=False)))

        def sample_vec_at_centers(gx: np.ndarray, gy: np.ndarray, centers: np.ndarray) -> np.ndarray:
            cx = centers[:, 0].astype(np.int64)
            cy = centers[:, 1].astype(np.int64)
            return np.column_stack([gx[cx, cy], gy[cx, cy]]).astype(np.float64, copy=False)

        y_low_by_field = [sample_vec_at_centers(split_cache[fid].low.gx, split_cache[fid].low.gy, centers_by_field[fid]) for fid in range(n_fields)]
        y_full_by_field = [
            sample_vec_at_centers(split_cache[fid].full.gx, split_cache[fid].full.gy, centers_by_field[fid]) for fid in range(n_fields)
        ]

        # Impulse kernels for low/high.
        rho_delta = np.zeros((grid_size, grid_size), dtype=np.float64)
        c0 = grid_size // 2
        rho_delta[c0, c0] = 1.0
        split_delta = band_split_poisson_2d(rho_delta, k0_frac=float(k0_frac))

        # Low-k (w_big) correlation kernels.
        g_patch_low_gx = split_delta.low.gx[c0 - r_big : c0 + r_big + 1, c0 - r_big : c0 + r_big + 1]
        g_patch_low_gy = split_delta.low.gy[c0 - r_big : c0 + r_big + 1, c0 - r_big : c0 + r_big + 1]
        kcorr_low_gx = g_patch_low_gx[::-1, ::-1].astype(np.float64, copy=False)
        kcorr_low_gy = g_patch_low_gy[::-1, ::-1].astype(np.float64, copy=False)
        kfft_low_gx = kernel_fft_centered(kcorr_low_gx, grid_size=grid_size)
        kfft_low_gy = kernel_fft_centered(kcorr_low_gy, grid_size=grid_size)

        # High-k (w_local) correlation kernels.
        r_loc = w_local // 2
        g_patch_high_gx = split_delta.high.gx[c0 - r_loc : c0 + r_loc + 1, c0 - r_loc : c0 + r_loc + 1]
        g_patch_high_gy = split_delta.high.gy[c0 - r_loc : c0 + r_loc + 1, c0 - r_loc : c0 + r_loc + 1]
        kcorr_high_gx = g_patch_high_gx[::-1, ::-1].astype(np.float64, copy=False)
        kcorr_high_gy = g_patch_high_gy[::-1, ::-1].astype(np.float64, copy=False)
        kfft_high_gx = kernel_fft_centered(kcorr_high_gx, grid_size=grid_size)
        kfft_high_gy = kernel_fft_centered(kcorr_high_gy, grid_size=grid_size)

        save_img(g_patch_low_gx, paths.run_dir / f"kernel_low_gx_wbig{w_big}.png", f"Kx_low truncated (w={w_big})")
        save_img(g_patch_low_gy, paths.run_dir / f"kernel_low_gy_wbig{w_big}.png", f"Ky_low truncated (w={w_big})")
        save_img(g_patch_high_gx, paths.run_dir / f"kernel_high_gx_wlocal{w_local}.png", f"Kx_high truncated (w={w_local})")
        save_img(g_patch_high_gy, paths.run_dir / f"kernel_high_gy_wlocal{w_local}.png", f"Ky_high truncated (w={w_local})")

        # Fixed channel predictions: low ceiling + high kernel.
        y_low_ceiling_pred_by_field: list[np.ndarray] = []
        y_high_kernel_pred_by_field: list[np.ndarray] = []
        for fid in range(n_fields):
            field_fft0 = np.fft.fftn(rho0_by_field[fid])
            gx_low_pred = np.fft.ifftn(field_fft0 * kfft_low_gx).real
            gy_low_pred = np.fft.ifftn(field_fft0 * kfft_low_gy).real
            y_low_ceiling_pred_by_field.append(sample_vec_at_centers(gx_low_pred, gy_low_pred, centers_by_field[fid]))

            gx_high_pred = np.fft.ifftn(field_fft0 * kfft_high_gx).real
            gy_high_pred = np.fft.ifftn(field_fft0 * kfft_high_gy).real
            y_high_kernel_pred_by_field.append(sample_vec_at_centers(gx_high_pred, gy_high_pred, centers_by_field[fid]))

        # Low-k features: annulus and rect (computed once).
        # Annulus dipoles.
        coords = np.arange(int(w_big), dtype=np.float64) - float(r_big)
        dx, dy = np.meshgrid(coords, coords, indexing="ij")
        rgrid = np.sqrt(dx * dx + dy * dy, dtype=np.float64)
        edges_r = annulus_edges(0.0, float(r_big), int(annulus_M))
        n_bins_ann = int(annulus_M)
        F_ann_by_field: list[np.ndarray] = [np.zeros((patches_per_field, 2 * n_bins_ann), dtype=np.float32) for _ in range(n_fields)]
        for m in range(n_bins_ann):
            lo = edges_r[m]
            hi = edges_r[m + 1]
            if m == n_bins_ann - 1:
                ring = (rgrid >= lo) & (rgrid <= hi)
            else:
                ring = (rgrid >= lo) & (rgrid < hi)
            ring_f = ring.astype(np.float64)
            kdx = (dx * ring_f).astype(np.float64, copy=False)
            kdy = (dy * ring_f).astype(np.float64, copy=False)
            kfft_dx = kernel_fft_centered(kdx, grid_size=grid_size)
            kfft_dy = kernel_fft_centered(kdy, grid_size=grid_size)
            for fid in range(n_fields):
                field_fft = field_fft_rho01_by_field[fid]
                cx, cy = centers_idx[fid]
                dx_grid = np.fft.ifftn(field_fft * kfft_dx).real
                dy_grid = np.fft.ifftn(field_fft * kfft_dy).real
                F_ann_by_field[fid][:, m] = dx_grid[cx, cy].astype(np.float32, copy=False)
                F_ann_by_field[fid][:, n_bins_ann + m] = dy_grid[cx, cy].astype(np.float32, copy=False)

        # Rect dipoles.
        dxr, dyr, bin_idx = build_rect_bins(w_big, nx_bins=int(rect_nx), ny_bins=int(rect_ny))
        n_bins_rect = int(rect_nx) * int(rect_ny)
        F_rect_by_field: list[np.ndarray] = [np.zeros((patches_per_field, 2 * n_bins_rect), dtype=np.float32) for _ in range(n_fields)]
        for b in range(n_bins_rect):
            mask = (bin_idx == int(b)).astype(np.float64, copy=False)
            kdx = (dxr * mask).astype(np.float64, copy=False)
            kdy = (dyr * mask).astype(np.float64, copy=False)
            kfft_dx = kernel_fft_centered(kdx, grid_size=grid_size)
            kfft_dy = kernel_fft_centered(kdy, grid_size=grid_size)
            for fid in range(n_fields):
                field_fft = field_fft_rho01_by_field[fid]
                cx, cy = centers_idx[fid]
                dx_grid = np.fft.ifftn(field_fft * kfft_dx).real
                dy_grid = np.fft.ifftn(field_fft * kfft_dy).real
                F_rect_by_field[fid][:, b] = dx_grid[cx, cy].astype(np.float32, copy=False)
                F_rect_by_field[fid][:, n_bins_rect + b] = dy_grid[cx, cy].astype(np.float32, copy=False)

        # Rect projection coefficients (from low kernel onto basis).
        bin_flat = bin_idx.reshape(-1).astype(np.int64, copy=False)
        dx_flat = dxr.reshape(-1).astype(np.float64, copy=False)
        dy_flat = dyr.reshape(-1).astype(np.float64, copy=False)
        kx_flat = kcorr_low_gx.reshape(-1).astype(np.float64, copy=False)
        ky_flat = kcorr_low_gy.reshape(-1).astype(np.float64, copy=False)
        num_x = np.bincount(bin_flat, weights=kx_flat * dx_flat, minlength=n_bins_rect).astype(np.float64, copy=False)
        den_x = np.bincount(bin_flat, weights=dx_flat * dx_flat, minlength=n_bins_rect).astype(np.float64, copy=False)
        coef_rect_x = np.where(den_x > 0, num_x / (den_x + 1e-12), 0.0).astype(np.float64, copy=False)
        num_y = np.bincount(bin_flat, weights=ky_flat * dy_flat, minlength=n_bins_rect).astype(np.float64, copy=False)
        den_y = np.bincount(bin_flat, weights=dy_flat * dy_flat, minlength=n_bins_rect).astype(np.float64, copy=False)
        coef_rect_y = np.where(den_y > 0, num_y / (den_y + 1e-12), 0.0).astype(np.float64, copy=False)

        # Helper: fit ridge on low-k features (LOFO) and predict on test field (fast, d<=256).
        def fit_predict_low(
            F_by_field: list[np.ndarray],
            y_by_field: list[np.ndarray],
            *,
            test_field: int,
            permute_train_y: bool = False,
            perm_rng: np.random.Generator | None = None,
        ) -> np.ndarray:
            d = int(F_by_field[0].shape[1])
            sums_F = [np.asarray(F_by_field[fid], dtype=np.float64).sum(axis=0) for fid in range(n_fields)]
            sums_y = [np.asarray(y_by_field[fid], dtype=np.float64).sum(axis=0) for fid in range(n_fields)]
            FtF = [(np.asarray(F_by_field[fid], dtype=np.float64).T @ np.asarray(F_by_field[fid], dtype=np.float64)) for fid in range(n_fields)]

            if permute_train_y:
                if perm_rng is None:
                    raise ValueError("perm_rng required when permute_train_y=True")
                FtY = []
                for fid in range(n_fields):
                    F = np.asarray(F_by_field[fid], dtype=np.float64)
                    y = np.asarray(y_by_field[fid], dtype=np.float64)
                    if fid == test_field:
                        FtY.append(F.T @ y)
                    else:
                        perm = perm_rng.permutation(y.shape[0])
                        FtY.append(F.T @ y[perm])
            else:
                FtY = [(np.asarray(F_by_field[fid], dtype=np.float64).T @ np.asarray(y_by_field[fid], dtype=np.float64)) for fid in range(n_fields)]

            total_sum_F = np.sum(np.stack(sums_F, axis=0), axis=0)
            total_sum_y = np.sum(np.stack(sums_y, axis=0), axis=0)
            total_FtF = np.sum(np.stack(FtF, axis=0), axis=0)
            total_FtY = np.sum(np.stack(FtY, axis=0), axis=0)

            n_total = int(n_fields) * int(patches_per_field)
            n_test = int(patches_per_field)
            n_train = n_total - n_test

            sum_F_tr = total_sum_F - sums_F[test_field]
            sum_y_tr = total_sum_y - sums_y[test_field]
            FtF_tr = total_FtF - FtF[test_field]
            FtY_tr = total_FtY - FtY[test_field]

            mu = sum_F_tr / float(n_train)
            y_mu = sum_y_tr / float(n_train)
            var = np.diag(FtF_tr) / float(n_train) - mu * mu
            var = np.where(var > 1e-12, var, 1e-12)
            sd = np.sqrt(var, dtype=np.float64)

            FtF_center = FtF_tr - float(n_train) * np.outer(mu, mu)
            FtY_center = FtY_tr - float(n_train) * (mu[:, None] * y_mu[None, :])

            XtX = FtF_center / (sd[:, None] * sd[None, :])
            Xty = FtY_center / sd[:, None]
            w = solve_ridge(XtX, Xty)

            Fte = np.asarray(F_by_field[test_field], dtype=np.float64)
            Xte = (Fte - mu) / sd
            return (Xte @ w + y_mu).astype(np.float64, copy=False)

        # Precompute rect projection low-k predictions for each field (no learning).
        y_low_rect_proj_by_field: list[np.ndarray] = []
        for fid in range(n_fields):
            F = F_rect_by_field[fid].astype(np.float64, copy=False)
            y_pred = np.column_stack([F[:, :n_bins_rect] @ coef_rect_x, F[:, n_bins_rect:] @ coef_rect_y]).astype(np.float64, copy=False)
            y_low_rect_proj_by_field.append(y_pred)

        # Pixel windows (for P_full).
        wins = [sliding_window_view(rhoz_by_field[fid], (w_big, w_big)) for fid in range(n_fields)]

        def train_pixel_full_model(test_field: int) -> tuple[np.ndarray, np.ndarray]:
            train_fields = [i for i in range(n_fields) if i != test_field]

            # Target normalization.
            y_train_all = np.concatenate([y_full_by_field[fid] for fid in train_fields], axis=0).astype(np.float64, copy=False)
            y_mu = y_train_all.mean(axis=0)
            y_sd = y_train_all.std(axis=0)
            y_sd = np.where(y_sd > 0, y_sd, 1.0)

            eta0_eff = float(pixel_eta0_base) / float(w_big)
            reg_gx = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(pixel_alpha),
                learning_rate="invscaling",
                eta0=float(eta0_eff),
                power_t=float(pixel_power_t),
                max_iter=1,
                tol=None,
                fit_intercept=True,
                average=True,
                random_state=int(placebo_seed) + 10_000 + 10 * test_field + 0,
            )
            reg_gy = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=float(pixel_alpha),
                learning_rate="invscaling",
                eta0=float(eta0_eff),
                power_t=float(pixel_power_t),
                max_iter=1,
                tol=None,
                fit_intercept=True,
                average=True,
                random_state=int(placebo_seed) + 10_000 + 10 * test_field + 1,
            )

            for epoch in range(1, int(pixel_epochs) + 1):
                for fid in train_fields:
                    centers = centers_by_field[fid]
                    y = y_full_by_field[fid].astype(np.float64, copy=False)
                    perm = np.random.default_rng(int(placebo_seed) + 20_000 + 100 * epoch + fid).permutation(centers.shape[0])
                    for start in range(0, centers.shape[0], pixel_batch_size):
                        idx = perm[start : start + pixel_batch_size]
                        cx = centers[idx, 0].astype(np.int64)
                        cy = centers[idx, 1].astype(np.int64)
                        Xb = wins[fid][cx - r_big, cy - r_big].reshape(len(idx), -1).astype(np.float32, copy=False)
                        yb = ((y[idx] - y_mu) / y_sd).astype(np.float64, copy=False)
                        reg_gx.partial_fit(Xb, yb[:, 0])
                        reg_gy.partial_fit(Xb, yb[:, 1])

            # Predict on test field.
            centers = centers_by_field[test_field]
            out = np.empty((centers.shape[0], 2), dtype=np.float64)
            for start in range(0, centers.shape[0], pixel_batch_size):
                idx = slice(start, min(centers.shape[0], start + pixel_batch_size))
                cx = centers[idx, 0].astype(np.int64)
                cy = centers[idx, 1].astype(np.int64)
                Xb = wins[test_field][cx - r_big, cy - r_big].reshape(len(cx), -1).astype(np.float32, copy=False)
                out[idx, 0] = reg_gx.predict(Xb)
                out[idx, 1] = reg_gy.predict(Xb)
            y_pred = out * y_sd + y_mu
            return y_pred.astype(np.float64, copy=False), y_mu.astype(np.float64, copy=False)

        # Run LOFO.
        fold_rows: list[dict[str, Any]] = []
        delta_rect_vs_pfull: list[float] = []
        delta_rel_rect_vs_pfull: list[float] = []
        placebo_pvals: list[float] = []

        for test_field in range(n_fields):
            y_full_true = y_full_by_field[test_field]

            # Pixels-full baseline.
            y_pfull, _ = train_pixel_full_model(test_field)
            p_pfull = pearson_mean_2d(y_full_true, y_pfull)
            rr_pfull = relrmse_mean_2d(y_full_true, y_pfull)

            # Two-channel components.
            y_high = y_high_kernel_pred_by_field[test_field]
            y_low_ceiling = y_low_ceiling_pred_by_field[test_field]
            y_low_rect_proj = y_low_rect_proj_by_field[test_field]
            y_low_ann = fit_predict_low(F_ann_by_field, y_low_by_field, test_field=test_field)
            y_low_rect = fit_predict_low(F_rect_by_field, y_low_by_field, test_field=test_field)

            y_full_ceiling = y_low_ceiling + y_high
            y_full_rect_proj = y_low_rect_proj + y_high
            y_full_rect = y_low_rect + y_high
            y_full_ann = y_low_ann + y_high

            # Metrics per model.
            p_ceiling = pearson_mean_2d(y_full_true, y_full_ceiling)
            rr_ceiling = relrmse_mean_2d(y_full_true, y_full_ceiling)
            p_rect_proj = pearson_mean_2d(y_full_true, y_full_rect_proj)
            rr_rect_proj = relrmse_mean_2d(y_full_true, y_full_rect_proj)
            p_rect = pearson_mean_2d(y_full_true, y_full_rect)
            rr_rect = relrmse_mean_2d(y_full_true, y_full_rect)
            p_ann = pearson_mean_2d(y_full_true, y_full_ann)
            rr_ann = relrmse_mean_2d(y_full_true, y_full_ann)

            dP_vs_pfull = float(p_rect - p_pfull)
            dR_vs_pfull = float(rr_rect - rr_pfull)
            delta_rect_vs_pfull.append(dP_vs_pfull)
            delta_rel_rect_vs_pfull.append(dR_vs_pfull)

            # Optional placebo: permute y_low in train for rect model and compute ΔPearson vs P_full.
            p_emp = float("nan")
            if rect_placebo_enabled:
                rng_fold = np.random.default_rng(int(placebo_seed) + 300_000 + 10_000 * test_field)
                deltas_perm = np.zeros((rect_placebo_perms,), dtype=np.float64)
                for j in range(rect_placebo_perms):
                    rng_pi = np.random.default_rng(rng_fold.integers(0, 2**32 - 1, dtype=np.uint64))
                    y_low_rect_perm = fit_predict_low(
                        F_rect_by_field,
                        y_low_by_field,
                        test_field=test_field,
                        permute_train_y=True,
                        perm_rng=rng_pi,
                    )
                    y_full_perm = y_low_rect_perm + y_high
                    p_perm = pearson_mean_2d(y_full_true, y_full_perm)
                    deltas_perm[j] = float(p_perm - p_pfull)
                p_emp = (float(np.sum(deltas_perm >= dP_vs_pfull)) + 1.0) / float(rect_placebo_perms + 1)
                placebo_pvals.append(p_emp)

            fold_rows.append(
                {
                    "field_id": int(test_field),
                    "Pearson_P_full": float(p_pfull),
                    "Pearson_twoch_rect": float(p_rect),
                    "Pearson_twoch_rect_proj": float(p_rect_proj),
                    "Pearson_twoch_ceiling": float(p_ceiling),
                    "Pearson_twoch_annulus": float(p_ann),
                    "ΔPearson_rect_minus_Pfull": float(dP_vs_pfull),
                    "ΔPearson_rect_minus_ceiling": float(p_rect - p_ceiling),
                    "ΔPearson_rectproj_minus_ceiling": float(p_rect_proj - p_ceiling),
                    "relRMSE_P_full": float(rr_pfull),
                    "relRMSE_twoch_rect": float(rr_rect),
                    "relRMSE_twoch_rect_proj": float(rr_rect_proj),
                    "relRMSE_twoch_ceiling": float(rr_ceiling),
                    "relRMSE_twoch_annulus": float(rr_ann),
                    "ΔrelRMSE_rect_minus_Pfull": float(dR_vs_pfull),
                    "ΔrelRMSE_rect_minus_ceiling": float(rr_rect - rr_ceiling),
                    "ΔrelRMSE_rectproj_minus_ceiling": float(rr_rect_proj - rr_ceiling),
                    "p_emp_placebo": float(p_emp),
                }
            )

        def mean_std(vals: list[float]) -> tuple[float, float]:
            arr = np.asarray(vals, dtype=np.float64)
            return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

        # Aggregate metrics per model across folds.
        def agg_model(key_p: str, key_r: str) -> tuple[float, float, float, float]:
            ps = np.asarray([r0[key_p] for r0 in fold_rows], dtype=np.float64)
            rs = np.asarray([r0[key_r] for r0 in fold_rows], dtype=np.float64)
            return float(ps.mean()), float(ps.std(ddof=1)) if len(ps) > 1 else 0.0, float(rs.mean()), float(rs.std(ddof=1)) if len(rs) > 1 else 0.0

        m_pfull = agg_model("Pearson_P_full", "relRMSE_P_full")
        m_rect = agg_model("Pearson_twoch_rect", "relRMSE_twoch_rect")
        m_rectproj = agg_model("Pearson_twoch_rect_proj", "relRMSE_twoch_rect_proj")
        m_ceiling = agg_model("Pearson_twoch_ceiling", "relRMSE_twoch_ceiling")

        dP_mean, dP_std = mean_std(delta_rect_vs_pfull)
        dR_mean, dR_std = mean_std(delta_rel_rect_vs_pfull)
        n_pos = int(np.sum(np.asarray(delta_rect_vs_pfull, dtype=np.float64) > 0))

        fisher_p = float("nan")
        if rect_placebo_enabled and placebo_pvals:
            pvals = np.clip(np.asarray(placebo_pvals, dtype=np.float64), 1e-300, 1.0)
            fisher_stat = float(-2.0 * np.sum(np.log(pvals)))
            fisher_p = float(chi2.sf(fisher_stat, df=2 * len(pvals)))

        passes = (dP_mean > 0) and (dR_mean < 0) and (n_pos >= pass_min_pos_folds) and (not rect_placebo_enabled or fisher_p < pass_fisher_p)

        # Markdown tables.
        def fmt_ms(m: tuple[float, float, float, float]) -> tuple[str, str]:
            return f"{m[0]:.4f} ± {m[1]:.4f}", f"{m[2]:.4f} ± {m[3]:.4f}"

        models_rows = []
        for name, m in [
            ("P_full_pixels", m_pfull),
            ("two_channel_rect_learn", m_rect),
            ("two_channel_rect_proj", m_rectproj),
            ("two_channel_ceiling", m_ceiling),
        ]:
            p_s, r_s = fmt_ms(m)
            models_rows.append({"model": name, "Pearson mean±std": p_s, "relRMSE mean±std": r_s})

        delta_rows = []
        # Δ vs P_full.
        for model_key in ["twoch_rect", "twoch_rect_proj", "twoch_ceiling"]:
            if model_key == "twoch_rect":
                key_p, key_r = "Pearson_twoch_rect", "relRMSE_twoch_rect"
                label = "two_channel_rect_learn"
            elif model_key == "twoch_rect_proj":
                key_p, key_r = "Pearson_twoch_rect_proj", "relRMSE_twoch_rect_proj"
                label = "two_channel_rect_proj"
            else:
                key_p, key_r = "Pearson_twoch_ceiling", "relRMSE_twoch_ceiling"
                label = "two_channel_ceiling"
            dp = np.asarray([r0[key_p] - r0["Pearson_P_full"] for r0 in fold_rows], dtype=np.float64)
            dr = np.asarray([r0[key_r] - r0["relRMSE_P_full"] for r0 in fold_rows], dtype=np.float64)
            delta_rows.append(
                {
                    "model": label,
                    "ΔPearson vs P_full": f"{float(dp.mean()):+.4f} ± {float(dp.std(ddof=1)):.4f}",
                    "ΔrelRMSE vs P_full": f"{float(dr.mean()):+.4f} ± {float(dr.std(ddof=1)):.4f}",
                }
            )

        # Δ vs ceiling.
        delta_rows_ceiling = []
        for label, key_p, key_r in [
            ("two_channel_rect_learn", "Pearson_twoch_rect", "relRMSE_twoch_rect"),
            ("two_channel_rect_proj", "Pearson_twoch_rect_proj", "relRMSE_twoch_rect_proj"),
        ]:
            dp = np.asarray([r0[key_p] - r0["Pearson_twoch_ceiling"] for r0 in fold_rows], dtype=np.float64)
            dr = np.asarray([r0[key_r] - r0["relRMSE_twoch_ceiling"] for r0 in fold_rows], dtype=np.float64)
            delta_rows_ceiling.append(
                {
                    "model": label,
                    "ΔPearson vs ceiling": f"{float(dp.mean()):+.4f} ± {float(dp.std(ddof=1)):.4f}",
                    "ΔrelRMSE vs ceiling": f"{float(dr.mean()):+.4f} ± {float(dr.std(ddof=1)):.4f}",
                }
            )

        md_var = [
            {
                "field_id": str(r0["field_id"]),
                "var_frac_mean": f"{float(r0['var_frac_mean']):.4f}",
            }
            for r0 in var_frac_rows
        ]

        md_folds = [
            {
                "field_id": str(r0["field_id"]),
                "Pearson_P_full": f"{r0['Pearson_P_full']:.4f}",
                "Pearson_rect": f"{r0['Pearson_twoch_rect']:.4f}",
                "Pearson_ceiling": f"{r0['Pearson_twoch_ceiling']:.4f}",
                "ΔP(rect-P_full)": f"{r0['ΔPearson_rect_minus_Pfull']:+.4f}",
                "relRMSE_P_full": f"{r0['relRMSE_P_full']:.4f}",
                "relRMSE_rect": f"{r0['relRMSE_twoch_rect']:.4f}",
                "relRMSE_ceiling": f"{r0['relRMSE_twoch_ceiling']:.4f}",
                "ΔR(rect-P_full)": f"{r0['ΔrelRMSE_rect_minus_Pfull']:+.4f}",
                "p_emp(placebo)": (f"{r0['p_emp_placebo']:.4f}" if np.isfinite(r0["p_emp_placebo"]) else ""),
            }
            for r0 in fold_rows
        ]

        summary_md = (
            "# E39 — Full-g LOFO baselines (pixels vs two-channel rect)\n\n"
            f"- run: `{paths.run_dir}`\n"
            f"- grid_size={grid_size}, alpha={alpha}, k0_frac={k0_frac}, n_fields={n_fields}, patches_per_field={patches_per_field}\n"
            f"- w_big={w_big}, w_local={w_local}\n"
            f"- low_rect: Nx={rect_nx},Ny={rect_ny} ; low_annulus: M={annulus_M}\n"
            f"- pixels baseline: SGD epochs={pixel_epochs}, batch={pixel_batch_size}, alpha={pixel_alpha}, eta0_base={pixel_eta0_base}, power_t={pixel_power_t}\n"
            + (f"- placebo: enabled, perms={rect_placebo_perms}, placebo_seed={placebo_seed}\n\n" if rect_placebo_enabled else "- placebo: disabled\n\n")
            + "## Low-k variance fraction (mean over gx/gy)\n\n"
            + md_table(md_var, ["field_id", "var_frac_mean"])
            + f"\n\n- var_frac_mean across fields: {var_frac_mean:.4f} ± {var_frac_std:.4f}\n\n"
            + "## Full-g metrics\n\n"
            + md_table(models_rows, ["model", "Pearson mean±std", "relRMSE mean±std"])
            + "\n\n## Δ vs P_full_pixels\n\n"
            + md_table(delta_rows, ["model", "ΔPearson vs P_full", "ΔrelRMSE vs P_full"])
            + "\n\n## Δ vs ceiling (ablation: learn vs proj)\n\n"
            + md_table(delta_rows_ceiling, ["model", "ΔPearson vs ceiling", "ΔrelRMSE vs ceiling"])
            + "\n\n## Per-fold table\n\n"
            + md_table(
                md_folds,
                [
                    "field_id",
                    "Pearson_P_full",
                    "Pearson_rect",
                    "Pearson_ceiling",
                    "ΔP(rect-P_full)",
                    "relRMSE_P_full",
                    "relRMSE_rect",
                    "relRMSE_ceiling",
                    "ΔR(rect-P_full)",
                    "p_emp(placebo)",
                ],
            )
            + "\n\n## Verdict (rect_learn vs P_full)\n\n"
            + f"- ΔPearson mean±std: {dP_mean:+.4f} ± {dP_std:.4f}\n"
            + f"- ΔrelRMSE mean±std: {dR_mean:+.4f} ± {dR_std:.4f}\n"
            + f"- #folds ΔPearson>0: {n_pos}/{n_fields}\n"
            + (f"- Fisher p-value (placebo ΔPearson): {fisher_p:.4g}\n" if rect_placebo_enabled else "")
            + f"- VERDICT: {'PASS' if passes else 'FAIL'}\n"
        )

        (paths.run_dir / "summary_e39_fullg_baselines_rectproj.md").write_text(summary_md, encoding="utf-8")
        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "grid_size": grid_size,
                "alpha": alpha,
                "k0_frac": k0_frac,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "w_big": int(w_big),
                "w_local": int(w_local),
                "low": {"rect_nx": int(rect_nx), "rect_ny": int(rect_ny), "annulus_M": int(annulus_M), "ring_bin_mode": ring_bin_mode},
                "ridge_alpha": ridge_alpha,
                "pixels": {
                    "epochs": pixel_epochs,
                    "batch_size": pixel_batch_size,
                    "alpha": pixel_alpha,
                    "eta0_base": pixel_eta0_base,
                    "power_t": pixel_power_t,
                },
                "placebo": {"enabled": rect_placebo_enabled, "perms": rect_placebo_perms, "seed": placebo_seed, "fisher_p": fisher_p},
                "var_frac_rows": var_frac_rows,
                "fold_rows": fold_rows,
                "verdict": {"pass": bool(passes), "dP_mean": dP_mean, "dR_mean": dR_mean, "n_pos": n_pos},
            },
        )
        return paths

    if experiment == "e40":
        # E40 — Diagnose why rect_learn beats ceiling in relRMSE: taper/apodization test on low-k kernel.
        #
        # Same setup as E39/E38:
        # - LOFO over 10 fields, rho 1/f alpha=2.0, n=256
        # - k0_frac low/high split, w_big=193 low-k support, w_local=33 high-k support
        #
        # Models:
        #  A) ceiling: low-k truncated impulse kernel + high-k kernel
        #  B) rect_learn: learned rectangular dipole low-k + high-k kernel (reference)
        #  C) taper_kernels: low-k kernel tapered (Tukey/Gaussian) + scalar rescale per fold (train-only) + high-k kernel
        import matplotlib.pyplot as plt

        grid_size = int(cfg.get("grid_size", 256))
        alpha = float(cfg.get("alpha", 2.0))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 10_000))
        k0_frac = float(cfg.get("k0_frac", 0.15))
        w_big = _require_odd("w_big", int(cfg.get("w_big", 193)))
        w_local = _require_odd("w_local", int(cfg.get("w_local", 33)))

        rect_nx = int(cfg.get("rect_nx", 16))
        rect_ny = int(cfg.get("rect_ny", 8))
        ridge_alpha = float(cfg.get("ridge_alpha", 1.0))

        # Taper grids.
        tukey_alphas = [float(x) for x in cfg.get("tukey_alphas", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])]
        gauss_sigma_fracs = [float(x) for x in cfg.get("gauss_sigma_fracs", [0.25, 0.35, 0.50, 0.70, 1.0])]
        radial_gauss_enabled = bool(cfg.get("radial_gauss_enabled", True))

        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if n_fields < 2:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        if not (0.0 < float(k0_frac) < 0.5):
            raise ValueError("k0_frac must be in (0,0.5)")
        if w_big > grid_size or w_local > grid_size:
            raise ValueError("w_big/w_local must be <= grid_size")
        if w_local > w_big:
            raise ValueError("w_local must be <= w_big")
        if rect_nx <= 0 or rect_ny <= 0:
            raise ValueError("rect_nx/rect_ny must be > 0")
        if ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be > 0")

        def safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            am = a - float(a.mean())
            bm = b - float(b.mean())
            denom = float(np.linalg.norm(am) * np.linalg.norm(bm)) + 1e-12
            return float((am @ bm) / denom)

        def relrmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
            e = y_pred - y_true
            rmse = float(np.sqrt(np.mean(e * e)))
            sd = float(np.std(y_true))
            return rmse / (sd + 1e-12)

        def pearson_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            px = safe_corr_1d(y_true[:, 0], y_pred[:, 0])
            py = safe_corr_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (px + py)

        def relrmse_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            rx = relrmse_1d(y_true[:, 0], y_pred[:, 0])
            ry = relrmse_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (rx + ry)

        def solve_ridge(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            d = int(A.shape[0])
            I = np.eye(d, dtype=np.float64)
            try:
                return np.linalg.solve(A + ridge_alpha * I, B)
            except np.linalg.LinAlgError:
                w, *_ = np.linalg.lstsq(A + ridge_alpha * I, B, rcond=None)
                return w

        def kernel_fft_centered(kernel: np.ndarray, *, grid_size: int) -> np.ndarray:
            kernel = np.asarray(kernel, dtype=np.float64)
            if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
                raise ValueError(f"kernel must be square 2D, got {kernel.shape}")
            w = int(kernel.shape[0])
            if (w % 2) == 0:
                raise ValueError(f"kernel size must be odd, got {w}")
            if w > grid_size:
                raise ValueError(f"kernel size {w} exceeds grid_size {grid_size}")
            r = w // 2
            kr = np.flip(kernel, axis=(0, 1))
            full = np.zeros((grid_size, grid_size), dtype=np.float64)
            cx = grid_size // 2
            cy = grid_size // 2
            full[cx - r : cx + r + 1, cy - r : cy + r + 1] = kr
            full0 = np.fft.ifftshift(full)
            return np.fft.fftn(full0)

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r0 in rows:
                out.append("| " + " | ".join(r0.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        def tukey_window(n: int, alpha_t: float) -> np.ndarray:
            n = int(n)
            if n <= 1:
                return np.ones((n,), dtype=np.float64)
            a = float(alpha_t)
            if a <= 0.0:
                return np.ones((n,), dtype=np.float64)
            if a >= 1.0:
                x = np.arange(n, dtype=np.float64) / float(n - 1)
                return 0.5 * (1.0 - np.cos(2.0 * np.pi * x))

            x = np.arange(n, dtype=np.float64) / float(n - 1)
            w = np.ones((n,), dtype=np.float64)
            t = a / 2.0
            left = x < t
            right = x > (1.0 - t)
            w[left] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * x[left] / a - 1.0)))
            w[right] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * x[right] / a - 2.0 / a + 1.0)))
            return w

        def gaussian_window_1d(n: int, sigma: float) -> np.ndarray:
            n = int(n)
            r = n // 2
            u = (np.arange(n, dtype=np.float64) - float(r)).astype(np.float64, copy=False)
            s = float(sigma)
            if s <= 0:
                raise ValueError("sigma must be > 0")
            return np.exp(-(u * u) / (2.0 * s * s), dtype=np.float64)

        def save_kernel_triptych(k0: np.ndarray, k1: np.ndarray, k2: np.ndarray, out: Path, title0: str, title1: str, title2: str) -> None:
            vmax = float(np.max(np.abs(np.stack([k0, k1, k2], axis=0)))) + 1e-12
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for ax, kk, tt in zip(axes, [k0, k1, k2], [title0, title1, title2]):
                im = ax.imshow(kk, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
                ax.set_title(tt)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)

        def sample_vec_at_centers(gx: np.ndarray, gy: np.ndarray, centers: np.ndarray) -> np.ndarray:
            cx = centers[:, 0].astype(np.int64)
            cy = centers[:, 1].astype(np.int64)
            return np.column_stack([gx[cx, cy], gy[cx, cy]]).astype(np.float64, copy=False)

        # Generate fields + band split.
        rho01_by_field: list[np.ndarray] = []
        rho0_by_field: list[np.ndarray] = []
        field_fft0_by_field: list[np.ndarray] = []
        field_fft_rho01_by_field: list[np.ndarray] = []
        split_cache: list[BandSplit2D] = []

        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=alpha, rng=rng_field)
            split = band_split_poisson_2d(rho01, k0_frac=float(k0_frac))
            rho01_by_field.append(rho01.astype(np.float64, copy=False))
            rho0 = (rho01 - float(rho01.mean())).astype(np.float64, copy=False)
            rho0_by_field.append(rho0)
            field_fft0_by_field.append(np.fft.fftn(rho0))
            field_fft_rho01_by_field.append(np.fft.fftn(rho01))
            split_cache.append(split)

        # Centers (margin w_big).
        r_big = w_big // 2
        centers_by_field: list[np.ndarray] = []
        centers_idx: list[tuple[np.ndarray, np.ndarray]] = []
        for field_id in range(n_fields):
            rng_cent = np.random.default_rng(seed + 555_555 + 10_000 * int(w_big) + 1_000 * field_id)
            cx = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
            cy = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
            centers = np.column_stack([cx, cy]).astype(np.int64, copy=False)
            centers_by_field.append(centers)
            centers_idx.append((cx.astype(np.int64, copy=False), cy.astype(np.int64, copy=False)))

        y_low_by_field = [sample_vec_at_centers(split_cache[fid].low.gx, split_cache[fid].low.gy, centers_by_field[fid]) for fid in range(n_fields)]
        y_full_by_field = [
            sample_vec_at_centers(split_cache[fid].full.gx, split_cache[fid].full.gy, centers_by_field[fid]) for fid in range(n_fields)
        ]

        # Impulse kernels for low/high.
        rho_delta = np.zeros((grid_size, grid_size), dtype=np.float64)
        c0 = grid_size // 2
        rho_delta[c0, c0] = 1.0
        split_delta = band_split_poisson_2d(rho_delta, k0_frac=float(k0_frac))

        r_loc = w_local // 2
        k_low_gx = split_delta.low.gx[c0 - r_big : c0 + r_big + 1, c0 - r_big : c0 + r_big + 1].astype(np.float64, copy=False)
        k_low_gy = split_delta.low.gy[c0 - r_big : c0 + r_big + 1, c0 - r_big : c0 + r_big + 1].astype(np.float64, copy=False)
        k_high_gx = split_delta.high.gx[c0 - r_loc : c0 + r_loc + 1, c0 - r_loc : c0 + r_loc + 1].astype(np.float64, copy=False)
        k_high_gy = split_delta.high.gy[c0 - r_loc : c0 + r_loc + 1, c0 - r_loc : c0 + r_loc + 1].astype(np.float64, copy=False)

        # FFT kernels for ceiling/high.
        kfft_low_gx = kernel_fft_centered(k_low_gx[::-1, ::-1], grid_size=grid_size)
        kfft_low_gy = kernel_fft_centered(k_low_gy[::-1, ::-1], grid_size=grid_size)
        kfft_high_gx = kernel_fft_centered(k_high_gx[::-1, ::-1], grid_size=grid_size)
        kfft_high_gy = kernel_fft_centered(k_high_gy[::-1, ::-1], grid_size=grid_size)

        # Fixed channel predictions: ceiling low + kernel high.
        y_low_ceiling_pred_by_field: list[np.ndarray] = []
        y_high_pred_by_field: list[np.ndarray] = []
        for fid in range(n_fields):
            field_fft0 = field_fft0_by_field[fid]
            gx_low_pred = np.fft.ifftn(field_fft0 * kfft_low_gx).real
            gy_low_pred = np.fft.ifftn(field_fft0 * kfft_low_gy).real
            y_low_ceiling_pred_by_field.append(sample_vec_at_centers(gx_low_pred, gy_low_pred, centers_by_field[fid]))

            gx_high_pred = np.fft.ifftn(field_fft0 * kfft_high_gx).real
            gy_high_pred = np.fft.ifftn(field_fft0 * kfft_high_gy).real
            y_high_pred_by_field.append(sample_vec_at_centers(gx_high_pred, gy_high_pred, centers_by_field[fid]))

        # Baseline: ceiling full predictions.
        y_full_ceiling_pred_by_field = [y_low_ceiling_pred_by_field[fid] + y_high_pred_by_field[fid] for fid in range(n_fields)]

        # Rect features (computed once).
        def candidate_edges(r: int) -> np.ndarray:
            base = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, r + 1]
            base = [int(x) for x in base if 0 <= int(x) <= r + 1]
            dense = list(range(0, min(16, r + 1) + 1))
            geom = np.unique(np.rint(np.geomspace(1, r + 1, num=64)).astype(int)).tolist()
            vals = sorted(set(base + dense + geom + [0, r + 1]))
            return np.asarray(vals, dtype=np.int64)

        def pick_edges(cand: np.ndarray, n_bins: int) -> np.ndarray:
            cand = np.asarray(cand, dtype=np.int64)
            if len(cand) < n_bins + 1:
                cand = np.arange(int(cand.min()), int(cand.max()) + 1, dtype=np.int64)
            idx = np.round(np.linspace(0, len(cand) - 1, n_bins + 1)).astype(int)
            idx[0] = 0
            idx[-1] = len(cand) - 1
            edges = np.unique(cand[idx])
            if len(edges) != n_bins + 1:
                edges = np.unique(np.rint(np.linspace(int(cand.min()), int(cand.max()), n_bins + 1)).astype(np.int64))
            edges[0] = int(cand.min())
            edges[-1] = int(cand.max())
            return edges

        def build_rect_bins(w: int, *, nx_bins: int, ny_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            w = _require_odd("w_big", int(w))
            r = w // 2
            coords = np.arange(w, dtype=np.float64) - float(r)
            dx, dy = np.meshgrid(coords, coords, indexing="ij")
            ax = np.abs(dx).astype(np.int64)
            ay = np.abs(dy).astype(np.int64)
            cand = candidate_edges(int(r))
            edges_x = pick_edges(cand, int(nx_bins))
            edges_y = pick_edges(cand, int(ny_bins))
            ix = np.digitize(ax, edges_x, right=False) - 1
            iy = np.digitize(ay, edges_y, right=False) - 1
            ix = np.clip(ix, 0, int(nx_bins) - 1).astype(np.int32, copy=False)
            iy = np.clip(iy, 0, int(ny_bins) - 1).astype(np.int32, copy=False)
            bin_idx = (ix * int(ny_bins) + iy).astype(np.int32, copy=False)
            return dx.astype(np.float64), dy.astype(np.float64), bin_idx

        dxr, dyr, bin_idx = build_rect_bins(w_big, nx_bins=int(rect_nx), ny_bins=int(rect_ny))
        n_bins_rect = int(rect_nx) * int(rect_ny)
        d_rect = 2 * n_bins_rect
        F_rect_by_field: list[np.ndarray] = [np.zeros((patches_per_field, d_rect), dtype=np.float32) for _ in range(n_fields)]
        for b in range(n_bins_rect):
            mask = (bin_idx == int(b)).astype(np.float64, copy=False)
            kdx = (dxr * mask).astype(np.float64, copy=False)
            kdy = (dyr * mask).astype(np.float64, copy=False)
            kfft_dx = kernel_fft_centered(kdx, grid_size=grid_size)
            kfft_dy = kernel_fft_centered(kdy, grid_size=grid_size)
            for fid in range(n_fields):
                field_fft = field_fft_rho01_by_field[fid]
                cx, cy = centers_idx[fid]
                dx_grid = np.fft.ifftn(field_fft * kfft_dx).real
                dy_grid = np.fft.ifftn(field_fft * kfft_dy).real
                F_rect_by_field[fid][:, b] = dx_grid[cx, cy].astype(np.float32, copy=False)
                F_rect_by_field[fid][:, n_bins_rect + b] = dy_grid[cx, cy].astype(np.float32, copy=False)

        def ridge_fit_params(F_by_field: list[np.ndarray], y_by_field: list[np.ndarray], *, test_field: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            sums_F = [np.asarray(F_by_field[fid], dtype=np.float64).sum(axis=0) for fid in range(n_fields)]
            sums_y = [np.asarray(y_by_field[fid], dtype=np.float64).sum(axis=0) for fid in range(n_fields)]
            FtF = [(np.asarray(F_by_field[fid], dtype=np.float64).T @ np.asarray(F_by_field[fid], dtype=np.float64)) for fid in range(n_fields)]
            FtY = [(np.asarray(F_by_field[fid], dtype=np.float64).T @ np.asarray(y_by_field[fid], dtype=np.float64)) for fid in range(n_fields)]

            total_sum_F = np.sum(np.stack(sums_F, axis=0), axis=0)
            total_sum_y = np.sum(np.stack(sums_y, axis=0), axis=0)
            total_FtF = np.sum(np.stack(FtF, axis=0), axis=0)
            total_FtY = np.sum(np.stack(FtY, axis=0), axis=0)

            n_total = int(n_fields) * int(patches_per_field)
            n_test = int(patches_per_field)
            n_train = n_total - n_test

            sum_F_tr = total_sum_F - sums_F[test_field]
            sum_y_tr = total_sum_y - sums_y[test_field]
            FtF_tr = total_FtF - FtF[test_field]
            FtY_tr = total_FtY - FtY[test_field]

            mu = sum_F_tr / float(n_train)
            y_mu = sum_y_tr / float(n_train)
            var = np.diag(FtF_tr) / float(n_train) - mu * mu
            var = np.where(var > 1e-12, var, 1e-12)
            sd = np.sqrt(var, dtype=np.float64)

            FtF_center = FtF_tr - float(n_train) * np.outer(mu, mu)
            FtY_center = FtY_tr - float(n_train) * (mu[:, None] * y_mu[None, :])

            XtX = FtF_center / (sd[:, None] * sd[None, :])
            Xty = FtY_center / sd[:, None]
            w = solve_ridge(XtX, Xty)
            return mu, sd, w, y_mu

        def ridge_predict(F: np.ndarray, mu: np.ndarray, sd: np.ndarray, w: np.ndarray, y_mu: np.ndarray) -> np.ndarray:
            X = (np.asarray(F, dtype=np.float64) - mu) / sd
            return (X @ w + y_mu).astype(np.float64, copy=False)

        # Rect-learn full predictions per fold.
        fold_rows: list[dict[str, float]] = []
        per_fold_p_ceiling: list[float] = []
        per_fold_r_ceiling: list[float] = []
        per_fold_p_rect: list[float] = []
        per_fold_r_rect: list[float] = []

        for test_field in range(n_fields):
            y_true = y_full_by_field[test_field]

            y_pred_ceiling = y_full_ceiling_pred_by_field[test_field]
            p_ceiling = pearson_mean_2d(y_true, y_pred_ceiling)
            r_ceiling = relrmse_mean_2d(y_true, y_pred_ceiling)
            per_fold_p_ceiling.append(p_ceiling)
            per_fold_r_ceiling.append(r_ceiling)

            mu, sd, w, y_mu = ridge_fit_params(F_rect_by_field, y_low_by_field, test_field=test_field)
            y_low_rect = ridge_predict(F_rect_by_field[test_field], mu, sd, w, y_mu)
            y_pred_rect = y_low_rect + y_high_pred_by_field[test_field]
            p_rect = pearson_mean_2d(y_true, y_pred_rect)
            r_rect = relrmse_mean_2d(y_true, y_pred_rect)
            per_fold_p_rect.append(p_rect)
            per_fold_r_rect.append(r_rect)

            fold_rows.append(
                {
                    "field_id": float(test_field),
                    "Pearson_ceiling": p_ceiling,
                    "relRMSE_ceiling": r_ceiling,
                    "Pearson_rect": p_rect,
                    "relRMSE_rect": r_rect,
                    "ΔPearson(rect-ceiling)": p_rect - p_ceiling,
                    "ΔrelRMSE(rect-ceiling)": r_rect - r_ceiling,
                }
            )

        p_ceiling_mean = float(np.mean(per_fold_p_ceiling))
        p_ceiling_std = float(np.std(per_fold_p_ceiling, ddof=1))
        r_ceiling_mean = float(np.mean(per_fold_r_ceiling))
        r_ceiling_std = float(np.std(per_fold_r_ceiling, ddof=1))
        p_rect_mean = float(np.mean(per_fold_p_rect))
        p_rect_std = float(np.std(per_fold_p_rect, ddof=1))
        r_rect_mean = float(np.mean(per_fold_r_rect))
        r_rect_std = float(np.std(per_fold_r_rect, ddof=1))

        # Taper candidates.
        coords = np.arange(int(w_big), dtype=np.float64) - float(r_big)
        uu, vv = np.meshgrid(coords, coords, indexing="ij")

        taper_specs: list[tuple[str, str, np.ndarray]] = []
        for a in tukey_alphas:
            if a < 0.0 or a > 1.0:
                raise ValueError(f"tukey_alphas must be in [0,1], got {a}")
            w1 = tukey_window(w_big, a)
            W = (w1[:, None] * w1[None, :]).astype(np.float64, copy=False)
            taper_specs.append(("tukey", f"alpha={a:.2f}", W))

        for sf in gauss_sigma_fracs:
            if sf <= 0.0:
                raise ValueError(f"gauss_sigma_fracs must be > 0, got {sf}")
            sigma = float(sf) * float(r_big)
            w1 = gaussian_window_1d(w_big, sigma)
            W = (w1[:, None] * w1[None, :]).astype(np.float64, copy=False)
            taper_specs.append(("gauss_sep", f"sigma_frac={sf:.2f}", W))
            if radial_gauss_enabled:
                Wr = np.exp(-((uu * uu + vv * vv) / (2.0 * sigma * sigma)), dtype=np.float64)
                taper_specs.append(("gauss_rad", f"sigma_frac={sf:.2f}", Wr))

        # Evaluate taper variants.
        taper_rows: list[dict[str, str]] = []
        best_idx = -1
        best_relrmse = float("inf")
        best_metrics: tuple[float, float, float, float] | None = None
        best_a_fold0 = 1.0

        for i, (family, param, W) in enumerate(taper_specs):
            kx_t = (k_low_gx * W).astype(np.float64, copy=False)
            ky_t = (k_low_gy * W).astype(np.float64, copy=False)
            kfft_t_gx = kernel_fft_centered(kx_t[::-1, ::-1], grid_size=grid_size)
            kfft_t_gy = kernel_fft_centered(ky_t[::-1, ::-1], grid_size=grid_size)

            # Precompute low predictions for this taper.
            y_low_pred_by_field: list[np.ndarray] = []
            for fid in range(n_fields):
                field_fft0 = field_fft0_by_field[fid]
                gx_pred = np.fft.ifftn(field_fft0 * kfft_t_gx).real
                gy_pred = np.fft.ifftn(field_fft0 * kfft_t_gy).real
                y_low_pred_by_field.append(sample_vec_at_centers(gx_pred, gy_pred, centers_by_field[fid]))

            p_folds: list[float] = []
            r_folds: list[float] = []
            dp_folds: list[float] = []
            dr_folds: list[float] = []
            a_folds: list[float] = []

            for test_field in range(n_fields):
                train_fields = [f for f in range(n_fields) if f != test_field]
                y_tr = np.concatenate([y_low_by_field[f] for f in train_fields], axis=0)
                p_tr = np.concatenate([y_low_pred_by_field[f] for f in train_fields], axis=0)

                num = float(np.sum(p_tr[:, 0] * y_tr[:, 0] + p_tr[:, 1] * y_tr[:, 1]))
                den = float(np.sum(p_tr[:, 0] * p_tr[:, 0] + p_tr[:, 1] * p_tr[:, 1])) + 1e-12
                a = num / den
                a_folds.append(a)

                y_pred_full = a * y_low_pred_by_field[test_field] + y_high_pred_by_field[test_field]
                y_true_full = y_full_by_field[test_field]
                p = pearson_mean_2d(y_true_full, y_pred_full)
                r = relrmse_mean_2d(y_true_full, y_pred_full)
                p_folds.append(p)
                r_folds.append(r)
                dp_folds.append(p - per_fold_p_ceiling[test_field])
                dr_folds.append(r - per_fold_r_ceiling[test_field])

            p_mean = float(np.mean(p_folds))
            p_std = float(np.std(p_folds, ddof=1))
            r_mean = float(np.mean(r_folds))
            r_std = float(np.std(r_folds, ddof=1))
            dp_mean = float(np.mean(dp_folds))
            dp_std = float(np.std(dp_folds, ddof=1))
            dr_mean = float(np.mean(dr_folds))
            dr_std = float(np.std(dr_folds, ddof=1))

            taper_rows.append(
                {
                    "taper": family,
                    "param": param,
                    "Pearson mean±std": f"{p_mean:.4f} ± {p_std:.4f}",
                    "relRMSE mean±std": f"{r_mean:.4f} ± {r_std:.4f}",
                    "ΔPearson vs ceiling": f"{dp_mean:+.4f} ± {dp_std:.4f}",
                    "ΔrelRMSE vs ceiling": f"{dr_mean:+.4f} ± {dr_std:.4f}",
                }
            )

            if r_mean < best_relrmse:
                best_relrmse = r_mean
                best_idx = i
                best_metrics = (p_mean, p_std, r_mean, r_std)
                best_a_fold0 = float(a_folds[0])

        if best_idx < 0 or best_metrics is None:
            raise RuntimeError("no taper variants evaluated")

        best_family, best_param, best_W = taper_specs[best_idx]
        best_p_mean, best_p_std, best_r_mean, best_r_std = best_metrics

        # Compare best taper to rect_learn (fold-wise deltas).
        best_delta_p = best_p_mean - p_rect_mean
        best_delta_r = best_r_mean - r_rect_mean

        # Kernel plots for ceiling vs best taper vs rect_learn (fold 0 weights).
        mu0, sd0, w0, ymu0 = ridge_fit_params(F_rect_by_field, y_low_by_field, test_field=0)
        beta0 = w0 / sd0[:, None]
        beta_dx_to_gx = beta0[:n_bins_rect, 0]
        beta_dy_to_gx = beta0[n_bins_rect:, 0]
        beta_dx_to_gy = beta0[:n_bins_rect, 1]
        beta_dy_to_gy = beta0[n_bins_rect:, 1]

        bin_flat = bin_idx.astype(np.int64, copy=False)
        k_rect_gx = dxr * beta_dx_to_gx[bin_flat] + dyr * beta_dy_to_gx[bin_flat]
        k_rect_gy = dxr * beta_dx_to_gy[bin_flat] + dyr * beta_dy_to_gy[bin_flat]

        k_best_gx = (best_a_fold0 * k_low_gx * best_W).astype(np.float64, copy=False)
        k_best_gy = (best_a_fold0 * k_low_gy * best_W).astype(np.float64, copy=False)

        save_kernel_triptych(
            k_low_gx,
            k_best_gx,
            k_rect_gx,
            paths.run_dir / "kernel_compare_low_kx.png",
            "ceiling Kx_low",
            f"best taper ({best_family}, {best_param})",
            "rect_learn (fold0)",
        )
        save_kernel_triptych(
            k_low_gy,
            k_best_gy,
            k_rect_gy,
            paths.run_dir / "kernel_compare_low_ky.png",
            "ceiling Ky_low",
            f"best taper ({best_family}, {best_param})",
            "rect_learn (fold0)",
        )

        baseline_rows = [
            {"model": "two_channel_ceiling", "Pearson mean±std": f"{p_ceiling_mean:.4f} ± {p_ceiling_std:.4f}", "relRMSE mean±std": f"{r_ceiling_mean:.4f} ± {r_ceiling_std:.4f}"},
            {"model": "two_channel_rect_learn", "Pearson mean±std": f"{p_rect_mean:.4f} ± {p_rect_std:.4f}", "relRMSE mean±std": f"{r_rect_mean:.4f} ± {r_rect_std:.4f}"},
        ]

        # Best taper vs ceiling deltas.
        best_dp = best_p_mean - p_ceiling_mean
        best_dr = best_r_mean - r_ceiling_mean

        summary_md = (
            "# E40 — Low-k taper/apodization diagnostic (LOFO full g)\n\n"
            f"- run: `{paths.run_dir}`\n"
            f"- grid_size={grid_size}, alpha={alpha}, k0_frac={k0_frac}, n_fields={n_fields}, patches_per_field={patches_per_field}\n"
            f"- w_big={w_big}, w_local={w_local}\n"
            f"- rect_learn: Nx={rect_nx},Ny={rect_ny}, ridge_alpha={ridge_alpha}\n\n"
            "## Baselines\n\n"
            + md_table(baseline_rows, ["model", "Pearson mean±std", "relRMSE mean±std"])
            + "\n\n## Taper sweep (metrics on full g; Δ vs ceiling)\n\n"
            + md_table(
                taper_rows,
                ["taper", "param", "Pearson mean±std", "relRMSE mean±std", "ΔPearson vs ceiling", "ΔrelRMSE vs ceiling"],
            )
            + "\n\n## Best taper (by mean relRMSE)\n\n"
            f"- best: {best_family} {best_param}\n"
            f"- full-g: Pearson {best_p_mean:.4f} ± {best_p_std:.4f} ; relRMSE {best_r_mean:.4f} ± {best_r_std:.4f}\n"
            f"- Δ vs ceiling: ΔPearson {best_dp:+.4f} ; ΔrelRMSE {best_dr:+.4f}\n"
            f"- Δ(best - rect_learn): ΔPearson {best_delta_p:+.4f} ; ΔrelRMSE {best_delta_r:+.4f}\n\n"
            "## Kernel plots\n\n"
            f"- `kernel_compare_low_kx.png` (ceiling vs best_taper vs rect_learn)\n"
            f"- `kernel_compare_low_ky.png` (ceiling vs best_taper vs rect_learn)\n"
        )

        (paths.run_dir / "summary_e40_lowk_taper_diag.md").write_text(summary_md, encoding="utf-8")
        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "grid_size": grid_size,
                "alpha": alpha,
                "k0_frac": k0_frac,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "w_big": int(w_big),
                "w_local": int(w_local),
                "baseline": {"ceiling": {"pearson_mean": p_ceiling_mean, "relRMSE_mean": r_ceiling_mean}, "rect_learn": {"pearson_mean": p_rect_mean, "relRMSE_mean": r_rect_mean}},
                "best_taper": {
                    "family": best_family,
                    "param": best_param,
                    "pearson_mean": best_p_mean,
                    "relRMSE_mean": best_r_mean,
                    "dPearson_vs_ceiling": best_dp,
                    "drelRMSE_vs_ceiling": best_dr,
                    "dPearson_vs_rect": best_delta_p,
                    "drelRMSE_vs_rect": best_delta_r,
                    "a_fold0": best_a_fold0,
                },
            },
        )
        return paths

    if experiment == "e41":
        # E41 — Alpha sweep: does rect_learn beat ceiling only when rho is correlated?
        #
        # Hypothesis: rect_learn improves over the truncated impulse-kernel ceiling by conditional inference of
        # outside-support context via correlations in rho. Therefore, the gain should vanish as alpha -> 0.
        #
        # Setup matches E39:
        # - n=256, k0_frac=0.15, LOFO over 10 fields
        # - w_big=193 for low-k, w_local=33 for high-k
        # Models:
        #  - two_channel_ceiling: low=truncated impulse kernel, high=fixed kernel
        #  - two_channel_rect_proj: low=projection of kernel onto rect dipole basis (optional, deterministic)
        #  - two_channel_rect_learn: learned rect dipole low-k (ridge), high=fixed kernel
        alpha_list = [float(x) for x in cfg.get("alpha_list", [0.0, 1.0, 1.5, 2.0])]

        grid_size = int(cfg.get("grid_size", 256))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 10_000))
        k0_frac = float(cfg.get("k0_frac", 0.15))
        w_big = _require_odd("w_big", int(cfg.get("w_big", 193)))
        w_local = _require_odd("w_local", int(cfg.get("w_local", 33)))

        rect_nx = int(cfg.get("rect_nx", 16))
        rect_ny = int(cfg.get("rect_ny", 8))
        ridge_alpha = float(cfg.get("ridge_alpha", 1.0))

        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if n_fields < 2:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        if not (0.0 < float(k0_frac) < 0.5):
            raise ValueError("k0_frac must be in (0,0.5)")
        if w_big > grid_size or w_local > grid_size:
            raise ValueError("w_big/w_local must be <= grid_size")
        if w_local > w_big:
            raise ValueError("w_local must be <= w_big")
        if rect_nx <= 0 or rect_ny <= 0:
            raise ValueError("rect_nx/rect_ny must be > 0")
        if ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be > 0")
        if not alpha_list:
            raise ValueError("alpha_list must be non-empty")
        if any(a < 0 for a in alpha_list):
            raise ValueError("alpha_list must be >= 0")

        def safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            am = a - float(a.mean())
            bm = b - float(b.mean())
            denom = float(np.linalg.norm(am) * np.linalg.norm(bm)) + 1e-12
            return float((am @ bm) / denom)

        def relrmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
            e = y_pred - y_true
            rmse = float(np.sqrt(np.mean(e * e)))
            sd = float(np.std(y_true))
            return rmse / (sd + 1e-12)

        def pearson_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            px = safe_corr_1d(y_true[:, 0], y_pred[:, 0])
            py = safe_corr_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (px + py)

        def relrmse_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            rx = relrmse_1d(y_true[:, 0], y_pred[:, 0])
            ry = relrmse_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (rx + ry)

        def solve_ridge(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            d = int(A.shape[0])
            I = np.eye(d, dtype=np.float64)
            try:
                return np.linalg.solve(A + ridge_alpha * I, B)
            except np.linalg.LinAlgError:
                w, *_ = np.linalg.lstsq(A + ridge_alpha * I, B, rcond=None)
                return w

        def kernel_fft_centered(kernel: np.ndarray, *, grid_size: int) -> np.ndarray:
            kernel = np.asarray(kernel, dtype=np.float64)
            if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
                raise ValueError(f"kernel must be square 2D, got {kernel.shape}")
            w = int(kernel.shape[0])
            if (w % 2) == 0:
                raise ValueError(f"kernel size must be odd, got {w}")
            if w > grid_size:
                raise ValueError(f"kernel size {w} exceeds grid_size {grid_size}")
            r = w // 2
            kr = np.flip(kernel, axis=(0, 1))
            full = np.zeros((grid_size, grid_size), dtype=np.float64)
            cx = grid_size // 2
            cy = grid_size // 2
            full[cx - r : cx + r + 1, cy - r : cy + r + 1] = kr
            full0 = np.fft.ifftshift(full)
            return np.fft.fftn(full0)

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r0 in rows:
                out.append("| " + " | ".join(r0.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        # Precompute shared patch centers (same across alphas).
        r_big = w_big // 2
        centers_by_field: list[np.ndarray] = []
        centers_idx: list[tuple[np.ndarray, np.ndarray]] = []
        for field_id in range(n_fields):
            rng_cent = np.random.default_rng(seed + 555_555 + 10_000 * int(w_big) + 1_000 * field_id)
            cx = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
            cy = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
            centers = np.column_stack([cx, cy]).astype(np.int64, copy=False)
            centers_by_field.append(centers)
            centers_idx.append((cx.astype(np.int64, copy=False), cy.astype(np.int64, copy=False)))

        def sample_vec_at_centers(gx: np.ndarray, gy: np.ndarray, centers: np.ndarray) -> np.ndarray:
            cx = centers[:, 0].astype(np.int64)
            cy = centers[:, 1].astype(np.int64)
            return np.column_stack([gx[cx, cy], gy[cx, cy]]).astype(np.float64, copy=False)

        # Impulse kernels (independent of alpha).
        rho_delta = np.zeros((grid_size, grid_size), dtype=np.float64)
        c0 = grid_size // 2
        rho_delta[c0, c0] = 1.0
        split_delta = band_split_poisson_2d(rho_delta, k0_frac=float(k0_frac))

        r_loc = w_local // 2
        g_patch_low_gx = split_delta.low.gx[c0 - r_big : c0 + r_big + 1, c0 - r_big : c0 + r_big + 1]
        g_patch_low_gy = split_delta.low.gy[c0 - r_big : c0 + r_big + 1, c0 - r_big : c0 + r_big + 1]
        kcorr_low_gx = g_patch_low_gx[::-1, ::-1].astype(np.float64, copy=False)
        kcorr_low_gy = g_patch_low_gy[::-1, ::-1].astype(np.float64, copy=False)
        kfft_low_gx = kernel_fft_centered(kcorr_low_gx, grid_size=grid_size)
        kfft_low_gy = kernel_fft_centered(kcorr_low_gy, grid_size=grid_size)

        g_patch_high_gx = split_delta.high.gx[c0 - r_loc : c0 + r_loc + 1, c0 - r_loc : c0 + r_loc + 1]
        g_patch_high_gy = split_delta.high.gy[c0 - r_loc : c0 + r_loc + 1, c0 - r_loc : c0 + r_loc + 1]
        kcorr_high_gx = g_patch_high_gx[::-1, ::-1].astype(np.float64, copy=False)
        kcorr_high_gy = g_patch_high_gy[::-1, ::-1].astype(np.float64, copy=False)
        kfft_high_gx = kernel_fft_centered(kcorr_high_gx, grid_size=grid_size)
        kfft_high_gy = kernel_fft_centered(kcorr_high_gy, grid_size=grid_size)

        # Rect bins + projection coefficients (from low kernel correlation onto basis).
        def candidate_edges(r: int) -> np.ndarray:
            base = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, r + 1]
            base = [int(x) for x in base if 0 <= int(x) <= r + 1]
            dense = list(range(0, min(16, r + 1) + 1))
            geom = np.unique(np.rint(np.geomspace(1, r + 1, num=64)).astype(int)).tolist()
            vals = sorted(set(base + dense + geom + [0, r + 1]))
            return np.asarray(vals, dtype=np.int64)

        def pick_edges(cand: np.ndarray, n_bins: int) -> np.ndarray:
            cand = np.asarray(cand, dtype=np.int64)
            if len(cand) < n_bins + 1:
                cand = np.arange(int(cand.min()), int(cand.max()) + 1, dtype=np.int64)
            idx = np.round(np.linspace(0, len(cand) - 1, n_bins + 1)).astype(int)
            idx[0] = 0
            idx[-1] = len(cand) - 1
            edges = np.unique(cand[idx])
            if len(edges) != n_bins + 1:
                edges = np.unique(np.rint(np.linspace(int(cand.min()), int(cand.max()), n_bins + 1)).astype(np.int64))
            edges[0] = int(cand.min())
            edges[-1] = int(cand.max())
            return edges

        def build_rect_bins(w: int, *, nx_bins: int, ny_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            w = _require_odd("w_big", int(w))
            r = w // 2
            coords = np.arange(w, dtype=np.float64) - float(r)
            dx, dy = np.meshgrid(coords, coords, indexing="ij")
            ax = np.abs(dx).astype(np.int64)
            ay = np.abs(dy).astype(np.int64)
            cand = candidate_edges(int(r))
            edges_x = pick_edges(cand, int(nx_bins))
            edges_y = pick_edges(cand, int(ny_bins))
            ix = np.digitize(ax, edges_x, right=False) - 1
            iy = np.digitize(ay, edges_y, right=False) - 1
            ix = np.clip(ix, 0, int(nx_bins) - 1).astype(np.int32, copy=False)
            iy = np.clip(iy, 0, int(ny_bins) - 1).astype(np.int32, copy=False)
            bin_idx = (ix * int(ny_bins) + iy).astype(np.int32, copy=False)
            return dx.astype(np.float64), dy.astype(np.float64), bin_idx

        dxr, dyr, bin_idx = build_rect_bins(w_big, nx_bins=int(rect_nx), ny_bins=int(rect_ny))
        n_bins_rect = int(rect_nx) * int(rect_ny)

        bin_flat = bin_idx.reshape(-1).astype(np.int64, copy=False)
        dx_flat = dxr.reshape(-1).astype(np.float64, copy=False)
        dy_flat = dyr.reshape(-1).astype(np.float64, copy=False)
        kx_flat = kcorr_low_gx.reshape(-1).astype(np.float64, copy=False)
        ky_flat = kcorr_low_gy.reshape(-1).astype(np.float64, copy=False)
        num_x = np.bincount(bin_flat, weights=kx_flat * dx_flat, minlength=n_bins_rect).astype(np.float64, copy=False)
        den_x = np.bincount(bin_flat, weights=dx_flat * dx_flat, minlength=n_bins_rect).astype(np.float64, copy=False)
        coef_rect_x = np.where(den_x > 0, num_x / (den_x + 1e-12), 0.0).astype(np.float64, copy=False)
        num_y = np.bincount(bin_flat, weights=ky_flat * dy_flat, minlength=n_bins_rect).astype(np.float64, copy=False)
        den_y = np.bincount(bin_flat, weights=dy_flat * dy_flat, minlength=n_bins_rect).astype(np.float64, copy=False)
        coef_rect_y = np.where(den_y > 0, num_y / (den_y + 1e-12), 0.0).astype(np.float64, copy=False)

        def fit_predict_low_rect(
            F_by_field: list[np.ndarray],
            y_by_field: list[np.ndarray],
            *,
            test_field: int,
        ) -> np.ndarray:
            sums_F = [np.asarray(F_by_field[fid], dtype=np.float64).sum(axis=0) for fid in range(n_fields)]
            sums_y = [np.asarray(y_by_field[fid], dtype=np.float64).sum(axis=0) for fid in range(n_fields)]
            FtF = [(np.asarray(F_by_field[fid], dtype=np.float64).T @ np.asarray(F_by_field[fid], dtype=np.float64)) for fid in range(n_fields)]
            FtY = [(np.asarray(F_by_field[fid], dtype=np.float64).T @ np.asarray(y_by_field[fid], dtype=np.float64)) for fid in range(n_fields)]

            total_sum_F = np.sum(np.stack(sums_F, axis=0), axis=0)
            total_sum_y = np.sum(np.stack(sums_y, axis=0), axis=0)
            total_FtF = np.sum(np.stack(FtF, axis=0), axis=0)
            total_FtY = np.sum(np.stack(FtY, axis=0), axis=0)

            n_total = int(n_fields) * int(patches_per_field)
            n_test = int(patches_per_field)
            n_train = n_total - n_test

            sum_F_tr = total_sum_F - sums_F[test_field]
            sum_y_tr = total_sum_y - sums_y[test_field]
            FtF_tr = total_FtF - FtF[test_field]
            FtY_tr = total_FtY - FtY[test_field]

            mu = sum_F_tr / float(n_train)
            y_mu = sum_y_tr / float(n_train)
            var = np.diag(FtF_tr) / float(n_train) - mu * mu
            var = np.where(var > 1e-12, var, 1e-12)
            sd = np.sqrt(var, dtype=np.float64)

            FtF_center = FtF_tr - float(n_train) * np.outer(mu, mu)
            FtY_center = FtY_tr - float(n_train) * (mu[:, None] * y_mu[None, :])

            XtX = FtF_center / (sd[:, None] * sd[None, :])
            Xty = FtY_center / sd[:, None]
            w = solve_ridge(XtX, Xty)

            Fte = np.asarray(F_by_field[test_field], dtype=np.float64)
            Xte = (Fte - mu) / sd
            return (Xte @ w + y_mu).astype(np.float64, copy=False)

        # Results aggregated by alpha.
        agg_rows: list[dict[str, str]] = []
        per_alpha_sections: list[str] = []
        alpha_metrics: list[dict[str, Any]] = []

        for alpha in alpha_list:
            rho01_by_field: list[np.ndarray] = []
            rho0_by_field: list[np.ndarray] = []
            field_fft0_by_field: list[np.ndarray] = []
            field_fft_rho01_by_field: list[np.ndarray] = []
            split_cache: list[BandSplit2D] = []

            # Generate fields (deterministic per field_id; reuse seeds across alphas for comparability).
            for field_id in range(n_fields):
                rng_field = np.random.default_rng(seed + field_id)
                rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=float(alpha), rng=rng_field)
                split = band_split_poisson_2d(rho01, k0_frac=float(k0_frac))
                rho01_by_field.append(rho01.astype(np.float64, copy=False))
                rho0 = (rho01 - float(rho01.mean())).astype(np.float64, copy=False)
                rho0_by_field.append(rho0)
                field_fft0_by_field.append(np.fft.fftn(rho0))
                field_fft_rho01_by_field.append(np.fft.fftn(rho01))
                split_cache.append(split)

            y_low_by_field = [sample_vec_at_centers(split_cache[fid].low.gx, split_cache[fid].low.gy, centers_by_field[fid]) for fid in range(n_fields)]
            y_full_by_field = [
                sample_vec_at_centers(split_cache[fid].full.gx, split_cache[fid].full.gy, centers_by_field[fid]) for fid in range(n_fields)
            ]

            # Fixed kernel channels.
            y_low_ceiling_pred_by_field: list[np.ndarray] = []
            y_high_pred_by_field: list[np.ndarray] = []
            for fid in range(n_fields):
                field_fft0 = field_fft0_by_field[fid]
                gx_low_pred = np.fft.ifftn(field_fft0 * kfft_low_gx).real
                gy_low_pred = np.fft.ifftn(field_fft0 * kfft_low_gy).real
                y_low_ceiling_pred_by_field.append(sample_vec_at_centers(gx_low_pred, gy_low_pred, centers_by_field[fid]))

                gx_high_pred = np.fft.ifftn(field_fft0 * kfft_high_gx).real
                gy_high_pred = np.fft.ifftn(field_fft0 * kfft_high_gy).real
                y_high_pred_by_field.append(sample_vec_at_centers(gx_high_pred, gy_high_pred, centers_by_field[fid]))

            # Rect low-k features (FFT correlation with basis kernels).
            d_rect = 2 * n_bins_rect
            F_rect_by_field: list[np.ndarray] = [np.zeros((patches_per_field, d_rect), dtype=np.float32) for _ in range(n_fields)]
            for b in range(n_bins_rect):
                mask = (bin_idx == int(b)).astype(np.float64, copy=False)
                kdx = (dxr * mask).astype(np.float64, copy=False)
                kdy = (dyr * mask).astype(np.float64, copy=False)
                kfft_dx = kernel_fft_centered(kdx, grid_size=grid_size)
                kfft_dy = kernel_fft_centered(kdy, grid_size=grid_size)
                for fid in range(n_fields):
                    field_fft = field_fft_rho01_by_field[fid]
                    cx, cy = centers_idx[fid]
                    dx_grid = np.fft.ifftn(field_fft * kfft_dx).real
                    dy_grid = np.fft.ifftn(field_fft * kfft_dy).real
                    F_rect_by_field[fid][:, b] = dx_grid[cx, cy].astype(np.float32, copy=False)
                    F_rect_by_field[fid][:, n_bins_rect + b] = dy_grid[cx, cy].astype(np.float32, copy=False)

            # Rect-projection low-k predictions (deterministic).
            y_low_rect_proj_by_field: list[np.ndarray] = []
            for fid in range(n_fields):
                F = F_rect_by_field[fid].astype(np.float64, copy=False)
                y_pred = np.column_stack([F[:, :n_bins_rect] @ coef_rect_x, F[:, n_bins_rect:] @ coef_rect_y]).astype(np.float64, copy=False)
                y_low_rect_proj_by_field.append(y_pred)

            # LOFO evaluation.
            p_ceiling: list[float] = []
            r_ceiling: list[float] = []
            p_proj: list[float] = []
            r_proj: list[float] = []
            p_rect: list[float] = []
            r_rect: list[float] = []
            dP_rect: list[float] = []
            dR_rect: list[float] = []

            for test_field in range(n_fields):
                y_true = y_full_by_field[test_field]
                y_high = y_high_pred_by_field[test_field]

                y_pred_ceiling = y_low_ceiling_pred_by_field[test_field] + y_high
                y_pred_proj = y_low_rect_proj_by_field[test_field] + y_high
                y_low_rect = fit_predict_low_rect(F_rect_by_field, y_low_by_field, test_field=test_field)
                y_pred_rect = y_low_rect + y_high

                pc = pearson_mean_2d(y_true, y_pred_ceiling)
                rc = relrmse_mean_2d(y_true, y_pred_ceiling)
                pp = pearson_mean_2d(y_true, y_pred_proj)
                rp = relrmse_mean_2d(y_true, y_pred_proj)
                pr = pearson_mean_2d(y_true, y_pred_rect)
                rr = relrmse_mean_2d(y_true, y_pred_rect)

                p_ceiling.append(pc)
                r_ceiling.append(rc)
                p_proj.append(pp)
                r_proj.append(rp)
                p_rect.append(pr)
                r_rect.append(rr)
                dP_rect.append(pr - pc)
                dR_rect.append(rr - rc)

            pc_m = float(np.mean(p_ceiling))
            pc_s = float(np.std(p_ceiling, ddof=1))
            rc_m = float(np.mean(r_ceiling))
            rc_s = float(np.std(r_ceiling, ddof=1))
            pp_m = float(np.mean(p_proj))
            pp_s = float(np.std(p_proj, ddof=1))
            rp_m = float(np.mean(r_proj))
            rp_s = float(np.std(r_proj, ddof=1))
            pr_m = float(np.mean(p_rect))
            pr_s = float(np.std(p_rect, ddof=1))
            rr_m = float(np.mean(r_rect))
            rr_s = float(np.std(r_rect, ddof=1))
            dp_m = float(np.mean(dP_rect))
            dp_s = float(np.std(dP_rect, ddof=1))
            dr_m = float(np.mean(dR_rect))
            dr_s = float(np.std(dR_rect, ddof=1))

            row = {
                "alpha": f"{float(alpha):.1f}",
                "Pearson_ceiling": f"{pc_m:.4f} ± {pc_s:.4f}",
                "relRMSE_ceiling": f"{rc_m:.4f} ± {rc_s:.4f}",
                "Pearson_rect_proj": f"{pp_m:.4f} ± {pp_s:.4f}",
                "relRMSE_rect_proj": f"{rp_m:.4f} ± {rp_s:.4f}",
                "Pearson_rect_learn": f"{pr_m:.4f} ± {pr_s:.4f}",
                "relRMSE_rect_learn": f"{rr_m:.4f} ± {rr_s:.4f}",
                "ΔPearson(rect-ceiling)": f"{dp_m:+.4f} ± {dp_s:.4f}",
                "ΔrelRMSE(rect-ceiling)": f"{dr_m:+.4f} ± {dr_s:.4f}",
            }
            agg_rows.append(row)
            per_alpha_sections.append(
                f"## alpha={float(alpha):.1f}\n\n"
                + md_table(
                    [row],
                    [
                        "alpha",
                        "Pearson_ceiling",
                        "relRMSE_ceiling",
                        "Pearson_rect_proj",
                        "relRMSE_rect_proj",
                        "Pearson_rect_learn",
                        "relRMSE_rect_learn",
                        "ΔPearson(rect-ceiling)",
                        "ΔrelRMSE(rect-ceiling)",
                    ],
                )
                + "\n"
            )
            alpha_metrics.append(
                {
                    "alpha": float(alpha),
                    "ceiling": {"pearson_mean": pc_m, "pearson_std": pc_s, "relRMSE_mean": rc_m, "relRMSE_std": rc_s},
                    "rect_proj": {"pearson_mean": pp_m, "pearson_std": pp_s, "relRMSE_mean": rp_m, "relRMSE_std": rp_s},
                    "rect_learn": {"pearson_mean": pr_m, "pearson_std": pr_s, "relRMSE_mean": rr_m, "relRMSE_std": rr_s},
                    "delta_rect_vs_ceiling": {"dPearson_mean": dp_m, "dPearson_std": dp_s, "drelRMSE_mean": dr_m, "drelRMSE_std": dr_s},
                }
            )

        summary_md = (
            "# E41 — Alpha sweep (rect_learn vs ceiling)\n\n"
            f"- run: `{paths.run_dir}`\n"
            f"- grid_size={grid_size}, k0_frac={k0_frac}, n_fields={n_fields}, patches_per_field={patches_per_field}\n"
            f"- w_big={w_big}, w_local={w_local}\n"
            f"- rect: Nx={rect_nx},Ny={rect_ny}, ridge_alpha={ridge_alpha}\n"
            f"- alpha_list={alpha_list}\n\n"
            + "\n".join(per_alpha_sections)
            + "\n## Aggregated table\n\n"
            + md_table(
                agg_rows,
                [
                    "alpha",
                    "Pearson_ceiling",
                    "relRMSE_ceiling",
                    "Pearson_rect_proj",
                    "relRMSE_rect_proj",
                    "Pearson_rect_learn",
                    "relRMSE_rect_learn",
                    "ΔPearson(rect-ceiling)",
                    "ΔrelRMSE(rect-ceiling)",
                ],
            )
            + "\n"
        )

        (paths.run_dir / "summary_e41_alpha_sweep_rect_vs_ceiling.md").write_text(summary_md, encoding="utf-8")
        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "grid_size": grid_size,
                "k0_frac": k0_frac,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "w_big": int(w_big),
                "w_local": int(w_local),
                "rect": {"nx": int(rect_nx), "ny": int(rect_ny)},
                "ridge_alpha": ridge_alpha,
                "alpha_list": alpha_list,
                "alpha_metrics": alpha_metrics,
            },
        )
        return paths

    if experiment == "e42":
        # E42 — Conditional/Wiener linear predictor (finite window) vs ceiling + rect_learn (alpha sweep).
        #
        # We estimate the optimal linear predictor of y=(gx_low,gy_low) at the center from a finite rho window
        # of size w_big via the normal equations:
        #   (C_xx + λI) w = C_xy
        # where x is the flattened rho patch (train fields only), and we solve in pixel-space using conjugate
        # gradients with FFT-based covariance matvecs (Toeplitz via stationarity).
        #
        # We compare low-k prediction performance and weight similarity vs:
        # - ceiling: truncated impulse kernel (patch weights kcorr)
        # - rect_learn: learned rectangular dipole basis (E41-style ridge)
        grid_size = int(cfg.get("grid_size", 256))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 10_000))
        k0_frac = float(cfg.get("k0_frac", 0.15))
        w_big = _require_odd("w_big", int(cfg.get("w_big", 193)))

        alpha_list = [float(x) for x in cfg.get("alpha_list", [0.0, 1.0, 1.5, 2.0])]

        # Rect-learn (for comparison).
        rect_nx = int(cfg.get("rect_nx", 16))
        rect_ny = int(cfg.get("rect_ny", 8))
        ridge_alpha = float(cfg.get("ridge_alpha", 1.0))

        # Wiener solver params.
        lambda_rel = float(cfg.get("lambda_rel", 1e-6))
        cg_max_iter = int(cfg.get("cg_max_iter", 200))
        cg_tol = float(cfg.get("cg_tol", 1e-6))

        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if n_fields < 2:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        if not (0.0 < float(k0_frac) < 0.5):
            raise ValueError("k0_frac must be in (0,0.5)")
        if w_big > grid_size:
            raise ValueError("w_big must be <= grid_size")
        if not alpha_list:
            raise ValueError("alpha_list must be non-empty")
        if any(a < 0 for a in alpha_list):
            raise ValueError("alpha_list must be >= 0")
        if rect_nx <= 0 or rect_ny <= 0:
            raise ValueError("rect_nx/rect_ny must be > 0")
        if ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be > 0")
        if lambda_rel <= 0:
            raise ValueError("lambda_rel must be > 0")
        if cg_max_iter <= 0:
            raise ValueError("cg_max_iter must be > 0")
        if cg_tol <= 0:
            raise ValueError("cg_tol must be > 0")

        def safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            am = a - float(a.mean())
            bm = b - float(b.mean())
            denom = float(np.linalg.norm(am) * np.linalg.norm(bm)) + 1e-12
            return float((am @ bm) / denom)

        def relrmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
            e = y_pred - y_true
            rmse = float(np.sqrt(np.mean(e * e)))
            sd = float(np.std(y_true))
            return rmse / (sd + 1e-12)

        def pearson_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            px = safe_corr_1d(y_true[:, 0], y_pred[:, 0])
            py = safe_corr_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (px + py)

        def relrmse_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            rx = relrmse_1d(y_true[:, 0], y_pred[:, 0])
            ry = relrmse_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (rx + ry)

        def kernel_fft_centered(kernel: np.ndarray, *, grid_size: int) -> np.ndarray:
            kernel = np.asarray(kernel, dtype=np.float64)
            if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
                raise ValueError(f"kernel must be square 2D, got {kernel.shape}")
            w = int(kernel.shape[0])
            if (w % 2) == 0:
                raise ValueError(f"kernel size must be odd, got {w}")
            if w > grid_size:
                raise ValueError(f"kernel size {w} exceeds grid_size {grid_size}")
            r = w // 2
            kr = np.flip(kernel, axis=(0, 1))
            full = np.zeros((grid_size, grid_size), dtype=np.float64)
            c0 = grid_size // 2
            full[c0 - r : c0 + r + 1, c0 - r : c0 + r + 1] = kr
            full0 = np.fft.ifftshift(full)
            return np.fft.fftn(full0)

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r0 in rows:
                out.append("| " + " | ".join(r0.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        def candidate_edges(r: int) -> np.ndarray:
            base = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, r + 1]
            base = [int(x) for x in base if 0 <= int(x) <= r + 1]
            dense = list(range(0, min(16, r + 1) + 1))
            geom = np.unique(np.rint(np.geomspace(1, r + 1, num=64)).astype(int)).tolist()
            vals = sorted(set(base + dense + geom + [0, r + 1]))
            return np.asarray(vals, dtype=np.int64)

        def pick_edges(cand: np.ndarray, n_bins: int) -> np.ndarray:
            cand = np.asarray(cand, dtype=np.int64)
            if len(cand) < n_bins + 1:
                cand = np.arange(int(cand.min()), int(cand.max()) + 1, dtype=np.int64)
            idx = np.round(np.linspace(0, len(cand) - 1, n_bins + 1)).astype(int)
            idx[0] = 0
            idx[-1] = len(cand) - 1
            edges = np.unique(cand[idx])
            if len(edges) != n_bins + 1:
                edges = np.unique(np.rint(np.linspace(int(cand.min()), int(cand.max()), n_bins + 1)).astype(np.int64))
            edges[0] = int(cand.min())
            edges[-1] = int(cand.max())
            return edges

        def build_rect_bins(w: int, *, nx_bins: int, ny_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            w = _require_odd("w_big", int(w))
            r = w // 2
            coords = np.arange(w, dtype=np.float64) - float(r)
            dx, dy = np.meshgrid(coords, coords, indexing="ij")
            ax = np.abs(dx).astype(np.int64)
            ay = np.abs(dy).astype(np.int64)
            cand = candidate_edges(int(r))
            edges_x = pick_edges(cand, int(nx_bins))
            edges_y = pick_edges(cand, int(ny_bins))
            ix = np.digitize(ax, edges_x, right=False) - 1
            iy = np.digitize(ay, edges_y, right=False) - 1
            ix = np.clip(ix, 0, int(nx_bins) - 1).astype(np.int32, copy=False)
            iy = np.clip(iy, 0, int(ny_bins) - 1).astype(np.int32, copy=False)
            bin_idx = (ix * int(ny_bins) + iy).astype(np.int32, copy=False)
            return dx.astype(np.float64), dy.astype(np.float64), bin_idx

        def cg_solve(
            apply_A: Any,
            b: np.ndarray,
            *,
            max_iter: int,
            tol: float,
        ) -> tuple[np.ndarray, int, float]:
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            x = np.zeros_like(b)
            r = b - apply_A(x)
            p = r.copy()
            rs0 = float(r @ r)
            rs = rs0
            bnorm = float(np.sqrt(float(b @ b))) + 1e-12
            if bnorm == 0:
                return x, 0, 0.0
            for it in range(1, int(max_iter) + 1):
                Ap = apply_A(p)
                denom = float(p @ Ap) + 1e-18
                alpha = rs / denom
                x = x + alpha * p
                r = r - alpha * Ap
                rs_new = float(r @ r)
                if float(np.sqrt(rs_new)) <= float(tol) * bnorm:
                    return x, it, float(np.sqrt(rs_new)) / bnorm
                beta = rs_new / (rs + 1e-18)
                p = r + beta * p
                rs = rs_new
            return x, int(max_iter), float(np.sqrt(rs)) / bnorm

        # Shared patch centers (same across alphas).
        r_big = w_big // 2
        centers_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_cent = np.random.default_rng(seed + 555_555 + 10_000 * int(w_big) + 1_000 * field_id)
            cx = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
            cy = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
            centers_by_field.append(np.column_stack([cx, cy]).astype(np.int64, copy=False))

        def sample_vec_at_centers(gx: np.ndarray, gy: np.ndarray, centers: np.ndarray) -> np.ndarray:
            cx = centers[:, 0].astype(np.int64)
            cy = centers[:, 1].astype(np.int64)
            return np.column_stack([gx[cx, cy], gy[cx, cy]]).astype(np.float64, copy=False)

        # Fixed kernels (independent of alpha).
        rho_delta = np.zeros((grid_size, grid_size), dtype=np.float64)
        c0 = grid_size // 2
        rho_delta[c0, c0] = 1.0
        split_delta = band_split_poisson_2d(rho_delta, k0_frac=float(k0_frac))
        g_patch_low_gx = split_delta.low.gx[c0 - r_big : c0 + r_big + 1, c0 - r_big : c0 + r_big + 1]
        g_patch_low_gy = split_delta.low.gy[c0 - r_big : c0 + r_big + 1, c0 - r_big : c0 + r_big + 1]
        # Patch weights for ceiling (kcorr = flipped impulse).
        kcorr_low_gx = g_patch_low_gx[::-1, ::-1].astype(np.float64, copy=False)
        kcorr_low_gy = g_patch_low_gy[::-1, ::-1].astype(np.float64, copy=False)

        # Rect bins (independent of alpha) + projection coefficients for deterministic rect_proj baseline.
        dxr, dyr, bin_idx = build_rect_bins(w_big, nx_bins=int(rect_nx), ny_bins=int(rect_ny))
        n_bins_rect = int(rect_nx) * int(rect_ny)
        kfft_dx_by_bin: list[np.ndarray] = []
        kfft_dy_by_bin: list[np.ndarray] = []
        for b in range(n_bins_rect):
            mask = (bin_idx == int(b)).astype(np.float64, copy=False)
            kdx = (dxr * mask).astype(np.float64, copy=False)
            kdy = (dyr * mask).astype(np.float64, copy=False)
            kfft_dx_by_bin.append(kernel_fft_centered(kdx, grid_size=grid_size))
            kfft_dy_by_bin.append(kernel_fft_centered(kdy, grid_size=grid_size))
        bin_flat = bin_idx.reshape(-1).astype(np.int64, copy=False)
        dx_flat = dxr.reshape(-1).astype(np.float64, copy=False)
        dy_flat = dyr.reshape(-1).astype(np.float64, copy=False)
        kx_flat = kcorr_low_gx.reshape(-1).astype(np.float64, copy=False)
        ky_flat = kcorr_low_gy.reshape(-1).astype(np.float64, copy=False)
        # Use symmetry: gx depends primarily on Dx features; gy on Dy features (E39 style).
        coef_rect_dx_to_gx = np.where(
            (np.bincount(bin_flat, weights=dx_flat * dx_flat, minlength=n_bins_rect) > 0),
            np.bincount(bin_flat, weights=kx_flat * dx_flat, minlength=n_bins_rect)
            / (np.bincount(bin_flat, weights=dx_flat * dx_flat, minlength=n_bins_rect) + 1e-12),
            0.0,
        ).astype(np.float64, copy=False)
        coef_rect_dy_to_gy = np.where(
            (np.bincount(bin_flat, weights=dy_flat * dy_flat, minlength=n_bins_rect) > 0),
            np.bincount(bin_flat, weights=ky_flat * dy_flat, minlength=n_bins_rect)
            / (np.bincount(bin_flat, weights=dy_flat * dy_flat, minlength=n_bins_rect) + 1e-12),
            0.0,
        ).astype(np.float64, copy=False)

        # Fourier-domain transfer function H for gx_low,gy_low (independent of alpha/rho).
        nx = grid_size
        kx1 = 2.0 * np.pi * np.fft.fftfreq(nx).astype(np.float64)
        ky1 = 2.0 * np.pi * np.fft.fftfreq(nx).astype(np.float64)
        kx = kx1[:, None]
        ky = ky1[None, :]
        k2 = kx * kx + ky * ky
        k = np.sqrt(k2, dtype=np.float64)
        k_ny = np.pi
        k0 = float(k0_frac) * k_ny
        mask_low = k <= k0
        H_gx = np.zeros((nx, nx), dtype=np.complex128)
        H_gy = np.zeros((nx, nx), dtype=np.complex128)
        nonzero = k2 > 0
        kx2 = np.broadcast_to(kx, (nx, nx))
        ky2 = np.broadcast_to(ky, (nx, nx))
        H_gx[nonzero] = -(1j * kx2[nonzero] / k2[nonzero]) * mask_low[nonzero]
        H_gy[nonzero] = -(1j * ky2[nonzero] / k2[nonzero]) * mask_low[nonzero]
        H_gx[0, 0] = 0.0 + 0.0j
        H_gy[0, 0] = 0.0 + 0.0j
        conjH_gx = np.conj(H_gx)
        conjH_gy = np.conj(H_gy)

        # Precompute centered slice indices.
        sx = slice(c0 - r_big, c0 + r_big + 1)
        sy = slice(c0 - r_big, c0 + r_big + 1)

        # Tables to fill.
        perf_rows: list[dict[str, str]] = []
        sim_rows_gx: list[dict[str, str]] = []
        sim_rows_gy: list[dict[str, str]] = []
        alpha_metrics: list[dict[str, Any]] = []

        # Per alpha sweep.
        for alpha in alpha_list:
            # Generate fields (deterministic per field_id; reuse seeds across alphas).
            rho01_by_field: list[np.ndarray] = []
            rho0_by_field: list[np.ndarray] = []
            rho0_fft_by_field: list[np.ndarray] = []
            rho01_fft_by_field: list[np.ndarray] = []
            split_cache: list[BandSplit2D] = []
            y_low_by_field: list[np.ndarray] = []

            for field_id in range(n_fields):
                rng_field = np.random.default_rng(seed + field_id)
                rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=float(alpha), rng=rng_field)
                split = band_split_poisson_2d(rho01, k0_frac=float(k0_frac))
                rho01_by_field.append(rho01.astype(np.float64, copy=False))
                rho0 = (rho01 - float(rho01.mean())).astype(np.float64, copy=False)
                rho0_by_field.append(rho0)
                rho0_fft_by_field.append(np.fft.fftn(rho0))
                rho01_fft_by_field.append(np.fft.fftn(rho01))
                split_cache.append(split)
                y_low_by_field.append(sample_vec_at_centers(split.low.gx, split.low.gy, centers_by_field[field_id]))

            # Ceiling predictions (low-k) for each field.
            kfft_ceiling_gx = kernel_fft_centered(kcorr_low_gx, grid_size=grid_size)
            kfft_ceiling_gy = kernel_fft_centered(kcorr_low_gy, grid_size=grid_size)
            y_low_ceiling_pred_by_field: list[np.ndarray] = []
            for fid in range(n_fields):
                gx_pred = np.fft.ifftn(rho0_fft_by_field[fid] * kfft_ceiling_gx).real
                gy_pred = np.fft.ifftn(rho0_fft_by_field[fid] * kfft_ceiling_gy).real
                y_low_ceiling_pred_by_field.append(sample_vec_at_centers(gx_pred, gy_pred, centers_by_field[fid]))

            # Rect_proj predictions (deterministic) for each field (via rect features).
            y_low_rect_proj_pred_by_field: list[np.ndarray] = []
            F_rect_by_field: list[np.ndarray] = []
            for field_id in range(n_fields):
                rho_fft = rho01_fft_by_field[field_id]
                centers = centers_by_field[field_id]
                cx = centers[:, 0].astype(np.int64)
                cy = centers[:, 1].astype(np.int64)
                F = np.zeros((patches_per_field, 2 * n_bins_rect), dtype=np.float32)
                for b in range(n_bins_rect):
                    dx_grid = np.fft.ifftn(rho_fft * kfft_dx_by_bin[b]).real
                    dy_grid = np.fft.ifftn(rho_fft * kfft_dy_by_bin[b]).real
                    F[:, b] = dx_grid[cx, cy].astype(np.float32, copy=False)
                    F[:, n_bins_rect + b] = dy_grid[cx, cy].astype(np.float32, copy=False)
                F_rect_by_field.append(F)
                y_low_rect_proj_pred_by_field.append(
                    np.column_stack([F[:, :n_bins_rect] @ coef_rect_dx_to_gx, F[:, n_bins_rect:] @ coef_rect_dy_to_gy]).astype(np.float64, copy=False)
                )

            # Rect_learn: ridge fit in rect-feature space (fold-safe) + effective pixel weights.
            def rect_learn_fit_predict_and_weights(test_field: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                train_fields = [i for i in range(n_fields) if i != test_field]
                Xtr = np.concatenate([F_rect_by_field[fid] for fid in train_fields], axis=0).astype(np.float64, copy=False)
                ytr = np.concatenate([y_low_by_field[fid] for fid in train_fields], axis=0).astype(np.float64, copy=False)
                mu = Xtr.mean(axis=0)
                sd = Xtr.std(axis=0)
                sd = np.where(sd > 1e-12, sd, 1.0)
                y_mu = ytr.mean(axis=0)

                Xtrz = (Xtr - mu) / sd
                XtX = (Xtrz.T @ Xtrz) / float(Xtrz.shape[0])
                Xty = (Xtrz.T @ (ytr - y_mu)) / float(Xtrz.shape[0])
                d = int(XtX.shape[0])
                w = np.linalg.solve(XtX + ridge_alpha * np.eye(d, dtype=np.float64), Xty)

                Xte = (np.asarray(F_rect_by_field[test_field], dtype=np.float64) - mu) / sd
                y_pred = Xte @ w + y_mu

                # Effective pixel weights (patch weights) on the w_big window.
                beta = w / sd[:, None]
                beta_dx_to_gx = beta[:n_bins_rect, 0]
                beta_dy_to_gx = beta[n_bins_rect:, 0]
                beta_dx_to_gy = beta[:n_bins_rect, 1]
                beta_dy_to_gy = beta[n_bins_rect:, 1]
                bin_flat2 = bin_idx.astype(np.int64, copy=False)
                w_gx = dxr * beta_dx_to_gx[bin_flat2] + dyr * beta_dy_to_gx[bin_flat2]
                w_gy = dxr * beta_dx_to_gy[bin_flat2] + dyr * beta_dy_to_gy[bin_flat2]
                return y_pred.astype(np.float64, copy=False), w_gx.astype(np.float64, copy=False), w_gy.astype(np.float64, copy=False)

            # Wiener conditional predictor (pixel-space) per fold: solve (C_xx+λI)w=b.
            def wiener_fit_weights(train_fields: list[int]) -> tuple[np.ndarray, np.ndarray, float]:
                # Estimate power spectrum S_rr (normalized by N^2 for covariance).
                acc = np.zeros((grid_size, grid_size), dtype=np.float64)
                var_acc = 0.0
                for fid in train_fields:
                    R = rho0_fft_by_field[fid]
                    acc += (R * np.conj(R)).real
                    var_acc += float(np.mean(rho0_by_field[fid] * rho0_by_field[fid]))
                S_rr = (acc / float(len(train_fields))) / float(grid_size * grid_size)
                var0 = var_acc / float(len(train_fields))
                lam = float(lambda_rel) * float(var0)

                # b = C_xy via cross-spectrum S_r_y = S_rr * conj(H).
                r_rgx = np.fft.ifftn(S_rr * conjH_gx).real
                r_rgy = np.fft.ifftn(S_rr * conjH_gy).real
                r_rgx_c = np.fft.fftshift(r_rgx)
                r_rgy_c = np.fft.fftshift(r_rgy)
                b_gx = r_rgx_c[sx, sy].reshape(-1).astype(np.float64, copy=False)
                b_gy = r_rgy_c[sx, sy].reshape(-1).astype(np.float64, copy=False)

                # Matvec for (C_xx + λI) on the centered window.
                V = np.zeros((grid_size, grid_size), dtype=np.float64)

                def apply_A(v: np.ndarray, *, S: np.ndarray) -> np.ndarray:
                    v = np.asarray(v, dtype=np.float64).reshape(w_big, w_big)
                    V.fill(0.0)
                    V[sx, sy] = v
                    conv = np.fft.ifftn(np.fft.fftn(V) * S).real
                    out = conv[sx, sy].reshape(-1).astype(np.float64, copy=False)
                    return out + lam * v.reshape(-1)

                wx, itx, relresx = cg_solve(lambda v: apply_A(v, S=S_rr), b_gx, max_iter=cg_max_iter, tol=cg_tol)
                wy, ity, relresy = cg_solve(lambda v: apply_A(v, S=S_rr), b_gy, max_iter=cg_max_iter, tol=cg_tol)
                info = float(max(relresx, relresy))
                return wx.reshape(w_big, w_big), wy.reshape(w_big, w_big), info

            # LOFO eval for this alpha.
            p_ceiling_f: list[float] = []
            r_ceiling_f: list[float] = []
            p_rect_f: list[float] = []
            r_rect_f: list[float] = []
            p_wiener_f: list[float] = []
            r_wiener_f: list[float] = []
            cg_relres_f: list[float] = []

            # Weight similarity per fold (gx/gy separately).
            corr_wk_gx: list[float] = []
            rel_wk_gx: list[float] = []
            corr_wr_gx: list[float] = []
            rel_wr_gx: list[float] = []
            corr_wk_gy: list[float] = []
            rel_wk_gy: list[float] = []
            corr_wr_gy: list[float] = []
            rel_wr_gy: list[float] = []

            for test_field in range(n_fields):
                train_fields = [f for f in range(n_fields) if f != test_field]

                # Predictions (ceiling + rect_proj + rect_learn + wiener) on test field.
                y_true = y_low_by_field[test_field]

                y_pred_ceiling = y_low_ceiling_pred_by_field[test_field]
                p_ceiling_f.append(pearson_mean_2d(y_true, y_pred_ceiling))
                r_ceiling_f.append(relrmse_mean_2d(y_true, y_pred_ceiling))

                y_pred_rect, w_rect_gx, w_rect_gy = rect_learn_fit_predict_and_weights(test_field)
                p_rect_f.append(pearson_mean_2d(y_true, y_pred_rect))
                r_rect_f.append(relrmse_mean_2d(y_true, y_pred_rect))

                w_wiener_gx, w_wiener_gy, cg_info = wiener_fit_weights(train_fields)
                cg_relres_f.append(cg_info)

                # Wiener predictions via FFT correlation with patch weights.
                kfft_w_gx = kernel_fft_centered(w_wiener_gx, grid_size=grid_size)
                kfft_w_gy = kernel_fft_centered(w_wiener_gy, grid_size=grid_size)
                gx_pred = np.fft.ifftn(rho0_fft_by_field[test_field] * kfft_w_gx).real
                gy_pred = np.fft.ifftn(rho0_fft_by_field[test_field] * kfft_w_gy).real
                y_pred_wiener = sample_vec_at_centers(gx_pred, gy_pred, centers_by_field[test_field])
                p_wiener_f.append(pearson_mean_2d(y_true, y_pred_wiener))
                r_wiener_f.append(relrmse_mean_2d(y_true, y_pred_wiener))

                # Weight similarity (patch weights).
                def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
                    a = np.asarray(a, dtype=np.float64).reshape(-1)
                    b = np.asarray(b, dtype=np.float64).reshape(-1)
                    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))

                corr_wk_gx.append(safe_corr_1d(w_wiener_gx, kcorr_low_gx))
                rel_wk_gx.append(rel_l2(w_wiener_gx, kcorr_low_gx))
                corr_wr_gx.append(safe_corr_1d(w_wiener_gx, w_rect_gx))
                rel_wr_gx.append(rel_l2(w_wiener_gx, w_rect_gx))

                corr_wk_gy.append(safe_corr_1d(w_wiener_gy, kcorr_low_gy))
                rel_wk_gy.append(rel_l2(w_wiener_gy, kcorr_low_gy))
                corr_wr_gy.append(safe_corr_1d(w_wiener_gy, w_rect_gy))
                rel_wr_gy.append(rel_l2(w_wiener_gy, w_rect_gy))

            # Aggregate metrics for this alpha.
            pc_m = float(np.mean(p_ceiling_f))
            pc_s = float(np.std(p_ceiling_f, ddof=1))
            rc_m = float(np.mean(r_ceiling_f))
            rc_s = float(np.std(r_ceiling_f, ddof=1))
            pr_m = float(np.mean(p_rect_f))
            pr_s = float(np.std(p_rect_f, ddof=1))
            rr_m = float(np.mean(r_rect_f))
            rr_s = float(np.std(r_rect_f, ddof=1))
            pw_m = float(np.mean(p_wiener_f))
            pw_s = float(np.std(p_wiener_f, ddof=1))
            rw_m = float(np.mean(r_wiener_f))
            rw_s = float(np.std(r_wiener_f, ddof=1))

            dpw_m = pw_m - pc_m
            drw_m = rw_m - rc_m
            dpr_m = pr_m - pc_m
            drr_m = rr_m - rc_m

            perf_rows.append(
                {
                    "alpha": f"{float(alpha):.1f}",
                    "Pearson_ceiling": f"{pc_m:.4f} ± {pc_s:.4f}",
                    "relRMSE_ceiling": f"{rc_m:.4f} ± {rc_s:.4f}",
                    "Pearson_rect_learn": f"{pr_m:.4f} ± {pr_s:.4f}",
                    "relRMSE_rect_learn": f"{rr_m:.4f} ± {rr_s:.4f}",
                    "Pearson_wiener": f"{pw_m:.4f} ± {pw_s:.4f}",
                    "relRMSE_wiener": f"{rw_m:.4f} ± {rw_s:.4f}",
                    "ΔPearson(wiener-ceiling)": f"{dpw_m:+.4f}",
                    "ΔPearson(rect-ceiling)": f"{dpr_m:+.4f}",
                    "ΔrelRMSE(wiener-ceiling)": f"{drw_m:+.4f}",
                    "ΔrelRMSE(rect-ceiling)": f"{drr_m:+.4f}",
                }
            )

            sim_rows_gx.append(
                {
                    "alpha": f"{float(alpha):.1f}",
                    "corr(wiener,kernel)": f"{float(np.mean(corr_wk_gx)):.4f} ± {float(np.std(corr_wk_gx, ddof=1)):.4f}",
                    "relL2(wiener,kernel)": f"{float(np.mean(rel_wk_gx)):.4f} ± {float(np.std(rel_wk_gx, ddof=1)):.4f}",
                    "corr(wiener,rect)": f"{float(np.mean(corr_wr_gx)):.4f} ± {float(np.std(corr_wr_gx, ddof=1)):.4f}",
                    "relL2(wiener,rect)": f"{float(np.mean(rel_wr_gx)):.4f} ± {float(np.std(rel_wr_gx, ddof=1)):.4f}",
                }
            )
            sim_rows_gy.append(
                {
                    "alpha": f"{float(alpha):.1f}",
                    "corr(wiener,kernel)": f"{float(np.mean(corr_wk_gy)):.4f} ± {float(np.std(corr_wk_gy, ddof=1)):.4f}",
                    "relL2(wiener,kernel)": f"{float(np.mean(rel_wk_gy)):.4f} ± {float(np.std(rel_wk_gy, ddof=1)):.4f}",
                    "corr(wiener,rect)": f"{float(np.mean(corr_wr_gy)):.4f} ± {float(np.std(corr_wr_gy, ddof=1)):.4f}",
                    "relL2(wiener,rect)": f"{float(np.mean(rel_wr_gy)):.4f} ± {float(np.std(rel_wr_gy, ddof=1)):.4f}",
                }
            )

            alpha_metrics.append(
                {
                    "alpha": float(alpha),
                    "perf": {
                        "ceiling": {"pearson_mean": pc_m, "relRMSE_mean": rc_m},
                        "rect_learn": {"pearson_mean": pr_m, "relRMSE_mean": rr_m},
                        "wiener": {"pearson_mean": pw_m, "relRMSE_mean": rw_m},
                        "delta_wiener_vs_ceiling": {"dPearson_mean": dpw_m, "drelRMSE_mean": drw_m},
                    },
                    "cg_relres_mean": float(np.mean(cg_relres_f)),
                }
            )

        # Summary markdown.
        summary_md = (
            "# E42 — Conditional/Wiener predictor vs ceiling/rect (alpha sweep)\n\n"
            f"- run: `{paths.run_dir}`\n"
            f"- grid_size={grid_size}, k0_frac={k0_frac}, n_fields={n_fields}, patches_per_field={patches_per_field}\n"
            f"- w_big={w_big}\n"
            f"- rect: Nx={rect_nx},Ny={rect_ny}, ridge_alpha={ridge_alpha}\n"
            f"- wiener solver: lambda_rel={lambda_rel}, cg_max_iter={cg_max_iter}, cg_tol={cg_tol}\n"
            f"- alpha_list={alpha_list}\n\n"
            "## A) Performance (low-k target)\n\n"
            + md_table(
                perf_rows,
                [
                    "alpha",
                    "Pearson_ceiling",
                    "relRMSE_ceiling",
                    "Pearson_rect_learn",
                    "relRMSE_rect_learn",
                    "Pearson_wiener",
                    "relRMSE_wiener",
                    "ΔPearson(wiener-ceiling)",
                    "ΔPearson(rect-ceiling)",
                    "ΔrelRMSE(wiener-ceiling)",
                    "ΔrelRMSE(rect-ceiling)",
                ],
            )
            + "\n\n## B) Weight similarity (gx)\n\n"
            + md_table(sim_rows_gx, ["alpha", "corr(wiener,kernel)", "relL2(wiener,kernel)", "corr(wiener,rect)", "relL2(wiener,rect)"])
            + "\n\n## C) Weight similarity (gy)\n\n"
            + md_table(sim_rows_gy, ["alpha", "corr(wiener,kernel)", "relL2(wiener,kernel)", "corr(wiener,rect)", "relL2(wiener,rect)"])
            + "\n"
        )

        (paths.run_dir / "summary_e42_wiener_conditional_vs_kernel.md").write_text(summary_md, encoding="utf-8")
        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "grid_size": grid_size,
                "k0_frac": k0_frac,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "w_big": int(w_big),
                "rect": {"nx": int(rect_nx), "ny": int(rect_ny), "ridge_alpha": ridge_alpha},
                "wiener": {"lambda_rel": lambda_rel, "cg_max_iter": cg_max_iter, "cg_tol": cg_tol},
                "alpha_list": alpha_list,
                "alpha_metrics": alpha_metrics,
            },
        )
        return paths

    if experiment == "e43":
        # E43 — Project Wiener weights onto rectangular dipole basis (alpha=2.0, w_big=193).
        #
        # For each LOFO fold:
        #  - compute Wiener-optimal weights w_wiener_gx/gy (from train fields)
        #  - project w_wiener onto rect basis at several (Nx,Ny) via least squares
        #  - evaluate low-k prediction on test field
        #
        # Report weight similarity vs Wiener and performance vs Wiener/ceiling.
        grid_size = int(cfg.get("grid_size", 256))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 10_000))
        k0_frac = float(cfg.get("k0_frac", 0.15))
        w_big = _require_odd("w_big", int(cfg.get("w_big", 193)))
        alpha = float(cfg.get("alpha", 2.0))

        # Rect resolutions: list of [Nx,Ny] pairs.
        rect_pairs = cfg.get("rect_pairs", [[8, 4], [12, 6], [16, 8], [24, 12]])
        rect_pairs = [(int(p[0]), int(p[1])) for p in rect_pairs]

        # Wiener solver params.
        lambda_rel = float(cfg.get("lambda_rel", 1e-6))
        cg_max_iter = int(cfg.get("cg_max_iter", 200))
        cg_tol = float(cfg.get("cg_tol", 1e-6))

        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if n_fields < 2:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        if not (0.0 < float(k0_frac) < 0.5):
            raise ValueError("k0_frac must be in (0,0.5)")
        if w_big > grid_size:
            raise ValueError("w_big must be <= grid_size")
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        if not rect_pairs:
            raise ValueError("rect_pairs must be non-empty")
        if any(nx <= 0 or ny <= 0 for nx, ny in rect_pairs):
            raise ValueError("rect_pairs entries must be positive")
        if lambda_rel <= 0:
            raise ValueError("lambda_rel must be > 0")
        if cg_max_iter <= 0:
            raise ValueError("cg_max_iter must be > 0")
        if cg_tol <= 0:
            raise ValueError("cg_tol must be > 0")

        def safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            am = a - float(a.mean())
            bm = b - float(b.mean())
            denom = float(np.linalg.norm(am) * np.linalg.norm(bm)) + 1e-12
            return float((am @ bm) / denom)

        def relrmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
            e = y_pred - y_true
            rmse = float(np.sqrt(np.mean(e * e)))
            sd = float(np.std(y_true))
            return rmse / (sd + 1e-12)

        def pearson_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            px = safe_corr_1d(y_true[:, 0], y_pred[:, 0])
            py = safe_corr_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (px + py)

        def relrmse_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            rx = relrmse_1d(y_true[:, 0], y_pred[:, 0])
            ry = relrmse_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (rx + ry)

        def kernel_fft_centered(kernel: np.ndarray, *, grid_size: int) -> np.ndarray:
            kernel = np.asarray(kernel, dtype=np.float64)
            if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
                raise ValueError(f"kernel must be square 2D, got {kernel.shape}")
            w = int(kernel.shape[0])
            if (w % 2) == 0:
                raise ValueError(f"kernel size must be odd, got {w}")
            if w > grid_size:
                raise ValueError(f"kernel size {w} exceeds grid_size {grid_size}")
            r = w // 2
            kr = np.flip(kernel, axis=(0, 1))
            full = np.zeros((grid_size, grid_size), dtype=np.float64)
            c0 = grid_size // 2
            full[c0 - r : c0 + r + 1, c0 - r : c0 + r + 1] = kr
            full0 = np.fft.ifftshift(full)
            return np.fft.fftn(full0)

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r0 in rows:
                out.append("| " + " | ".join(r0.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        def candidate_edges(r: int) -> np.ndarray:
            base = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, r + 1]
            base = [int(x) for x in base if 0 <= int(x) <= r + 1]
            dense = list(range(0, min(16, r + 1) + 1))
            geom = np.unique(np.rint(np.geomspace(1, r + 1, num=64)).astype(int)).tolist()
            vals = sorted(set(base + dense + geom + [0, r + 1]))
            return np.asarray(vals, dtype=np.int64)

        def pick_edges(cand: np.ndarray, n_bins: int) -> np.ndarray:
            cand = np.asarray(cand, dtype=np.int64)
            if len(cand) < n_bins + 1:
                cand = np.arange(int(cand.min()), int(cand.max()) + 1, dtype=np.int64)
            idx = np.round(np.linspace(0, len(cand) - 1, n_bins + 1)).astype(int)
            idx[0] = 0
            idx[-1] = len(cand) - 1
            edges = np.unique(cand[idx])
            if len(edges) != n_bins + 1:
                edges = np.unique(np.rint(np.linspace(int(cand.min()), int(cand.max()), n_bins + 1)).astype(np.int64))
            edges[0] = int(cand.min())
            edges[-1] = int(cand.max())
            return edges

        def build_rect_bins(w: int, *, nx_bins: int, ny_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            w = _require_odd("w_big", int(w))
            r = w // 2
            coords = np.arange(w, dtype=np.float64) - float(r)
            dx, dy = np.meshgrid(coords, coords, indexing="ij")
            ax = np.abs(dx).astype(np.int64)
            ay = np.abs(dy).astype(np.int64)
            cand = candidate_edges(int(r))
            edges_x = pick_edges(cand, int(nx_bins))
            edges_y = pick_edges(cand, int(ny_bins))
            ix = np.digitize(ax, edges_x, right=False) - 1
            iy = np.digitize(ay, edges_y, right=False) - 1
            ix = np.clip(ix, 0, int(nx_bins) - 1).astype(np.int32, copy=False)
            iy = np.clip(iy, 0, int(ny_bins) - 1).astype(np.int32, copy=False)
            bin_idx = (ix * int(ny_bins) + iy).astype(np.int32, copy=False)
            return dx.astype(np.float64), dy.astype(np.float64), bin_idx

        def cg_solve(
            apply_A: Any,
            b: np.ndarray,
            *,
            max_iter: int,
            tol: float,
        ) -> tuple[np.ndarray, int, float]:
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            x = np.zeros_like(b)
            r = b - apply_A(x)
            p = r.copy()
            rs0 = float(r @ r)
            rs = rs0
            bnorm = float(np.sqrt(float(b @ b))) + 1e-12
            if bnorm == 0:
                return x, 0, 0.0
            for it in range(1, int(max_iter) + 1):
                Ap = apply_A(p)
                denom = float(p @ Ap) + 1e-18
                alpha = rs / denom
                x = x + alpha * p
                r = r - alpha * Ap
                rs_new = float(r @ r)
                if float(np.sqrt(rs_new)) <= float(tol) * bnorm:
                    return x, it, float(np.sqrt(rs_new)) / bnorm
                beta = rs_new / (rs + 1e-18)
                p = r + beta * p
                rs = rs_new
            return x, int(max_iter), float(np.sqrt(rs)) / bnorm

        # Patch centers (shared across fields).
        r_big = w_big // 2
        centers_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_cent = np.random.default_rng(seed + 555_555 + 10_000 * int(w_big) + 1_000 * field_id)
            cx = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
            cy = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
            centers_by_field.append(np.column_stack([cx, cy]).astype(np.int64, copy=False))

        def sample_vec_at_centers(gx: np.ndarray, gy: np.ndarray, centers: np.ndarray) -> np.ndarray:
            cx = centers[:, 0].astype(np.int64)
            cy = centers[:, 1].astype(np.int64)
            return np.column_stack([gx[cx, cy], gy[cx, cy]]).astype(np.float64, copy=False)

        # Generate fields (alpha fixed).
        rho01_by_field: list[np.ndarray] = []
        rho0_by_field: list[np.ndarray] = []
        rho0_fft_by_field: list[np.ndarray] = []
        split_cache: list[BandSplit2D] = []
        y_low_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=float(alpha), rng=rng_field)
            split = band_split_poisson_2d(rho01, k0_frac=float(k0_frac))
            rho01_by_field.append(rho01.astype(np.float64, copy=False))
            rho0 = (rho01 - float(rho01.mean())).astype(np.float64, copy=False)
            rho0_by_field.append(rho0)
            rho0_fft_by_field.append(np.fft.fftn(rho0))
            split_cache.append(split)
            y_low_by_field.append(sample_vec_at_centers(split.low.gx, split.low.gy, centers_by_field[field_id]))

        # Ceiling kernel (truncated impulse).
        rho_delta = np.zeros((grid_size, grid_size), dtype=np.float64)
        c0 = grid_size // 2
        rho_delta[c0, c0] = 1.0
        split_delta = band_split_poisson_2d(rho_delta, k0_frac=float(k0_frac))
        g_patch_low_gx = split_delta.low.gx[c0 - r_big : c0 + r_big + 1, c0 - r_big : c0 + r_big + 1]
        g_patch_low_gy = split_delta.low.gy[c0 - r_big : c0 + r_big + 1, c0 - r_big : c0 + r_big + 1]
        kcorr_low_gx = g_patch_low_gx[::-1, ::-1].astype(np.float64, copy=False)
        kcorr_low_gy = g_patch_low_gy[::-1, ::-1].astype(np.float64, copy=False)
        kfft_ceiling_gx = kernel_fft_centered(kcorr_low_gx, grid_size=grid_size)
        kfft_ceiling_gy = kernel_fft_centered(kcorr_low_gy, grid_size=grid_size)

        # Fourier-domain transfer H for low-k (independent of alpha/rho).
        nx = grid_size
        kx1 = 2.0 * np.pi * np.fft.fftfreq(nx).astype(np.float64)
        ky1 = 2.0 * np.pi * np.fft.fftfreq(nx).astype(np.float64)
        kx = kx1[:, None]
        ky = ky1[None, :]
        k2 = kx * kx + ky * ky
        k = np.sqrt(k2, dtype=np.float64)
        k_ny = np.pi
        k0 = float(k0_frac) * k_ny
        mask_low = k <= k0
        H_gx = np.zeros((nx, nx), dtype=np.complex128)
        H_gy = np.zeros((nx, nx), dtype=np.complex128)
        nonzero = k2 > 0
        kx2 = np.broadcast_to(kx, (nx, nx))
        ky2 = np.broadcast_to(ky, (nx, nx))
        H_gx[nonzero] = -(1j * kx2[nonzero] / k2[nonzero]) * mask_low[nonzero]
        H_gy[nonzero] = -(1j * ky2[nonzero] / k2[nonzero]) * mask_low[nonzero]
        H_gx[0, 0] = 0.0 + 0.0j
        H_gy[0, 0] = 0.0 + 0.0j
        conjH_gx = np.conj(H_gx)
        conjH_gy = np.conj(H_gy)

        # Window slices.
        sx = slice(c0 - r_big, c0 + r_big + 1)
        sy = slice(c0 - r_big, c0 + r_big + 1)

        # Wiener weights per fold (train fields only).
        def wiener_fit_weights(train_fields: list[int]) -> tuple[np.ndarray, np.ndarray, float]:
            acc = np.zeros((grid_size, grid_size), dtype=np.float64)
            var_acc = 0.0
            for fid in train_fields:
                R = rho0_fft_by_field[fid]
                acc += (R * np.conj(R)).real
                var_acc += float(np.mean(rho0_by_field[fid] * rho0_by_field[fid]))
            S_rr = (acc / float(len(train_fields))) / float(grid_size * grid_size)
            var0 = var_acc / float(len(train_fields))
            lam = float(lambda_rel) * float(var0)

            r_rgx = np.fft.ifftn(S_rr * conjH_gx).real
            r_rgy = np.fft.ifftn(S_rr * conjH_gy).real
            r_rgx_c = np.fft.fftshift(r_rgx)
            r_rgy_c = np.fft.fftshift(r_rgy)
            b_gx = r_rgx_c[sx, sy].reshape(-1).astype(np.float64, copy=False)
            b_gy = r_rgy_c[sx, sy].reshape(-1).astype(np.float64, copy=False)

            V = np.zeros((grid_size, grid_size), dtype=np.float64)

            def apply_A(v: np.ndarray, *, S: np.ndarray) -> np.ndarray:
                v = np.asarray(v, dtype=np.float64).reshape(w_big, w_big)
                V.fill(0.0)
                V[sx, sy] = v
                conv = np.fft.ifftn(np.fft.fftn(V) * S).real
                out = conv[sx, sy].reshape(-1).astype(np.float64, copy=False)
                return out + lam * v.reshape(-1)

            wx, itx, relresx = cg_solve(lambda v: apply_A(v, S=S_rr), b_gx, max_iter=cg_max_iter, tol=cg_tol)
            wy, ity, relresy = cg_solve(lambda v: apply_A(v, S=S_rr), b_gy, max_iter=cg_max_iter, tol=cg_tol)
            info = float(max(relresx, relresy))
            return wx.reshape(w_big, w_big), wy.reshape(w_big, w_big), info

        # Baseline performance arrays (per fold).
        p_ceiling_f: list[float] = []
        r_ceiling_f: list[float] = []
        p_wiener_f: list[float] = []
        r_wiener_f: list[float] = []
        cg_relres_f: list[float] = []

        # Per-resolution accumulators.
        perf_by_pair: dict[tuple[int, int], dict[str, list[float]]] = {}
        sim_by_pair: dict[tuple[int, int], dict[str, list[float]]] = {}
        for pair in rect_pairs:
            perf_by_pair[pair] = {"pearson": [], "relrmse": [], "dr_wiener": [], "dr_ceiling": []}
            sim_by_pair[pair] = {"corr": [], "relL2": []}

        # Precompute dx,dy and bin_idx per rect pair.
        rect_bins: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for pair in rect_pairs:
            rect_bins[pair] = build_rect_bins(w_big, nx_bins=pair[0], ny_bins=pair[1])

        def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))

        # LOFO.
        for test_field in range(n_fields):
            train_fields = [f for f in range(n_fields) if f != test_field]

            # Wiener weights from train fields.
            w_wiener_gx, w_wiener_gy, cg_info = wiener_fit_weights(train_fields)
            cg_relres_f.append(cg_info)

            # Wiener predictions on test field.
            kfft_w_gx = kernel_fft_centered(w_wiener_gx, grid_size=grid_size)
            kfft_w_gy = kernel_fft_centered(w_wiener_gy, grid_size=grid_size)
            gx_pred = np.fft.ifftn(rho0_fft_by_field[test_field] * kfft_w_gx).real
            gy_pred = np.fft.ifftn(rho0_fft_by_field[test_field] * kfft_w_gy).real
            y_pred_w = sample_vec_at_centers(gx_pred, gy_pred, centers_by_field[test_field])

            # Ceiling predictions.
            gx_c = np.fft.ifftn(rho0_fft_by_field[test_field] * kfft_ceiling_gx).real
            gy_c = np.fft.ifftn(rho0_fft_by_field[test_field] * kfft_ceiling_gy).real
            y_pred_c = sample_vec_at_centers(gx_c, gy_c, centers_by_field[test_field])

            y_true = y_low_by_field[test_field]

            p_ceiling_f.append(pearson_mean_2d(y_true, y_pred_c))
            r_ceiling_f.append(relrmse_mean_2d(y_true, y_pred_c))
            p_wiener_f.append(pearson_mean_2d(y_true, y_pred_w))
            r_wiener_f.append(relrmse_mean_2d(y_true, y_pred_w))

            # Projections for each rect pair.
            for pair in rect_pairs:
                dx, dy, bin_idx = rect_bins[pair]
                n_bins = pair[0] * pair[1]
                bin_flat = bin_idx.reshape(-1).astype(np.int64, copy=False)
                dx_flat = dx.reshape(-1).astype(np.float64, copy=False)
                dy_flat = dy.reshape(-1).astype(np.float64, copy=False)

                # Project w_wiener onto dx/dy bases independently.
                num_x = np.bincount(bin_flat, weights=w_wiener_gx.reshape(-1) * dx_flat, minlength=n_bins)
                den_x = np.bincount(bin_flat, weights=dx_flat * dx_flat, minlength=n_bins) + 1e-12
                coef_dx = (num_x / den_x).astype(np.float64, copy=False)

                num_y = np.bincount(bin_flat, weights=w_wiener_gy.reshape(-1) * dy_flat, minlength=n_bins)
                den_y = np.bincount(bin_flat, weights=dy_flat * dy_flat, minlength=n_bins) + 1e-12
                coef_dy = (num_y / den_y).astype(np.float64, copy=False)

                w_proj_gx = (dx * coef_dx[bin_idx]).astype(np.float64, copy=False)
                w_proj_gy = (dy * coef_dy[bin_idx]).astype(np.float64, copy=False)

                # Weight similarity (mean over gx/gy).
                corr_gx = safe_corr_1d(w_proj_gx, w_wiener_gx)
                corr_gy = safe_corr_1d(w_proj_gy, w_wiener_gy)
                rel_gx = rel_l2(w_proj_gx, w_wiener_gx)
                rel_gy = rel_l2(w_proj_gy, w_wiener_gy)
                sim_by_pair[pair]["corr"].append(0.5 * (corr_gx + corr_gy))
                sim_by_pair[pair]["relL2"].append(0.5 * (rel_gx + rel_gy))

                # Prediction on test field.
                kfft_p_gx = kernel_fft_centered(w_proj_gx, grid_size=grid_size)
                kfft_p_gy = kernel_fft_centered(w_proj_gy, grid_size=grid_size)
                gx_p = np.fft.ifftn(rho0_fft_by_field[test_field] * kfft_p_gx).real
                gy_p = np.fft.ifftn(rho0_fft_by_field[test_field] * kfft_p_gy).real
                y_pred_p = sample_vec_at_centers(gx_p, gy_p, centers_by_field[test_field])

                p_proj = pearson_mean_2d(y_true, y_pred_p)
                r_proj = relrmse_mean_2d(y_true, y_pred_p)

                perf_by_pair[pair]["pearson"].append(p_proj)
                perf_by_pair[pair]["relrmse"].append(r_proj)
                perf_by_pair[pair]["dr_wiener"].append(r_proj - r_wiener_f[-1])
                perf_by_pair[pair]["dr_ceiling"].append(r_proj - r_ceiling_f[-1])

        # Baseline stats.
        p_ceiling_mean = float(np.mean(p_ceiling_f))
        p_ceiling_std = float(np.std(p_ceiling_f, ddof=1))
        r_ceiling_mean = float(np.mean(r_ceiling_f))
        r_ceiling_std = float(np.std(r_ceiling_f, ddof=1))
        p_wiener_mean = float(np.mean(p_wiener_f))
        p_wiener_std = float(np.std(p_wiener_f, ddof=1))
        r_wiener_mean = float(np.mean(r_wiener_f))
        r_wiener_std = float(np.std(r_wiener_f, ddof=1))

        perf_rows: list[dict[str, str]] = []
        sim_rows: list[dict[str, str]] = []

        perf_rows.append(
            {
                "Nx Ny": "ceiling",
                "Pearson_proj mean±std": f"{p_ceiling_mean:.4f} ± {p_ceiling_std:.4f}",
                "relRMSE_proj mean±std": f"{r_ceiling_mean:.4f} ± {r_ceiling_std:.4f}",
                "ΔrelRMSE vs wiener": f"{(r_ceiling_mean - r_wiener_mean):+.4f}",
                "ΔrelRMSE vs ceiling": f"{0.0:+.4f}",
            }
        )
        perf_rows.append(
            {
                "Nx Ny": "wiener",
                "Pearson_proj mean±std": f"{p_wiener_mean:.4f} ± {p_wiener_std:.4f}",
                "relRMSE_proj mean±std": f"{r_wiener_mean:.4f} ± {r_wiener_std:.4f}",
                "ΔrelRMSE vs wiener": f"{0.0:+.4f}",
                "ΔrelRMSE vs ceiling": f"{(r_wiener_mean - r_ceiling_mean):+.4f}",
            }
        )

        for pair in rect_pairs:
            p_mean = float(np.mean(perf_by_pair[pair]["pearson"]))
            p_std = float(np.std(perf_by_pair[pair]["pearson"], ddof=1))
            r_mean = float(np.mean(perf_by_pair[pair]["relrmse"]))
            r_std = float(np.std(perf_by_pair[pair]["relrmse"], ddof=1))
            drw = float(np.mean(perf_by_pair[pair]["dr_wiener"]))
            drc = float(np.mean(perf_by_pair[pair]["dr_ceiling"]))
            perf_rows.append(
                {
                    "Nx Ny": f"{pair[0]}x{pair[1]}",
                    "Pearson_proj mean±std": f"{p_mean:.4f} ± {p_std:.4f}",
                    "relRMSE_proj mean±std": f"{r_mean:.4f} ± {r_std:.4f}",
                    "ΔrelRMSE vs wiener": f"{drw:+.4f}",
                    "ΔrelRMSE vs ceiling": f"{drc:+.4f}",
                }
            )

            corr_mean = float(np.mean(sim_by_pair[pair]["corr"]))
            corr_std = float(np.std(sim_by_pair[pair]["corr"], ddof=1))
            rel_mean = float(np.mean(sim_by_pair[pair]["relL2"]))
            rel_std = float(np.std(sim_by_pair[pair]["relL2"], ddof=1))
            sim_rows.append(
                {
                    "Nx Ny": f"{pair[0]}x{pair[1]}",
                    "corr_proj_wiener": f"{corr_mean:.4f} ± {corr_std:.4f}",
                    "relL2_proj_wiener": f"{rel_mean:.4f} ± {rel_std:.4f}",
                }
            )

        summary_md = (
            "# E43 — Rect projection onto Wiener weights (alpha=2.0, low-k)\n\n"
            f"- run: `{paths.run_dir}`\n"
            f"- grid_size={grid_size}, k0_frac={k0_frac}, n_fields={n_fields}, patches_per_field={patches_per_field}\n"
            f"- w_big={w_big}, alpha={alpha}\n"
            f"- rect_pairs={rect_pairs}\n"
            f"- wiener solver: lambda_rel={lambda_rel}, cg_max_iter={cg_max_iter}, cg_tol={cg_tol}\n\n"
            "## Performance (low-k)\n\n"
            + md_table(perf_rows, ["Nx Ny", "Pearson_proj mean±std", "relRMSE_proj mean±std", "ΔrelRMSE vs wiener", "ΔrelRMSE vs ceiling"])
            + "\n\n## Weight similarity vs Wiener (mean of gx/gy)\n\n"
            + md_table(sim_rows, ["Nx Ny", "corr_proj_wiener", "relL2_proj_wiener"])
            + "\n"
        )

        (paths.run_dir / "summary_e43_rectproj_to_wiener_alpha2.md").write_text(summary_md, encoding="utf-8")
        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "grid_size": grid_size,
                "k0_frac": k0_frac,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "w_big": int(w_big),
                "alpha": float(alpha),
                "rect_pairs": rect_pairs,
                "wiener": {"lambda_rel": lambda_rel, "cg_max_iter": cg_max_iter, "cg_tol": cg_tol},
            },
        )
        return paths

    if experiment == "e44":
        # E44 — Project Wiener weights onto a 2D Fourier parity basis (alpha=2.0, w_big=193).
        #
        # Compare Fourier-basis projection vs rect-bins and baselines (Wiener, ceiling) on low-k target.
        grid_size = int(cfg.get("grid_size", 256))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 10_000))
        k0_frac = float(cfg.get("k0_frac", 0.15))
        w_big = _require_odd("w_big", int(cfg.get("w_big", 193)))
        alpha = float(cfg.get("alpha", 2.0))

        # Fourier basis sizes (P,Q).
        pq_list = cfg.get("pq_list", [[8, 4], [12, 6], [16, 8], [24, 12]])
        pq_list = [(int(p[0]), int(p[1])) for p in pq_list]

        # Optional rect baseline (one setting).
        rect_pair = cfg.get("rect_pair", [24, 12])
        rect_pair = (int(rect_pair[0]), int(rect_pair[1]))

        # Projection ridge for basis weights.
        proj_lambda_rel = float(cfg.get("proj_lambda_rel", 1e-8))

        # Wiener solver params.
        lambda_rel = float(cfg.get("lambda_rel", 1e-6))
        cg_max_iter = int(cfg.get("cg_max_iter", 200))
        cg_tol = float(cfg.get("cg_tol", 1e-6))

        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if n_fields < 2:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        if not (0.0 < float(k0_frac) < 0.5):
            raise ValueError("k0_frac must be in (0,0.5)")
        if w_big > grid_size:
            raise ValueError("w_big must be <= grid_size")
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        if not pq_list:
            raise ValueError("pq_list must be non-empty")
        if any(p <= 0 or q <= 0 for p, q in pq_list):
            raise ValueError("pq_list entries must be positive")
        if rect_pair[0] <= 0 or rect_pair[1] <= 0:
            raise ValueError("rect_pair must be positive")
        if proj_lambda_rel <= 0:
            raise ValueError("proj_lambda_rel must be > 0")
        if lambda_rel <= 0:
            raise ValueError("lambda_rel must be > 0")
        if cg_max_iter <= 0:
            raise ValueError("cg_max_iter must be > 0")
        if cg_tol <= 0:
            raise ValueError("cg_tol must be > 0")

        def safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            am = a - float(a.mean())
            bm = b - float(b.mean())
            denom = float(np.linalg.norm(am) * np.linalg.norm(bm)) + 1e-12
            return float((am @ bm) / denom)

        def relrmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
            e = y_pred - y_true
            rmse = float(np.sqrt(np.mean(e * e)))
            sd = float(np.std(y_true))
            return rmse / (sd + 1e-12)

        def pearson_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            px = safe_corr_1d(y_true[:, 0], y_pred[:, 0])
            py = safe_corr_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (px + py)

        def relrmse_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            rx = relrmse_1d(y_true[:, 0], y_pred[:, 0])
            ry = relrmse_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (rx + ry)

        def kernel_fft_centered(kernel: np.ndarray, *, grid_size: int) -> np.ndarray:
            kernel = np.asarray(kernel, dtype=np.float64)
            if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
                raise ValueError(f"kernel must be square 2D, got {kernel.shape}")
            w = int(kernel.shape[0])
            if (w % 2) == 0:
                raise ValueError(f"kernel size must be odd, got {w}")
            if w > grid_size:
                raise ValueError(f"kernel size {w} exceeds grid_size {grid_size}")
            r = w // 2
            kr = np.flip(kernel, axis=(0, 1))
            full = np.zeros((grid_size, grid_size), dtype=np.float64)
            c0 = grid_size // 2
            full[c0 - r : c0 + r + 1, c0 - r : c0 + r + 1] = kr
            full0 = np.fft.ifftshift(full)
            return np.fft.fftn(full0)

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r0 in rows:
                out.append("| " + " | ".join(r0.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        def candidate_edges(r: int) -> np.ndarray:
            base = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, r + 1]
            base = [int(x) for x in base if 0 <= int(x) <= r + 1]
            dense = list(range(0, min(16, r + 1) + 1))
            geom = np.unique(np.rint(np.geomspace(1, r + 1, num=64)).astype(int)).tolist()
            vals = sorted(set(base + dense + geom + [0, r + 1]))
            return np.asarray(vals, dtype=np.int64)

        def pick_edges(cand: np.ndarray, n_bins: int) -> np.ndarray:
            cand = np.asarray(cand, dtype=np.int64)
            if len(cand) < n_bins + 1:
                cand = np.arange(int(cand.min()), int(cand.max()) + 1, dtype=np.int64)
            idx = np.round(np.linspace(0, len(cand) - 1, n_bins + 1)).astype(int)
            idx[0] = 0
            idx[-1] = len(cand) - 1
            edges = np.unique(cand[idx])
            if len(edges) != n_bins + 1:
                edges = np.unique(np.rint(np.linspace(int(cand.min()), int(cand.max()), n_bins + 1)).astype(np.int64))
            edges[0] = int(cand.min())
            edges[-1] = int(cand.max())
            return edges

        def build_rect_bins(w: int, *, nx_bins: int, ny_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            w = _require_odd("w_big", int(w))
            r = w // 2
            coords = np.arange(w, dtype=np.float64) - float(r)
            dx, dy = np.meshgrid(coords, coords, indexing="ij")
            ax = np.abs(dx).astype(np.int64)
            ay = np.abs(dy).astype(np.int64)
            cand = candidate_edges(int(r))
            edges_x = pick_edges(cand, int(nx_bins))
            edges_y = pick_edges(cand, int(ny_bins))
            ix = np.digitize(ax, edges_x, right=False) - 1
            iy = np.digitize(ay, edges_y, right=False) - 1
            ix = np.clip(ix, 0, int(nx_bins) - 1).astype(np.int32, copy=False)
            iy = np.clip(iy, 0, int(ny_bins) - 1).astype(np.int32, copy=False)
            bin_idx = (ix * int(ny_bins) + iy).astype(np.int32, copy=False)
            return dx.astype(np.float64), dy.astype(np.float64), bin_idx

        def cg_solve(
            apply_A: Any,
            b: np.ndarray,
            *,
            max_iter: int,
            tol: float,
        ) -> tuple[np.ndarray, int, float]:
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            x = np.zeros_like(b)
            r = b - apply_A(x)
            p = r.copy()
            rs0 = float(r @ r)
            rs = rs0
            bnorm = float(np.sqrt(float(b @ b))) + 1e-12
            if bnorm == 0:
                return x, 0, 0.0
            for it in range(1, int(max_iter) + 1):
                Ap = apply_A(p)
                denom = float(p @ Ap) + 1e-18
                alpha = rs / denom
                x = x + alpha * p
                r = r - alpha * Ap
                rs_new = float(r @ r)
                if float(np.sqrt(rs_new)) <= float(tol) * bnorm:
                    return x, it, float(np.sqrt(rs_new)) / bnorm
                beta = rs_new / (rs + 1e-18)
                p = r + beta * p
                rs = rs_new
            return x, int(max_iter), float(np.sqrt(rs)) / bnorm

        # Patch centers (shared across fields).
        r_big = w_big // 2
        centers_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_cent = np.random.default_rng(seed + 555_555 + 10_000 * int(w_big) + 1_000 * field_id)
            cx = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
            cy = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
            centers_by_field.append(np.column_stack([cx, cy]).astype(np.int64, copy=False))

        def sample_vec_at_centers(gx: np.ndarray, gy: np.ndarray, centers: np.ndarray) -> np.ndarray:
            cx = centers[:, 0].astype(np.int64)
            cy = centers[:, 1].astype(np.int64)
            return np.column_stack([gx[cx, cy], gy[cx, cy]]).astype(np.float64, copy=False)

        # Generate fields (alpha fixed).
        rho01_by_field: list[np.ndarray] = []
        rho0_by_field: list[np.ndarray] = []
        rho0_fft_by_field: list[np.ndarray] = []
        split_cache: list[BandSplit2D] = []
        y_low_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=float(alpha), rng=rng_field)
            split = band_split_poisson_2d(rho01, k0_frac=float(k0_frac))
            rho01_by_field.append(rho01.astype(np.float64, copy=False))
            rho0 = (rho01 - float(rho01.mean())).astype(np.float64, copy=False)
            rho0_by_field.append(rho0)
            rho0_fft_by_field.append(np.fft.fftn(rho0))
            split_cache.append(split)
            y_low_by_field.append(sample_vec_at_centers(split.low.gx, split.low.gy, centers_by_field[field_id]))

        # Ceiling kernel (truncated impulse).
        rho_delta = np.zeros((grid_size, grid_size), dtype=np.float64)
        c0 = grid_size // 2
        rho_delta[c0, c0] = 1.0
        split_delta = band_split_poisson_2d(rho_delta, k0_frac=float(k0_frac))
        g_patch_low_gx = split_delta.low.gx[c0 - r_big : c0 + r_big + 1, c0 - r_big : c0 + r_big + 1]
        g_patch_low_gy = split_delta.low.gy[c0 - r_big : c0 + r_big + 1, c0 - r_big : c0 + r_big + 1]
        kcorr_low_gx = g_patch_low_gx[::-1, ::-1].astype(np.float64, copy=False)
        kcorr_low_gy = g_patch_low_gy[::-1, ::-1].astype(np.float64, copy=False)
        kfft_ceiling_gx = kernel_fft_centered(kcorr_low_gx, grid_size=grid_size)
        kfft_ceiling_gy = kernel_fft_centered(kcorr_low_gy, grid_size=grid_size)

        # Fourier-domain transfer H for low-k (independent of alpha/rho).
        nx = grid_size
        kx1 = 2.0 * np.pi * np.fft.fftfreq(nx).astype(np.float64)
        ky1 = 2.0 * np.pi * np.fft.fftfreq(nx).astype(np.float64)
        kx = kx1[:, None]
        ky = ky1[None, :]
        k2 = kx * kx + ky * ky
        k = np.sqrt(k2, dtype=np.float64)
        k_ny = np.pi
        k0 = float(k0_frac) * k_ny
        mask_low = k <= k0
        H_gx = np.zeros((nx, nx), dtype=np.complex128)
        H_gy = np.zeros((nx, nx), dtype=np.complex128)
        nonzero = k2 > 0
        kx2 = np.broadcast_to(kx, (nx, nx))
        ky2 = np.broadcast_to(ky, (nx, nx))
        H_gx[nonzero] = -(1j * kx2[nonzero] / k2[nonzero]) * mask_low[nonzero]
        H_gy[nonzero] = -(1j * ky2[nonzero] / k2[nonzero]) * mask_low[nonzero]
        H_gx[0, 0] = 0.0 + 0.0j
        H_gy[0, 0] = 0.0 + 0.0j
        conjH_gx = np.conj(H_gx)
        conjH_gy = np.conj(H_gy)

        # Window slices.
        sx = slice(c0 - r_big, c0 + r_big + 1)
        sy = slice(c0 - r_big, c0 + r_big + 1)

        # Wiener weights per fold (train fields only).
        def wiener_fit_weights(train_fields: list[int]) -> tuple[np.ndarray, np.ndarray, float]:
            acc = np.zeros((grid_size, grid_size), dtype=np.float64)
            var_acc = 0.0
            for fid in train_fields:
                R = rho0_fft_by_field[fid]
                acc += (R * np.conj(R)).real
                var_acc += float(np.mean(rho0_by_field[fid] * rho0_by_field[fid]))
            S_rr = (acc / float(len(train_fields))) / float(grid_size * grid_size)
            var0 = var_acc / float(len(train_fields))
            lam = float(lambda_rel) * float(var0)

            r_rgx = np.fft.ifftn(S_rr * conjH_gx).real
            r_rgy = np.fft.ifftn(S_rr * conjH_gy).real
            r_rgx_c = np.fft.fftshift(r_rgx)
            r_rgy_c = np.fft.fftshift(r_rgy)
            b_gx = r_rgx_c[sx, sy].reshape(-1).astype(np.float64, copy=False)
            b_gy = r_rgy_c[sx, sy].reshape(-1).astype(np.float64, copy=False)

            V = np.zeros((grid_size, grid_size), dtype=np.float64)

            def apply_A(v: np.ndarray, *, S: np.ndarray) -> np.ndarray:
                v = np.asarray(v, dtype=np.float64).reshape(w_big, w_big)
                V.fill(0.0)
                V[sx, sy] = v
                conv = np.fft.ifftn(np.fft.fftn(V) * S).real
                out = conv[sx, sy].reshape(-1).astype(np.float64, copy=False)
                return out + lam * v.reshape(-1)

            wx, itx, relresx = cg_solve(lambda v: apply_A(v, S=S_rr), b_gx, max_iter=cg_max_iter, tol=cg_tol)
            wy, ity, relresy = cg_solve(lambda v: apply_A(v, S=S_rr), b_gy, max_iter=cg_max_iter, tol=cg_tol)
            info = float(max(relresx, relresy))
            return wx.reshape(w_big, w_big), wy.reshape(w_big, w_big), info

        # Rect bins for optional baseline.
        rect_dx, rect_dy, rect_bin_idx = build_rect_bins(w_big, nx_bins=rect_pair[0], ny_bins=rect_pair[1])
        rect_n_bins = rect_pair[0] * rect_pair[1]

        # Fourier basis caches per (P,Q).
        basis_by_pq: dict[tuple[int, int], dict[str, Any]] = {}
        coords = (np.arange(w_big, dtype=np.float64) - float(r_big)).astype(np.float64)
        dx_grid, dy_grid = np.meshgrid(coords, coords, indexing="ij")
        L = float(r_big + 1)
        for P, Q in pq_list:
            m = int(P * Q)
            Bx = np.zeros((w_big * w_big, m), dtype=np.float64)
            By = np.zeros((w_big * w_big, m), dtype=np.float64)
            col = 0
            for p in range(1, P + 1):
                sinx = np.sin(np.pi * float(p) * dx_grid / L)
                siny = np.sin(np.pi * float(p) * dy_grid / L)
                for q in range(0, Q):
                    cosy = np.cos(np.pi * float(q) * dy_grid / L)
                    cosx = np.cos(np.pi * float(q) * dx_grid / L)
                    Bx[:, col] = (sinx * cosy).reshape(-1)
                    By[:, col] = (cosx * siny).reshape(-1)
                    col += 1
            BtB_x = Bx.T @ Bx
            BtB_y = By.T @ By
            lam_x = float(proj_lambda_rel) * (float(np.trace(BtB_x)) / float(m))
            lam_y = float(proj_lambda_rel) * (float(np.trace(BtB_y)) / float(m))
            basis_by_pq[(P, Q)] = {
                "Bx": Bx,
                "By": By,
                "BtB_x": BtB_x,
                "BtB_y": BtB_y,
                "lam_x": lam_x,
                "lam_y": lam_y,
            }

        def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))

        # Baseline performance arrays (per fold).
        p_ceiling_f: list[float] = []
        r_ceiling_f: list[float] = []
        p_wiener_f: list[float] = []
        r_wiener_f: list[float] = []

        # Per-basis accumulators.
        perf_by_pq: dict[tuple[int, int], dict[str, list[float]]] = {}
        sim_by_pq: dict[tuple[int, int], dict[str, list[float]]] = {}
        for pair in pq_list:
            perf_by_pq[pair] = {"pearson": [], "relrmse": [], "dr_wiener": [], "dr_ceiling": []}
            sim_by_pq[pair] = {"corr": [], "relL2": []}

        # Rect baseline accumulators.
        rect_perf: dict[str, list[float]] = {"pearson": [], "relrmse": [], "dr_wiener": [], "dr_ceiling": []}

        # LOFO.
        for test_field in range(n_fields):
            train_fields = [f for f in range(n_fields) if f != test_field]

            # Wiener weights from train fields.
            w_wiener_gx, w_wiener_gy, cg_info = wiener_fit_weights(train_fields)

            # Wiener predictions on test field.
            kfft_w_gx = kernel_fft_centered(w_wiener_gx, grid_size=grid_size)
            kfft_w_gy = kernel_fft_centered(w_wiener_gy, grid_size=grid_size)
            gx_pred = np.fft.ifftn(rho0_fft_by_field[test_field] * kfft_w_gx).real
            gy_pred = np.fft.ifftn(rho0_fft_by_field[test_field] * kfft_w_gy).real
            y_pred_w = sample_vec_at_centers(gx_pred, gy_pred, centers_by_field[test_field])

            # Ceiling predictions.
            gx_c = np.fft.ifftn(rho0_fft_by_field[test_field] * kfft_ceiling_gx).real
            gy_c = np.fft.ifftn(rho0_fft_by_field[test_field] * kfft_ceiling_gy).real
            y_pred_c = sample_vec_at_centers(gx_c, gy_c, centers_by_field[test_field])

            y_true = y_low_by_field[test_field]

            p_ceiling_f.append(pearson_mean_2d(y_true, y_pred_c))
            r_ceiling_f.append(relrmse_mean_2d(y_true, y_pred_c))
            p_wiener_f.append(pearson_mean_2d(y_true, y_pred_w))
            r_wiener_f.append(relrmse_mean_2d(y_true, y_pred_w))

            # Rect baseline projection (dx/dy bins).
            rect_bin_flat = rect_bin_idx.reshape(-1).astype(np.int64, copy=False)
            rect_dx_flat = rect_dx.reshape(-1).astype(np.float64, copy=False)
            rect_dy_flat = rect_dy.reshape(-1).astype(np.float64, copy=False)
            num_x = np.bincount(rect_bin_flat, weights=w_wiener_gx.reshape(-1) * rect_dx_flat, minlength=rect_n_bins)
            den_x = np.bincount(rect_bin_flat, weights=rect_dx_flat * rect_dx_flat, minlength=rect_n_bins) + 1e-12
            coef_dx = (num_x / den_x).astype(np.float64, copy=False)
            num_y = np.bincount(rect_bin_flat, weights=w_wiener_gy.reshape(-1) * rect_dy_flat, minlength=rect_n_bins)
            den_y = np.bincount(rect_bin_flat, weights=rect_dy_flat * rect_dy_flat, minlength=rect_n_bins) + 1e-12
            coef_dy = (num_y / den_y).astype(np.float64, copy=False)
            w_rect_gx = (rect_dx * coef_dx[rect_bin_idx]).astype(np.float64, copy=False)
            w_rect_gy = (rect_dy * coef_dy[rect_bin_idx]).astype(np.float64, copy=False)
            kfft_r_gx = kernel_fft_centered(w_rect_gx, grid_size=grid_size)
            kfft_r_gy = kernel_fft_centered(w_rect_gy, grid_size=grid_size)
            gx_r = np.fft.ifftn(rho0_fft_by_field[test_field] * kfft_r_gx).real
            gy_r = np.fft.ifftn(rho0_fft_by_field[test_field] * kfft_r_gy).real
            y_pred_r = sample_vec_at_centers(gx_r, gy_r, centers_by_field[test_field])
            p_r = pearson_mean_2d(y_true, y_pred_r)
            r_r = relrmse_mean_2d(y_true, y_pred_r)
            rect_perf["pearson"].append(p_r)
            rect_perf["relrmse"].append(r_r)
            rect_perf["dr_wiener"].append(r_r - r_wiener_f[-1])
            rect_perf["dr_ceiling"].append(r_r - r_ceiling_f[-1])

            # Fourier projection per (P,Q).
            for (P, Q), basis in basis_by_pq.items():
                Bx = basis["Bx"]
                By = basis["By"]
                BtB_x = basis["BtB_x"]
                BtB_y = basis["BtB_y"]
                lam_x = basis["lam_x"]
                lam_y = basis["lam_y"]
                BtW_x = Bx.T @ w_wiener_gx.reshape(-1)
                BtW_y = By.T @ w_wiener_gy.reshape(-1)
                a_x = np.linalg.solve(BtB_x + lam_x * np.eye(BtB_x.shape[0], dtype=np.float64), BtW_x)
                a_y = np.linalg.solve(BtB_y + lam_y * np.eye(BtB_y.shape[0], dtype=np.float64), BtW_y)
                w_proj_gx = (Bx @ a_x).reshape(w_big, w_big).astype(np.float64, copy=False)
                w_proj_gy = (By @ a_y).reshape(w_big, w_big).astype(np.float64, copy=False)

                # Weight similarity (mean over gx/gy).
                corr_gx = safe_corr_1d(w_proj_gx, w_wiener_gx)
                corr_gy = safe_corr_1d(w_proj_gy, w_wiener_gy)
                rel_gx = rel_l2(w_proj_gx, w_wiener_gx)
                rel_gy = rel_l2(w_proj_gy, w_wiener_gy)
                sim_by_pq[(P, Q)]["corr"].append(0.5 * (corr_gx + corr_gy))
                sim_by_pq[(P, Q)]["relL2"].append(0.5 * (rel_gx + rel_gy))

                kfft_p_gx = kernel_fft_centered(w_proj_gx, grid_size=grid_size)
                kfft_p_gy = kernel_fft_centered(w_proj_gy, grid_size=grid_size)
                gx_p = np.fft.ifftn(rho0_fft_by_field[test_field] * kfft_p_gx).real
                gy_p = np.fft.ifftn(rho0_fft_by_field[test_field] * kfft_p_gy).real
                y_pred_p = sample_vec_at_centers(gx_p, gy_p, centers_by_field[test_field])
                p_proj = pearson_mean_2d(y_true, y_pred_p)
                r_proj = relrmse_mean_2d(y_true, y_pred_p)
                perf_by_pq[(P, Q)]["pearson"].append(p_proj)
                perf_by_pq[(P, Q)]["relrmse"].append(r_proj)
                perf_by_pq[(P, Q)]["dr_wiener"].append(r_proj - r_wiener_f[-1])
                perf_by_pq[(P, Q)]["dr_ceiling"].append(r_proj - r_ceiling_f[-1])

        # Baseline stats.
        p_ceiling_mean = float(np.mean(p_ceiling_f))
        p_ceiling_std = float(np.std(p_ceiling_f, ddof=1))
        r_ceiling_mean = float(np.mean(r_ceiling_f))
        r_ceiling_std = float(np.std(r_ceiling_f, ddof=1))
        p_wiener_mean = float(np.mean(p_wiener_f))
        p_wiener_std = float(np.std(p_wiener_f, ddof=1))
        r_wiener_mean = float(np.mean(r_wiener_f))
        r_wiener_std = float(np.std(r_wiener_f, ddof=1))

        perf_rows: list[dict[str, str]] = []
        sim_rows: list[dict[str, str]] = []

        perf_rows.append(
            {
                "P×Q": "ceiling",
                "Pearson_proj mean±std": f"{p_ceiling_mean:.4f} ± {p_ceiling_std:.4f}",
                "relRMSE_proj mean±std": f"{r_ceiling_mean:.4f} ± {r_ceiling_std:.4f}",
                "ΔrelRMSE vs wiener": f"{(r_ceiling_mean - r_wiener_mean):+.4f}",
                "ΔrelRMSE vs ceiling": f"{0.0:+.4f}",
            }
        )
        perf_rows.append(
            {
                "P×Q": "wiener",
                "Pearson_proj mean±std": f"{p_wiener_mean:.4f} ± {p_wiener_std:.4f}",
                "relRMSE_proj mean±std": f"{r_wiener_mean:.4f} ± {r_wiener_std:.4f}",
                "ΔrelRMSE vs wiener": f"{0.0:+.4f}",
                "ΔrelRMSE vs ceiling": f"{(r_wiener_mean - r_ceiling_mean):+.4f}",
            }
        )
        perf_rows.append(
            {
                "P×Q": f"rect {rect_pair[0]}x{rect_pair[1]}",
                "Pearson_proj mean±std": f"{float(np.mean(rect_perf['pearson'])):.4f} ± {float(np.std(rect_perf['pearson'], ddof=1)):.4f}",
                "relRMSE_proj mean±std": f"{float(np.mean(rect_perf['relrmse'])):.4f} ± {float(np.std(rect_perf['relrmse'], ddof=1)):.4f}",
                "ΔrelRMSE vs wiener": f"{float(np.mean(rect_perf['dr_wiener'])):+.4f}",
                "ΔrelRMSE vs ceiling": f"{float(np.mean(rect_perf['dr_ceiling'])):+.4f}",
            }
        )

        for (P, Q) in pq_list:
            p_mean = float(np.mean(perf_by_pq[(P, Q)]["pearson"]))
            p_std = float(np.std(perf_by_pq[(P, Q)]["pearson"], ddof=1))
            r_mean = float(np.mean(perf_by_pq[(P, Q)]["relrmse"]))
            r_std = float(np.std(perf_by_pq[(P, Q)]["relrmse"], ddof=1))
            drw = float(np.mean(perf_by_pq[(P, Q)]["dr_wiener"]))
            drc = float(np.mean(perf_by_pq[(P, Q)]["dr_ceiling"]))
            perf_rows.append(
                {
                    "P×Q": f"{P}x{Q}",
                    "Pearson_proj mean±std": f"{p_mean:.4f} ± {p_std:.4f}",
                    "relRMSE_proj mean±std": f"{r_mean:.4f} ± {r_std:.4f}",
                    "ΔrelRMSE vs wiener": f"{drw:+.4f}",
                    "ΔrelRMSE vs ceiling": f"{drc:+.4f}",
                }
            )

            corr_mean = float(np.mean(sim_by_pq[(P, Q)]["corr"]))
            corr_std = float(np.std(sim_by_pq[(P, Q)]["corr"], ddof=1))
            rel_mean = float(np.mean(sim_by_pq[(P, Q)]["relL2"]))
            rel_std = float(np.std(sim_by_pq[(P, Q)]["relL2"], ddof=1))
            sim_rows.append(
                {
                    "P×Q": f"{P}x{Q}",
                    "corr_proj_wiener": f"{corr_mean:.4f} ± {corr_std:.4f}",
                    "relL2_proj_wiener": f"{rel_mean:.4f} ± {rel_std:.4f}",
                }
            )

        summary_md = (
            "# E44 — Fourier parity projection onto Wiener weights (alpha=2.0, low-k)\n\n"
            f"- run: `{paths.run_dir}`\n"
            f"- grid_size={grid_size}, k0_frac={k0_frac}, n_fields={n_fields}, patches_per_field={patches_per_field}\n"
            f"- w_big={w_big}, alpha={alpha}\n"
            f"- pq_list={pq_list}, rect_pair={rect_pair}\n"
            f"- wiener solver: lambda_rel={lambda_rel}, cg_max_iter={cg_max_iter}, cg_tol={cg_tol}\n\n"
            "## Performance (low-k)\n\n"
            + md_table(perf_rows, ["P×Q", "Pearson_proj mean±std", "relRMSE_proj mean±std", "ΔrelRMSE vs wiener", "ΔrelRMSE vs ceiling"])
            + "\n\n## Weight similarity vs Wiener (mean of gx/gy)\n\n"
            + md_table(sim_rows, ["P×Q", "corr_proj_wiener", "relL2_proj_wiener"])
            + "\n"
        )

        (paths.run_dir / "summary_e44_fourierproj_to_wiener_alpha2.md").write_text(summary_md, encoding="utf-8")
        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "grid_size": grid_size,
                "k0_frac": k0_frac,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "w_big": int(w_big),
                "alpha": float(alpha),
                "pq_list": pq_list,
                "rect_pair": rect_pair,
                "wiener": {"lambda_rel": lambda_rel, "cg_max_iter": cg_max_iter, "cg_tol": cg_tol},
            },
        )
        return paths

    if experiment == "e45":
        # E45 — Discrete-DFT parity basis (2π/N) projection to Wiener (alpha=2.0, w_big=193).
        #
        # Compare DFT-parity basis projection vs rect bins and (optional) pi/L Fourier baseline.
        grid_size = int(cfg.get("grid_size", 256))
        n_fields = int(cfg.get("n_fields", 10))
        patches_per_field = int(cfg.get("patches_per_field", 10_000))
        k0_frac = float(cfg.get("k0_frac", 0.15))
        w_big = _require_odd("w_big", int(cfg.get("w_big", 193)))
        alpha = float(cfg.get("alpha", 2.0))

        pq_list = cfg.get("pq_list", [[24, 12], [32, 16], [48, 24]])
        pq_list = [(int(p[0]), int(p[1])) for p in pq_list]

        rect_pair = cfg.get("rect_pair", [24, 12])
        rect_pair = (int(rect_pair[0]), int(rect_pair[1]))

        include_piL = bool(cfg.get("include_piL_baseline", True))
        piL_pair = cfg.get("piL_pair", [24, 12])
        piL_pair = (int(piL_pair[0]), int(piL_pair[1]))

        proj_lambda_rel = float(cfg.get("proj_lambda_rel", 1e-8))
        lambda_rel = float(cfg.get("lambda_rel", 1e-6))
        cg_max_iter = int(cfg.get("cg_max_iter", 200))
        cg_tol = float(cfg.get("cg_tol", 1e-6))

        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if n_fields < 2:
            raise ValueError("n_fields must be >= 2")
        if patches_per_field <= 0:
            raise ValueError("patches_per_field must be > 0")
        if not (0.0 < float(k0_frac) < 0.5):
            raise ValueError("k0_frac must be in (0,0.5)")
        if w_big > grid_size:
            raise ValueError("w_big must be <= grid_size")
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        if not pq_list:
            raise ValueError("pq_list must be non-empty")
        if any(p <= 0 or q <= 0 for p, q in pq_list):
            raise ValueError("pq_list entries must be positive")
        if rect_pair[0] <= 0 or rect_pair[1] <= 0:
            raise ValueError("rect_pair must be positive")
        if include_piL and (piL_pair[0] <= 0 or piL_pair[1] <= 0):
            raise ValueError("piL_pair must be positive")
        if proj_lambda_rel <= 0:
            raise ValueError("proj_lambda_rel must be > 0")
        if lambda_rel <= 0:
            raise ValueError("lambda_rel must be > 0")
        if cg_max_iter <= 0:
            raise ValueError("cg_max_iter must be > 0")
        if cg_tol <= 0:
            raise ValueError("cg_tol must be > 0")

        def safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            am = a - float(a.mean())
            bm = b - float(b.mean())
            denom = float(np.linalg.norm(am) * np.linalg.norm(bm)) + 1e-12
            return float((am @ bm) / denom)

        def relrmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
            e = y_pred - y_true
            rmse = float(np.sqrt(np.mean(e * e)))
            sd = float(np.std(y_true))
            return rmse / (sd + 1e-12)

        def pearson_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            px = safe_corr_1d(y_true[:, 0], y_pred[:, 0])
            py = safe_corr_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (px + py)

        def relrmse_mean_2d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            rx = relrmse_1d(y_true[:, 0], y_pred[:, 0])
            ry = relrmse_1d(y_true[:, 1], y_pred[:, 1])
            return 0.5 * (rx + ry)

        def kernel_fft_centered(kernel: np.ndarray, *, grid_size: int) -> np.ndarray:
            kernel = np.asarray(kernel, dtype=np.float64)
            if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
                raise ValueError(f"kernel must be square 2D, got {kernel.shape}")
            w = int(kernel.shape[0])
            if (w % 2) == 0:
                raise ValueError(f"kernel size must be odd, got {w}")
            if w > grid_size:
                raise ValueError(f"kernel size {w} exceeds grid_size {grid_size}")
            r = w // 2
            kr = np.flip(kernel, axis=(0, 1))
            full = np.zeros((grid_size, grid_size), dtype=np.float64)
            c0 = grid_size // 2
            full[c0 - r : c0 + r + 1, c0 - r : c0 + r + 1] = kr
            full0 = np.fft.ifftshift(full)
            return np.fft.fftn(full0)

        def md_table(rows: list[dict[str, str]], cols: list[str]) -> str:
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            out = [header, sep]
            for r0 in rows:
                out.append("| " + " | ".join(r0.get(c, "") for c in cols) + " |")
            return "\n".join(out)

        def candidate_edges(r: int) -> np.ndarray:
            base = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, r + 1]
            base = [int(x) for x in base if 0 <= int(x) <= r + 1]
            dense = list(range(0, min(16, r + 1) + 1))
            geom = np.unique(np.rint(np.geomspace(1, r + 1, num=64)).astype(int)).tolist()
            vals = sorted(set(base + dense + geom + [0, r + 1]))
            return np.asarray(vals, dtype=np.int64)

        def pick_edges(cand: np.ndarray, n_bins: int) -> np.ndarray:
            cand = np.asarray(cand, dtype=np.int64)
            if len(cand) < n_bins + 1:
                cand = np.arange(int(cand.min()), int(cand.max()) + 1, dtype=np.int64)
            idx = np.round(np.linspace(0, len(cand) - 1, n_bins + 1)).astype(int)
            idx[0] = 0
            idx[-1] = len(cand) - 1
            edges = np.unique(cand[idx])
            if len(edges) != n_bins + 1:
                edges = np.unique(np.rint(np.linspace(int(cand.min()), int(cand.max()), n_bins + 1)).astype(np.int64))
            edges[0] = int(cand.min())
            edges[-1] = int(cand.max())
            return edges

        def build_rect_bins(w: int, *, nx_bins: int, ny_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            w = _require_odd("w_big", int(w))
            r = w // 2
            coords = np.arange(w, dtype=np.float64) - float(r)
            dx, dy = np.meshgrid(coords, coords, indexing="ij")
            ax = np.abs(dx).astype(np.int64)
            ay = np.abs(dy).astype(np.int64)
            cand = candidate_edges(int(r))
            edges_x = pick_edges(cand, int(nx_bins))
            edges_y = pick_edges(cand, int(ny_bins))
            ix = np.digitize(ax, edges_x, right=False) - 1
            iy = np.digitize(ay, edges_y, right=False) - 1
            ix = np.clip(ix, 0, int(nx_bins) - 1).astype(np.int32, copy=False)
            iy = np.clip(iy, 0, int(ny_bins) - 1).astype(np.int32, copy=False)
            bin_idx = (ix * int(ny_bins) + iy).astype(np.int32, copy=False)
            return dx.astype(np.float64), dy.astype(np.float64), bin_idx

        def cg_solve(
            apply_A: Any,
            b: np.ndarray,
            *,
            max_iter: int,
            tol: float,
        ) -> tuple[np.ndarray, int, float]:
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            x = np.zeros_like(b)
            r = b - apply_A(x)
            p = r.copy()
            rs0 = float(r @ r)
            rs = rs0
            bnorm = float(np.sqrt(float(b @ b))) + 1e-12
            if bnorm == 0:
                return x, 0, 0.0
            for it in range(1, int(max_iter) + 1):
                Ap = apply_A(p)
                denom = float(p @ Ap) + 1e-18
                alpha = rs / denom
                x = x + alpha * p
                r = r - alpha * Ap
                rs_new = float(r @ r)
                if float(np.sqrt(rs_new)) <= float(tol) * bnorm:
                    return x, it, float(np.sqrt(rs_new)) / bnorm
                beta = rs_new / (rs + 1e-18)
                p = r + beta * p
                rs = rs_new
            return x, int(max_iter), float(np.sqrt(rs)) / bnorm

        def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))

        # Patch centers (shared across fields).
        r_big = w_big // 2
        centers_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_cent = np.random.default_rng(seed + 555_555 + 10_000 * int(w_big) + 1_000 * field_id)
            cx = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
            cy = rng_cent.integers(r_big, grid_size - r_big, size=patches_per_field, dtype=np.int64)
            centers_by_field.append(np.column_stack([cx, cy]).astype(np.int64, copy=False))

        def sample_vec_at_centers(gx: np.ndarray, gy: np.ndarray, centers: np.ndarray) -> np.ndarray:
            cx = centers[:, 0].astype(np.int64)
            cy = centers[:, 1].astype(np.int64)
            return np.column_stack([gx[cx, cy], gy[cx, cy]]).astype(np.float64, copy=False)

        # Generate fields (alpha fixed).
        rho01_by_field: list[np.ndarray] = []
        rho0_by_field: list[np.ndarray] = []
        rho0_fft_by_field: list[np.ndarray] = []
        split_cache: list[BandSplit2D] = []
        y_low_by_field: list[np.ndarray] = []
        for field_id in range(n_fields):
            rng_field = np.random.default_rng(seed + field_id)
            rho01 = generate_1overf_field_2d((grid_size, grid_size), alpha=float(alpha), rng=rng_field)
            split = band_split_poisson_2d(rho01, k0_frac=float(k0_frac))
            rho01_by_field.append(rho01.astype(np.float64, copy=False))
            rho0 = (rho01 - float(rho01.mean())).astype(np.float64, copy=False)
            rho0_by_field.append(rho0)
            rho0_fft_by_field.append(np.fft.fftn(rho0))
            split_cache.append(split)
            y_low_by_field.append(sample_vec_at_centers(split.low.gx, split.low.gy, centers_by_field[field_id]))

        # Ceiling kernel (truncated impulse).
        rho_delta = np.zeros((grid_size, grid_size), dtype=np.float64)
        c0 = grid_size // 2
        rho_delta[c0, c0] = 1.0
        split_delta = band_split_poisson_2d(rho_delta, k0_frac=float(k0_frac))
        g_patch_low_gx = split_delta.low.gx[c0 - r_big : c0 + r_big + 1, c0 - r_big : c0 + r_big + 1]
        g_patch_low_gy = split_delta.low.gy[c0 - r_big : c0 + r_big + 1, c0 - r_big : c0 + r_big + 1]
        kcorr_low_gx = g_patch_low_gx[::-1, ::-1].astype(np.float64, copy=False)
        kcorr_low_gy = g_patch_low_gy[::-1, ::-1].astype(np.float64, copy=False)
        kfft_ceiling_gx = kernel_fft_centered(kcorr_low_gx, grid_size=grid_size)
        kfft_ceiling_gy = kernel_fft_centered(kcorr_low_gy, grid_size=grid_size)

        # Fourier-domain transfer H for low-k (independent of alpha/rho).
        nx = grid_size
        kx1 = 2.0 * np.pi * np.fft.fftfreq(nx).astype(np.float64)
        ky1 = 2.0 * np.pi * np.fft.fftfreq(nx).astype(np.float64)
        kx = kx1[:, None]
        ky = ky1[None, :]
        k2 = kx * kx + ky * ky
        k = np.sqrt(k2, dtype=np.float64)
        k_ny = np.pi
        k0 = float(k0_frac) * k_ny
        mask_low = k <= k0
        H_gx = np.zeros((nx, nx), dtype=np.complex128)
        H_gy = np.zeros((nx, nx), dtype=np.complex128)
        nonzero = k2 > 0
        kx2 = np.broadcast_to(kx, (nx, nx))
        ky2 = np.broadcast_to(ky, (nx, nx))
        H_gx[nonzero] = -(1j * kx2[nonzero] / k2[nonzero]) * mask_low[nonzero]
        H_gy[nonzero] = -(1j * ky2[nonzero] / k2[nonzero]) * mask_low[nonzero]
        H_gx[0, 0] = 0.0 + 0.0j
        H_gy[0, 0] = 0.0 + 0.0j
        conjH_gx = np.conj(H_gx)
        conjH_gy = np.conj(H_gy)

        sx = slice(c0 - r_big, c0 + r_big + 1)
        sy = slice(c0 - r_big, c0 + r_big + 1)

        def wiener_fit_weights(train_fields: list[int]) -> tuple[np.ndarray, np.ndarray, float]:
            acc = np.zeros((grid_size, grid_size), dtype=np.float64)
            var_acc = 0.0
            for fid in train_fields:
                R = rho0_fft_by_field[fid]
                acc += (R * np.conj(R)).real
                var_acc += float(np.mean(rho0_by_field[fid] * rho0_by_field[fid]))
            S_rr = (acc / float(len(train_fields))) / float(grid_size * grid_size)
            var0 = var_acc / float(len(train_fields))
            lam = float(lambda_rel) * float(var0)

            r_rgx = np.fft.ifftn(S_rr * conjH_gx).real
            r_rgy = np.fft.ifftn(S_rr * conjH_gy).real
            r_rgx_c = np.fft.fftshift(r_rgx)
            r_rgy_c = np.fft.fftshift(r_rgy)
            b_gx = r_rgx_c[sx, sy].reshape(-1).astype(np.float64, copy=False)
            b_gy = r_rgy_c[sx, sy].reshape(-1).astype(np.float64, copy=False)

            V = np.zeros((grid_size, grid_size), dtype=np.float64)

            def apply_A(v: np.ndarray, *, S: np.ndarray) -> np.ndarray:
                v = np.asarray(v, dtype=np.float64).reshape(w_big, w_big)
                V.fill(0.0)
                V[sx, sy] = v
                conv = np.fft.ifftn(np.fft.fftn(V) * S).real
                out = conv[sx, sy].reshape(-1).astype(np.float64, copy=False)
                return out + lam * v.reshape(-1)

            wx, itx, relresx = cg_solve(lambda v: apply_A(v, S=S_rr), b_gx, max_iter=cg_max_iter, tol=cg_tol)
            wy, ity, relresy = cg_solve(lambda v: apply_A(v, S=S_rr), b_gy, max_iter=cg_max_iter, tol=cg_tol)
            info = float(max(relresx, relresy))
            return wx.reshape(w_big, w_big), wy.reshape(w_big, w_big), info

        # Rect baseline bins.
        rect_dx, rect_dy, rect_bin_idx = build_rect_bins(w_big, nx_bins=rect_pair[0], ny_bins=rect_pair[1])
        rect_n_bins = rect_pair[0] * rect_pair[1]

        # Precompute trig tables for DFT parity basis.
        N = w_big
        coords = (np.arange(w_big, dtype=np.float64) - float(r_big)).astype(np.float64)
        dx_grid, dy_grid = np.meshgrid(coords, coords, indexing="ij")
        Pmax = max(p for p, _ in pq_list)
        Qmax = max(q for _, q in pq_list)
        sin_p = np.zeros((Pmax, N), dtype=np.float64)
        cos_q = np.zeros((Qmax, N), dtype=np.float64)
        for p in range(1, Pmax + 1):
            sin_p[p - 1] = np.sin(2.0 * np.pi * float(p) * coords / float(N))
        for q in range(0, Qmax):
            cos_q[q] = np.cos(2.0 * np.pi * float(q) * coords / float(N))
        sin_norm = np.sum(sin_p * sin_p, axis=1)
        cos_norm = np.sum(cos_q * cos_q, axis=1)

        # Optional pi/L basis for baseline (24x12).
        if include_piL:
            L = float(r_big + 1)
            sin_p_pi = np.zeros((piL_pair[0], N), dtype=np.float64)
            cos_q_pi = np.zeros((piL_pair[1], N), dtype=np.float64)
            for p in range(1, piL_pair[0] + 1):
                sin_p_pi[p - 1] = np.sin(np.pi * float(p) * coords / L)
            for q in range(0, piL_pair[1]):
                cos_q_pi[q] = np.cos(np.pi * float(q) * coords / L)
            sin_norm_pi = np.sum(sin_p_pi * sin_p_pi, axis=1)
            cos_norm_pi = np.sum(cos_q_pi * cos_q_pi, axis=1)

        # Baseline performance arrays.
        p_ceiling_f: list[float] = []
        r_ceiling_f: list[float] = []
        p_wiener_f: list[float] = []
        r_wiener_f: list[float] = []

        # DFT basis accumulators.
        perf_by_pq: dict[tuple[int, int], dict[str, list[float]]] = {}
        sim_by_pq: dict[tuple[int, int], dict[str, list[float]]] = {}
        for pair in pq_list:
            perf_by_pq[pair] = {"pearson": [], "relrmse": [], "dr_wiener": [], "dr_ceiling": []}
            sim_by_pq[pair] = {"corr": [], "relL2": []}

        rect_perf: dict[str, list[float]] = {"pearson": [], "relrmse": [], "dr_wiener": [], "dr_ceiling": []}
        piL_perf: dict[str, list[float]] = {"pearson": [], "relrmse": [], "dr_wiener": [], "dr_ceiling": []}

        # LOFO.
        for test_field in range(n_fields):
            train_fields = [f for f in range(n_fields) if f != test_field]

            w_wiener_gx, w_wiener_gy, cg_info = wiener_fit_weights(train_fields)

            # Wiener predictions.
            kfft_w_gx = kernel_fft_centered(w_wiener_gx, grid_size=grid_size)
            kfft_w_gy = kernel_fft_centered(w_wiener_gy, grid_size=grid_size)
            gx_pred = np.fft.ifftn(rho0_fft_by_field[test_field] * kfft_w_gx).real
            gy_pred = np.fft.ifftn(rho0_fft_by_field[test_field] * kfft_w_gy).real
            y_pred_w = sample_vec_at_centers(gx_pred, gy_pred, centers_by_field[test_field])

            # Ceiling predictions.
            gx_c = np.fft.ifftn(rho0_fft_by_field[test_field] * kfft_ceiling_gx).real
            gy_c = np.fft.ifftn(rho0_fft_by_field[test_field] * kfft_ceiling_gy).real
            y_pred_c = sample_vec_at_centers(gx_c, gy_c, centers_by_field[test_field])

            y_true = y_low_by_field[test_field]
            p_ceiling_f.append(pearson_mean_2d(y_true, y_pred_c))
            r_ceiling_f.append(relrmse_mean_2d(y_true, y_pred_c))
            p_wiener_f.append(pearson_mean_2d(y_true, y_pred_w))
            r_wiener_f.append(relrmse_mean_2d(y_true, y_pred_w))

            # Rect baseline projection.
            rect_bin_flat = rect_bin_idx.reshape(-1).astype(np.int64, copy=False)
            rect_dx_flat = rect_dx.reshape(-1).astype(np.float64, copy=False)
            rect_dy_flat = rect_dy.reshape(-1).astype(np.float64, copy=False)
            num_x = np.bincount(rect_bin_flat, weights=w_wiener_gx.reshape(-1) * rect_dx_flat, minlength=rect_n_bins)
            den_x = np.bincount(rect_bin_flat, weights=rect_dx_flat * rect_dx_flat, minlength=rect_n_bins) + 1e-12
            coef_dx = (num_x / den_x).astype(np.float64, copy=False)
            num_y = np.bincount(rect_bin_flat, weights=w_wiener_gy.reshape(-1) * rect_dy_flat, minlength=rect_n_bins)
            den_y = np.bincount(rect_bin_flat, weights=rect_dy_flat * rect_dy_flat, minlength=rect_n_bins) + 1e-12
            coef_dy = (num_y / den_y).astype(np.float64, copy=False)
            w_rect_gx = (rect_dx * coef_dx[rect_bin_idx]).astype(np.float64, copy=False)
            w_rect_gy = (rect_dy * coef_dy[rect_bin_idx]).astype(np.float64, copy=False)
            kfft_r_gx = kernel_fft_centered(w_rect_gx, grid_size=grid_size)
            kfft_r_gy = kernel_fft_centered(w_rect_gy, grid_size=grid_size)
            gx_r = np.fft.ifftn(rho0_fft_by_field[test_field] * kfft_r_gx).real
            gy_r = np.fft.ifftn(rho0_fft_by_field[test_field] * kfft_r_gy).real
            y_pred_r = sample_vec_at_centers(gx_r, gy_r, centers_by_field[test_field])
            p_r = pearson_mean_2d(y_true, y_pred_r)
            r_r = relrmse_mean_2d(y_true, y_pred_r)
            rect_perf["pearson"].append(p_r)
            rect_perf["relrmse"].append(r_r)
            rect_perf["dr_wiener"].append(r_r - r_wiener_f[-1])
            rect_perf["dr_ceiling"].append(r_r - r_ceiling_f[-1])

            # Optional pi/L baseline projection (same parity basis but π/L).
            if include_piL:
                Pp, Qp = piL_pair
                Wgx = w_wiener_gx
                Wgy = w_wiener_gy
                # gx: sin_p_pi * cos_q_pi
                Rpx = sin_p_pi[:Pp] @ Wgx
                num = Rpx @ cos_q_pi[:Qp].T
                den = sin_norm_pi[:Pp][:, None] * cos_norm_pi[:Qp][None, :]
                lam = float(proj_lambda_rel) * (float(np.sum(den)) / float(Pp * Qp))
                A = num / (den + lam)
                V = A @ cos_q_pi[:Qp]
                w_pi_gx = sin_p_pi[:Pp].T @ V
                # gy: cos_q_pi * sin_p_pi
                Rpy = cos_q_pi[:Qp] @ Wgy
                numy = Rpy @ sin_p_pi[:Pp].T  # Q x P
                den_y = cos_norm_pi[:Qp][:, None] * sin_norm_pi[:Pp][None, :]
                Ay = (numy / (den_y + lam)).T  # P x Q
                VY = Ay.T @ sin_p_pi[:Pp]
                w_pi_gy = cos_q_pi[:Qp].T @ VY

                kfft_p_gx = kernel_fft_centered(w_pi_gx, grid_size=grid_size)
                kfft_p_gy = kernel_fft_centered(w_pi_gy, grid_size=grid_size)
                gx_p = np.fft.ifftn(rho0_fft_by_field[test_field] * kfft_p_gx).real
                gy_p = np.fft.ifftn(rho0_fft_by_field[test_field] * kfft_p_gy).real
                y_pred_p = sample_vec_at_centers(gx_p, gy_p, centers_by_field[test_field])
                p_pi = pearson_mean_2d(y_true, y_pred_p)
                r_pi = relrmse_mean_2d(y_true, y_pred_p)
                piL_perf["pearson"].append(p_pi)
                piL_perf["relrmse"].append(r_pi)
                piL_perf["dr_wiener"].append(r_pi - r_wiener_f[-1])
                piL_perf["dr_ceiling"].append(r_pi - r_ceiling_f[-1])

            # DFT parity projections (2π/N).
            for (P, Q) in pq_list:
                Wgx = w_wiener_gx
                Wgy = w_wiener_gy
                # gx: sin_p * cos_q
                Rpx = sin_p[:P] @ Wgx
                num = Rpx @ cos_q[:Q].T
                den = sin_norm[:P][:, None] * cos_norm[:Q][None, :]
                lam = float(proj_lambda_rel) * (float(np.sum(den)) / float(P * Q))
                A = num / (den + lam)
                V = A @ cos_q[:Q]
                w_proj_gx = sin_p[:P].T @ V

                # gy: cos_q * sin_p (with p along y)
                Rpy = cos_q[:Q] @ Wgy
                numy = Rpy @ sin_p[:P].T  # Q x P
                den_y = cos_norm[:Q][:, None] * sin_norm[:P][None, :]
                Ay = (numy / (den_y + lam)).T  # P x Q
                VY = Ay.T @ sin_p[:P]
                w_proj_gy = cos_q[:Q].T @ VY

                # Similarity vs Wiener.
                corr_gx = safe_corr_1d(w_proj_gx, Wgx)
                corr_gy = safe_corr_1d(w_proj_gy, Wgy)
                rel_gx = rel_l2(w_proj_gx, Wgx)
                rel_gy = rel_l2(w_proj_gy, Wgy)
                sim_by_pq[(P, Q)]["corr"].append(0.5 * (corr_gx + corr_gy))
                sim_by_pq[(P, Q)]["relL2"].append(0.5 * (rel_gx + rel_gy))

                kfft_p_gx = kernel_fft_centered(w_proj_gx, grid_size=grid_size)
                kfft_p_gy = kernel_fft_centered(w_proj_gy, grid_size=grid_size)
                gx_p = np.fft.ifftn(rho0_fft_by_field[test_field] * kfft_p_gx).real
                gy_p = np.fft.ifftn(rho0_fft_by_field[test_field] * kfft_p_gy).real
                y_pred_p = sample_vec_at_centers(gx_p, gy_p, centers_by_field[test_field])
                p_proj = pearson_mean_2d(y_true, y_pred_p)
                r_proj = relrmse_mean_2d(y_true, y_pred_p)
                perf_by_pq[(P, Q)]["pearson"].append(p_proj)
                perf_by_pq[(P, Q)]["relrmse"].append(r_proj)
                perf_by_pq[(P, Q)]["dr_wiener"].append(r_proj - r_wiener_f[-1])
                perf_by_pq[(P, Q)]["dr_ceiling"].append(r_proj - r_ceiling_f[-1])

        # Baseline stats.
        p_ceiling_mean = float(np.mean(p_ceiling_f))
        p_ceiling_std = float(np.std(p_ceiling_f, ddof=1))
        r_ceiling_mean = float(np.mean(r_ceiling_f))
        r_ceiling_std = float(np.std(r_ceiling_f, ddof=1))
        p_wiener_mean = float(np.mean(p_wiener_f))
        p_wiener_std = float(np.std(p_wiener_f, ddof=1))
        r_wiener_mean = float(np.mean(r_wiener_f))
        r_wiener_std = float(np.std(r_wiener_f, ddof=1))

        perf_rows: list[dict[str, str]] = []
        sim_rows: list[dict[str, str]] = []

        perf_rows.append(
            {
                "P×Q": "ceiling",
                "Pearson_proj mean±std": f"{p_ceiling_mean:.4f} ± {p_ceiling_std:.4f}",
                "relRMSE_proj mean±std": f"{r_ceiling_mean:.4f} ± {r_ceiling_std:.4f}",
                "ΔrelRMSE vs wiener": f"{(r_ceiling_mean - r_wiener_mean):+.4f}",
                "ΔrelRMSE vs ceiling": f"{0.0:+.4f}",
            }
        )
        perf_rows.append(
            {
                "P×Q": "wiener",
                "Pearson_proj mean±std": f"{p_wiener_mean:.4f} ± {p_wiener_std:.4f}",
                "relRMSE_proj mean±std": f"{r_wiener_mean:.4f} ± {r_wiener_std:.4f}",
                "ΔrelRMSE vs wiener": f"{0.0:+.4f}",
                "ΔrelRMSE vs ceiling": f"{(r_wiener_mean - r_ceiling_mean):+.4f}",
            }
        )
        perf_rows.append(
            {
                "P×Q": f"rect {rect_pair[0]}x{rect_pair[1]}",
                "Pearson_proj mean±std": f"{float(np.mean(rect_perf['pearson'])):.4f} ± {float(np.std(rect_perf['pearson'], ddof=1)):.4f}",
                "relRMSE_proj mean±std": f"{float(np.mean(rect_perf['relrmse'])):.4f} ± {float(np.std(rect_perf['relrmse'], ddof=1)):.4f}",
                "ΔrelRMSE vs wiener": f"{float(np.mean(rect_perf['dr_wiener'])):+.4f}",
                "ΔrelRMSE vs ceiling": f"{float(np.mean(rect_perf['dr_ceiling'])):+.4f}",
            }
        )
        if include_piL:
            perf_rows.append(
                {
                    "P×Q": f"pi/L {piL_pair[0]}x{piL_pair[1]}",
                    "Pearson_proj mean±std": f"{float(np.mean(piL_perf['pearson'])):.4f} ± {float(np.std(piL_perf['pearson'], ddof=1)):.4f}",
                    "relRMSE_proj mean±std": f"{float(np.mean(piL_perf['relrmse'])):.4f} ± {float(np.std(piL_perf['relrmse'], ddof=1)):.4f}",
                    "ΔrelRMSE vs wiener": f"{float(np.mean(piL_perf['dr_wiener'])):+.4f}",
                    "ΔrelRMSE vs ceiling": f"{float(np.mean(piL_perf['dr_ceiling'])):+.4f}",
                }
            )

        for (P, Q) in pq_list:
            p_mean = float(np.mean(perf_by_pq[(P, Q)]["pearson"]))
            p_std = float(np.std(perf_by_pq[(P, Q)]["pearson"], ddof=1))
            r_mean = float(np.mean(perf_by_pq[(P, Q)]["relrmse"]))
            r_std = float(np.std(perf_by_pq[(P, Q)]["relrmse"], ddof=1))
            drw = float(np.mean(perf_by_pq[(P, Q)]["dr_wiener"]))
            drc = float(np.mean(perf_by_pq[(P, Q)]["dr_ceiling"]))
            perf_rows.append(
                {
                    "P×Q": f"{P}x{Q}",
                    "Pearson_proj mean±std": f"{p_mean:.4f} ± {p_std:.4f}",
                    "relRMSE_proj mean±std": f"{r_mean:.4f} ± {r_std:.4f}",
                    "ΔrelRMSE vs wiener": f"{drw:+.4f}",
                    "ΔrelRMSE vs ceiling": f"{drc:+.4f}",
                }
            )

            corr_mean = float(np.mean(sim_by_pq[(P, Q)]["corr"]))
            corr_std = float(np.std(sim_by_pq[(P, Q)]["corr"], ddof=1))
            rel_mean = float(np.mean(sim_by_pq[(P, Q)]["relL2"]))
            rel_std = float(np.std(sim_by_pq[(P, Q)]["relL2"], ddof=1))
            sim_rows.append(
                {
                    "P×Q": f"{P}x{Q}",
                    "corr_proj_wiener": f"{corr_mean:.4f} ± {corr_std:.4f}",
                    "relL2_proj_wiener": f"{rel_mean:.4f} ± {rel_std:.4f}",
                }
            )

        summary_md = (
            "# E45 — DFT parity projection onto Wiener weights (alpha=2.0, low-k)\n\n"
            f"- run: `{paths.run_dir}`\n"
            f"- grid_size={grid_size}, k0_frac={k0_frac}, n_fields={n_fields}, patches_per_field={patches_per_field}\n"
            f"- w_big={w_big}, alpha={alpha}\n"
            f"- pq_list={pq_list}, rect_pair={rect_pair}, include_piL={include_piL}, piL_pair={piL_pair}\n"
            f"- wiener solver: lambda_rel={lambda_rel}, cg_max_iter={cg_max_iter}, cg_tol={cg_tol}\n\n"
            "## Performance (low-k)\n\n"
            + md_table(perf_rows, ["P×Q", "Pearson_proj mean±std", "relRMSE_proj mean±std", "ΔrelRMSE vs wiener", "ΔrelRMSE vs ceiling"])
            + "\n\n## Weight similarity vs Wiener (mean of gx/gy)\n\n"
            + md_table(sim_rows, ["P×Q", "corr_proj_wiener", "relL2_proj_wiener"])
            + "\n"
        )

        (paths.run_dir / "summary_e45_dftparityproj_to_wiener_alpha2.md").write_text(summary_md, encoding="utf-8")
        write_json(
            paths.metrics_json,
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "grid_size": grid_size,
                "k0_frac": k0_frac,
                "n_fields": n_fields,
                "patches_per_field": patches_per_field,
                "w_big": int(w_big),
                "alpha": float(alpha),
                "pq_list": pq_list,
                "rect_pair": rect_pair,
                "include_piL": include_piL,
                "piL_pair": piL_pair,
                "proj_lambda_rel": proj_lambda_rel,
                "wiener": {"lambda_rel": lambda_rel, "cg_max_iter": cg_max_iter, "cg_tol": cg_tol},
            },
        )
        return paths

    if experiment == "e3":
        sigma_path = Path(str(cfg.get("sigma_path", "")))
        g_path = Path(str(cfg.get("g_path", "")))
        if not sigma_path.exists():
            raise FileNotFoundError(f"sigma_path not found: {sigma_path}")
        if not g_path.exists():
            raise FileNotFoundError(f"g_path not found: {g_path}")

        sigma = np.load(sigma_path)
        g_proj = np.load(g_path)
        if sigma.ndim != 2 or g_proj.ndim != 2:
            raise ValueError(f"Sigma_norm.npy and g_proj.npy must be 2D, got {sigma.shape}, {g_proj.shape}")
        if sigma.shape != g_proj.shape:
            raise ValueError(f"sigma and g_proj shape mismatch: {sigma.shape} vs {g_proj.shape}")
        sigma = np.asarray(sigma, dtype=np.float64)
        g_proj = np.asarray(g_proj, dtype=np.float64)
        assert_finite("sigma", sigma)
        assert_finite("g_proj", g_proj)
        if (sigma.min() < -1e-6) or (sigma.max() > 1.0 + 1e-6):
            raise ValueError("Sigma_norm.npy must be in [0,1]")

        feats, y = _sample_patches_2d(
            sigma=sigma,
            g_proj=g_proj,
            n_patches=n_patches,
            patch_size=patch_size,
            thresholds_b0=thresholds_b0,
            rng=rng,
            shuffle_b0=False,
        )
        train_idx, test_idx = _split_indices(n_patches, test_frac=test_frac, rng=rng_split)
        metrics, preds = _run_models(
            feats=feats,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            thresholds_b0=thresholds_b0,
            ridge_alpha=ridge_alpha,
        )
        metrics.update(
            {
                "experiment": experiment,
                "exp_name": exp_name,
                "seed": seed,
                "split_seed": split_seed,
                "n_patches": n_patches,
                "patch_size": patch_size,
                "test_frac": test_frac,
                "ridge_alpha": ridge_alpha,
                "thresholds_b0": thresholds_b0,
                "sigma_path": str(sigma_path),
                "g_path": str(g_path),
                "grid_shape": list(sigma.shape),
            }
        )
        split_is_test = np.zeros(n_patches, dtype=bool)
        split_is_test[test_idx] = True
        save_pred_vs_true(
            paths.pred_vs_true_csv,
            y=y,
            split_is_test=split_is_test,
            mass=np.asarray(feats["mass"], dtype=np.float64),
            preds=preds,
        )

        y_test = y[test_idx]
        pC_test = preds["C"][test_idx]
        plot_pred_vs_true(
            paths.pred_vs_true_png,
            y_true=y_test,
            y_pred=pC_test,
            title=f"{exp_name} (model C, test)",
        )
        plot_residuals_hist(
            paths.residuals_hist_png,
            residuals=(pC_test - y_test),
            title=f"{exp_name} residuals (model C, test)",
        )
        write_json(paths.metrics_json, metrics)
        return paths

    raise ValueError(
        f"Unknown experiment: {experiment} (expected e0/e1/e2/e3/e4/e5/e6/e7/e8/e10/e11/e12/e13/e14/e15/e15b/e15c/e16/e17/e18/e19/e20/e21/e21b/e22/e23/e24/e25/e26/e27/e28/e29)"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str, help="Path to YAML/JSON config.")
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config value at runtime (key=value). Can be repeated.",
    )
    args = p.parse_args()

    cfg = read_config(Path(args.config))
    if args.override:
        cfg = apply_overrides(cfg, list(args.override))
    paths = run_experiment(cfg)
    print(str(paths.run_dir))


if __name__ == "__main__":
    main()
