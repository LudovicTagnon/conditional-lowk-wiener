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
        f"Unknown experiment: {experiment} (expected e0/e1/e2/e3/e4/e5/e6/e7/e8/e10/e11/e12/e13/e14/e15/e15b/e15c/e16/e17/e18/e19/e20/e21/e21b/e22/e23/e24/e25/e26/e27/e28)"
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
