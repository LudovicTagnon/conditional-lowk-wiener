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
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

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

        # theory kernel for sweep_w
        rho_delta = np.zeros((grid_size, grid_size), dtype=np.float64)
        c = grid_size // 2
        r = sweep_ps // 2
        rho_delta[c, c] = 1.0
        split_th = band_split_poisson_2d(rho_delta, k0_frac=k0_frac)
        kth_gx = split_th.high.gx[c - r : c + r + 1, c - r : c + r + 1][::-1, ::-1]
        kth_gy = split_th.high.gy[c - r : c + r + 1, c - r : c + r + 1][::-1, ::-1]

        # build training set (all fields)
        from numpy.lib.stride_tricks import sliding_window_view

        Xs: list[np.ndarray] = []
        Ys: list[np.ndarray] = []
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
            Xs.append(X)
            Ys.append(y)

        X_all = np.concatenate(Xs, axis=0)
        y_all = np.concatenate(Ys, axis=0)
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xs_all = scaler.fit_transform(X_all)

        sweep_rows: list[dict[str, Any]] = []
        for ra in ridge_alphas:
            model = Ridge(alpha=float(ra), fit_intercept=True, solver="sag", max_iter=2000, random_state=int(placebo_seed))
            model.fit(Xs_all, y_all)
            coef = np.asarray(model.coef_, dtype=np.float64)  # (2, p)
            kgx = coef[0].reshape(sweep_ps, sweep_ps)
            kgy = coef[1].reshape(sweep_ps, sweep_ps)
            ax, kgx_s = scale_best(kgx, kth_gx)
            ay, kgy_s = scale_best(kgy, kth_gy)

            best = {"corr": -1e9, "rel": 1e9, "dx": 0, "dy": 0}
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    x = shift_zero(kgx, dx, dy)
                    y = shift_zero(kgy, dx, dy)
                    _, xs = scale_best(x, kth_gx)
                    _, ys = scale_best(y, kth_gy)
                    cmean = 0.5 * (corr(xs, kth_gx) + corr(ys, kth_gy))
                    rmean = 0.5 * (rel_l2(xs, kth_gx) + rel_l2(ys, kth_gy))
                    if cmean > best["corr"]:
                        best = {"corr": cmean, "rel": rmean, "dx": dx, "dy": dy}

            sweep_rows.append(
                {
                    "ridge_alpha": float(ra),
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

        metrics = {
            "experiment": experiment,
            "exp_name": exp_name,
            "diag": diag_rows,
            "ridge_sweep": sweep_rows,
        }
        write_json(paths.metrics_json, metrics)
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
                "lofo_folds": fold_rows,
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

    raise ValueError(f"Unknown experiment: {experiment} (expected e0/e1/e2/e3/e4/e5/e6/e7/e8/e11/e12/e13/e14/e15/e16/e15b/e17)")


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
