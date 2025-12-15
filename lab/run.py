from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .data_synth import generate_1overf_field_3d, solve_poisson_periodic_fft
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

    raise ValueError(f"Unknown experiment: {experiment} (expected e0/e1/e2/e3/e4/e5/e6/e7/e8)")


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
