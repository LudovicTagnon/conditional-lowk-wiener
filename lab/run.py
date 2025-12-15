from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .data_synth import generate_1overf_field_3d, solve_poisson_periodic_fft
from .features import b0_counts, grad_energy, patch_basic_stats
from .models import deltas_vs_baseline, fit_predict_ridge
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

    output_root = Path(cfg.get("output_root", "outputs"))
    paths = make_run_paths(output_root=output_root, exp_name=exp_name)
    write_yaml(paths.config_used, cfg)

    thresholds_b0 = [float(x) for x in cfg.get("thresholds_b0", [0.4, 0.5, 0.6, 0.7])]
    n_patches = int(cfg.get("n_patches", 10_000))
    patch_size = int(cfg.get("patch_size", 9))
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

    raise ValueError(f"Unknown experiment: {experiment} (expected e0/e1/e2/e3)")


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
