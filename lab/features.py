from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import ndimage

from .utils import assert_finite


def grad_energy(patch: np.ndarray) -> float:
    patch = np.asarray(patch, dtype=np.float64)
    assert_finite("patch", patch)
    grads = np.gradient(patch)
    energy = np.zeros_like(patch, dtype=np.float64)
    for g in grads:
        energy += np.asarray(g, dtype=np.float64) ** 2
    return float(energy.mean())


def b0_multi(
    patch: np.ndarray,
    thresholds: Iterable[float] = (0.4, 0.5, 0.6, 0.7),
    *,
    adjacency: int,
) -> dict[str, int]:
    """
    Multi-threshold b0 (connected components count) helper.

    Returns keys:
    - 0.4 -> b0_t04, 0.5 -> b0_t05, 0.6 -> b0_t06, 0.7 -> b0_t07
    """
    ts = [float(t) for t in thresholds]
    counts = b0_counts(patch, ts, adjacency=adjacency)
    out: dict[str, int] = {}
    for t, c in zip(ts, counts):
        key = f"b0_t{int(round(t * 10)):02d}"
        out[key] = int(c)
    return out


def b0_named(
    patch: np.ndarray,
    thresholds_by_name: dict[str, float],
    *,
    adjacency: int,
) -> dict[str, int]:
    names = list(thresholds_by_name.keys())
    thresholds = [float(thresholds_by_name[n]) for n in names]
    counts = b0_counts(patch, thresholds, adjacency=adjacency)
    return {f"b0_{n}": int(c) for n, c in zip(names, counts)}


def b0_counts(
    patch: np.ndarray,
    thresholds: Iterable[float],
    adjacency: int,
) -> list[int]:
    patch = np.asarray(patch, dtype=np.float64)
    if patch.ndim not in (2, 3):
        raise ValueError(f"patch must be 2D/3D, got shape {patch.shape}")
    if patch.ndim == 2 and adjacency != 4:
        raise ValueError("2D b0 requires 4-connectivity (adjacency=4)")
    if patch.ndim == 3 and adjacency != 6:
        raise ValueError("3D b0 requires 6-connectivity (adjacency=6)")

    # ndimage: connectivity=1 corresponds to 4-neigh in 2D and 6-neigh in 3D.
    structure = ndimage.generate_binary_structure(patch.ndim, 1)
    out: list[int] = []
    for t in thresholds:
        binary = patch > float(t)
        if not binary.any():
            out.append(0)
            continue
        _, num = ndimage.label(binary, structure=structure)
        out.append(int(num))
    return out


def patch_basic_stats(patch: np.ndarray) -> tuple[float, float, float, float]:
    patch = np.asarray(patch, dtype=np.float64)
    assert_finite("patch", patch)
    mass = float(patch.sum())
    mass2 = float(mass * mass)
    var = float(patch.var())
    mx = float(patch.max())
    return mass, mass2, var, mx


def nonlocal_annulus_moments(
    patch_big: np.ndarray,
    *,
    w_local: int,
    n_radial_bins: int = 6,
    include_dipole: bool = True,
    include_quadrupole: bool = True,
    eps: float = 1e-12,
) -> dict[str, float]:
    """
    Compute "outer annulus" moments on a big patch, excluding the inner local square core.

    Definitions:
    - Inner core: |dx| <= r_local AND |dy| <= r_local (square, consistent with w_local patches)
    - Outer mask: NOT(core) within patch_big
    - Coordinates dx,dy are integers centered at the patch center.
    - Radial rings are computed by binning r = sqrt(dx^2+dy^2) on the outer mask.

    Returns a dict with:
    - M0: sum(rho_outer)
    - ring_sum_k and ring_frac_k for k in [0..n_radial_bins-1]
    - optional dipole: Dx,Dy and normalized Dx_n,Dy_n
    - optional quadrupole (traceless): Qxx,Qxy and normalized Qxx_n,Qxy_n
    """
    patch_big = np.asarray(patch_big, dtype=np.float64)
    if patch_big.ndim != 2 or patch_big.shape[0] != patch_big.shape[1]:
        raise ValueError(f"patch_big must be square 2D, got {patch_big.shape}")
    w_big = int(patch_big.shape[0])
    if w_big <= 0 or (w_big % 2) == 0:
        raise ValueError(f"patch_big size must be positive odd, got {w_big}")
    w_local = int(w_local)
    if w_local <= 0 or (w_local % 2) == 0:
        raise ValueError(f"w_local must be positive odd, got {w_local}")
    if w_local > w_big:
        raise ValueError(f"w_local must be <= w_big, got {w_local} > {w_big}")
    n_radial_bins = int(n_radial_bins)
    if n_radial_bins <= 0:
        raise ValueError("n_radial_bins must be > 0")
    eps = float(eps)
    if eps <= 0:
        raise ValueError("eps must be > 0")
    assert_finite("patch_big", patch_big)

    r_big = w_big // 2
    r_local = w_local // 2
    coords = np.arange(w_big, dtype=np.float64) - float(r_big)
    dx = coords[:, None]
    dy = coords[None, :]
    core = (np.abs(dx) <= float(r_local)) & (np.abs(dy) <= float(r_local))
    outer = ~core

    rho_outer = patch_big * outer
    M0 = float(rho_outer.sum())
    denom = M0 + eps

    out: dict[str, float] = {"M0": M0}

    # Radial rings on the outer mask (equal-width bins in r).
    if r_big <= r_local or not outer.any():
        for k in range(n_radial_bins):
            out[f"ring_sum_{k}"] = 0.0
            out[f"ring_frac_{k}"] = 0.0
    else:
        r = np.sqrt(dx * dx + dy * dy, dtype=np.float64)
        edges = np.linspace(float(r_local), float(r_big) + 1e-9, n_radial_bins + 1, dtype=np.float64)
        for k in range(n_radial_bins):
            lo = edges[k]
            hi = edges[k + 1]
            if k == n_radial_bins - 1:
                m = outer & (r >= lo) & (r <= hi)
            else:
                m = outer & (r >= lo) & (r < hi)
            s = float(patch_big[m].sum()) if m.any() else 0.0
            out[f"ring_sum_{k}"] = s
            out[f"ring_frac_{k}"] = float(s / denom)

    if include_dipole:
        Dx = float((rho_outer * dx).sum())
        Dy = float((rho_outer * dy).sum())
        out["Dx"] = Dx
        out["Dy"] = Dy
        out["Dx_n"] = float(Dx / denom)
        out["Dy_n"] = float(Dy / denom)

    if include_quadrupole:
        qxx = float((rho_outer * (dx * dx - dy * dy)).sum())
        qxy = float((rho_outer * (2.0 * dx * dy)).sum())
        out["Qxx"] = qxx
        out["Qxy"] = qxy
        out["Qxx_n"] = float(qxx / denom)
        out["Qxy_n"] = float(qxy / denom)

    for k, v in out.items():
        if not np.isfinite(v):
            raise ValueError(f"nonlocal_annulus_moments produced non-finite {k}: {v}")
    return out
