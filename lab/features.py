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
