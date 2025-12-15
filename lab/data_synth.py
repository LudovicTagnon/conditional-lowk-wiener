from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import assert_finite, normalize_01


@dataclass(frozen=True)
class PoissonResult3D:
    rho: np.ndarray  # (nx, ny, nz) in [0, 1]
    phi: np.ndarray  # (nx, ny, nz) real
    g: np.ndarray  # (nx, ny, nz) = ||grad(phi)||


def generate_1overf_field_3d(
    shape: tuple[int, int, int],
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a real 3D field with power spectrum ~ 1/|k|^alpha using rFFT.

    Implementation notes:
    - We draw complex Gaussian coefficients in Fourier space and scale by |k|^{-alpha/2}.
    - The k=0 mode is set to 0 to avoid a singularity / DC bias.
    """
    if len(shape) != 3:
        raise ValueError(f"shape must be 3D, got {shape}")
    if any(s <= 0 for s in shape):
        raise ValueError(f"invalid shape: {shape}")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    nx, ny, nz = shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx)[:, None, None]
    ky = 2.0 * np.pi * np.fft.fftfreq(ny)[None, :, None]
    kz = 2.0 * np.pi * np.fft.rfftfreq(nz)[None, None, :]
    k2 = kx * kx + ky * ky + kz * kz
    k = np.sqrt(k2, dtype=np.float64)

    scale = np.zeros_like(k, dtype=np.float64)
    nonzero = k > 0
    scale[nonzero] = k[nonzero] ** (-alpha / 2.0)

    real = rng.normal(size=(nx, ny, nz // 2 + 1))
    imag = rng.normal(size=(nx, ny, nz // 2 + 1))
    coeff = (real + 1j * imag) * scale
    coeff[0, 0, 0] = 0.0 + 0.0j

    field = np.fft.irfftn(coeff, s=shape).real
    assert_finite("rho_raw", field)
    rho = normalize_01(field)
    if (rho.min() < -1e-9) or (rho.max() > 1.0 + 1e-9):
        raise RuntimeError("normalize_01 returned values outside [0,1]")
    return rho


def solve_poisson_periodic_fft(rho: np.ndarray) -> PoissonResult3D:
    """
    Solve periodic Poisson via FFT: phi_k = -rho_k / k^2 (k=0 -> 0).
    Compute g = ||∇phi|| via spectral derivatives: d/dx <-> i*kx.

    Notes:
    - Periodic Poisson has a solution only if the source has zero mean; we subtract mean(rho)
      before solving. This does not change g, since the DC mode affects only a constant offset in phi.
    """
    rho = np.asarray(rho, dtype=np.float64)
    if rho.ndim != 3:
        raise ValueError(f"rho must be 3D, got shape {rho.shape}")
    assert_finite("rho", rho)

    nx, ny, nz = rho.shape
    rho0 = rho - float(rho.mean())
    rho_k = np.fft.fftn(rho0)

    kx = 2.0 * np.pi * np.fft.fftfreq(nx)[:, None, None]
    ky = 2.0 * np.pi * np.fft.fftfreq(ny)[None, :, None]
    kz = 2.0 * np.pi * np.fft.fftfreq(nz)[None, None, :]
    k2 = kx * kx + ky * ky + kz * kz

    phi_k = np.zeros_like(rho_k)
    nonzero = k2 > 0
    phi_k[nonzero] = -rho_k[nonzero] / k2[nonzero]
    phi_k[0, 0, 0] = 0.0 + 0.0j

    phi = np.fft.ifftn(phi_k).real
    assert_finite("phi", phi)

    gx = np.fft.ifftn(1j * kx * phi_k).real
    gy = np.fft.ifftn(1j * ky * phi_k).real
    gz = np.fft.ifftn(1j * kz * phi_k).real
    assert_finite("gx", gx)
    assert_finite("gy", gy)
    assert_finite("gz", gz)
    g = np.sqrt(gx * gx + gy * gy + gz * gz, dtype=np.float64)
    assert_finite("g", g)
    return PoissonResult3D(rho=rho, phi=phi, g=g)
