from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import assert_finite, normalize_01


@dataclass(frozen=True)
class PoissonResult3D:
    rho: np.ndarray  # (nx, ny, nz) in [0, 1]
    phi: np.ndarray  # (nx, ny, nz) real
    g: np.ndarray  # (nx, ny, nz) = ||grad(phi)||


@dataclass(frozen=True)
class PoissonResult2D:
    rho: np.ndarray  # (nx, ny) real (not necessarily [0,1] if preprocessed)
    phi: np.ndarray  # (nx, ny) real
    gx: np.ndarray  # (nx, ny)
    gy: np.ndarray  # (nx, ny)
    gmag: np.ndarray  # (nx, ny)


@dataclass(frozen=True)
class BandSplit2D:
    rho: np.ndarray
    rho_low: np.ndarray
    rho_high: np.ndarray
    low: PoissonResult2D
    high: PoissonResult2D
    full: PoissonResult2D
    rel_err_gx: float
    rel_err_gy: float


def normalize_rho(field: np.ndarray, mode: str = "minmax_01") -> np.ndarray:
    """
    Normalize a field for Poisson solving.
    - minmax_01: scale to [0,1]
    - delta: (rho - mean)/mean (per field)
    - zscore: (rho - mean)/std (per field)
    """
    field = np.asarray(field, dtype=np.float64)
    mode = str(mode).lower().strip()
    if mode == "minmax_01":
        rho = normalize_01(field)
        if (rho.min() < -1e-9) or (rho.max() > 1.0 + 1e-9):
            raise RuntimeError("normalize_01 returned values outside [0,1]")
        return rho
    if mode == "delta":
        mu = float(field.mean())
        denom = mu if abs(mu) > 1e-8 else (1e-8 if mu >= 0 else -1e-8)
        return (field - mu) / denom
    if mode == "zscore":
        mu = float(field.mean())
        sd = float(field.std())
        if sd <= 0:
            return field - mu
        return (field - mu) / sd
    raise ValueError(f"unsupported norm_mode={mode}")


def generate_1overf_field_2d(
    shape: tuple[int, int],
    alpha: float,
    rng: np.random.Generator,
    spectrum: str = "powerlaw",
    bbks_k0: float | None = None,
    bbks_ns: float = 1.0,
    bbks_beta: float = 0.0,
    bbks_k_eps: float = 1e-6,
    lognormal: bool = False,
    lognormal_sigma: float = 1.0,
    norm_mode: str = "minmax_01",
) -> np.ndarray:
    """
    Generate a real 2D field using rFFT.
    spectrum="powerlaw" uses ~1/|k|^alpha, spectrum="bbks" uses BBKS transfer.
    Returns rho normalized by norm_mode (lognormal optionally applied).
    """
    if len(shape) != 2:
        raise ValueError(f"shape must be 2D, got {shape}")
    if any(s <= 0 for s in shape):
        raise ValueError(f"invalid shape: {shape}")
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    spectrum = str(spectrum).lower().strip()
    if spectrum not in {"powerlaw", "bbks", "bbks_tilt"}:
        raise ValueError(f"unsupported spectrum={spectrum}")
    if lognormal_sigma <= 0:
        raise ValueError("lognormal_sigma must be > 0")
    if bbks_k_eps <= 0:
        raise ValueError("bbks_k_eps must be > 0")

    nx, ny = shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx)[:, None]
    ky = 2.0 * np.pi * np.fft.rfftfreq(ny)[None, :]
    k2 = kx * kx + ky * ky
    k = np.sqrt(k2, dtype=np.float64)
    scale = np.zeros_like(k, dtype=np.float64)
    nonzero = k > 0
    if spectrum == "powerlaw":
        scale[nonzero] = k[nonzero] ** (-alpha / 2.0)
    else:
        if bbks_k0 is None:
            bbks_k0 = 0.15 * np.pi
        if bbks_k0 <= 0:
            raise ValueError("bbks_k0 must be > 0")
        q = np.zeros_like(k, dtype=np.float64)
        q[nonzero] = k[nonzero] / float(bbks_k0)
        x = 2.34 * q
        with np.errstate(divide="ignore", invalid="ignore"):
            t0 = np.log1p(x) / np.where(x > 0, x, 1.0)
        denom = 1.0 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4
        t = t0 * np.power(denom, -0.25)
        p = np.zeros_like(k, dtype=np.float64)
        p[nonzero] = (k[nonzero] ** float(bbks_ns)) * (t[nonzero] ** 2)
        if spectrum == "bbks_tilt":
            p[nonzero] = p[nonzero] * (k[nonzero] + float(bbks_k_eps)) ** (-float(bbks_beta))
        scale[nonzero] = np.sqrt(p[nonzero])

    real = rng.normal(size=(nx, ny // 2 + 1))
    imag = rng.normal(size=(nx, ny // 2 + 1))
    coeff = (real + 1j * imag) * scale
    coeff[0, 0] = 0.0 + 0.0j

    field = np.fft.irfftn(coeff, s=shape).real
    assert_finite("rho_raw_2d", field)
    if lognormal:
        field = np.exp(float(lognormal_sigma) * field)
        assert_finite("rho_lognormal_2d", field)
    rho = normalize_rho(field, mode=norm_mode)
    return rho


def solve_poisson_periodic_fft_2d(rho: np.ndarray) -> PoissonResult2D:
    """
    Periodic 2D Poisson via FFT, with mean subtraction of rho.
    phi_k = -rho_k/k^2 (k=0 -> 0), gradients via spectral derivatives.
    """
    rho = np.asarray(rho, dtype=np.float64)
    if rho.ndim != 2:
        raise ValueError(f"rho must be 2D, got {rho.shape}")
    assert_finite("rho", rho)

    nx, ny = rho.shape
    rho0 = rho - float(rho.mean())
    rho_k = np.fft.fftn(rho0)

    kx = 2.0 * np.pi * np.fft.fftfreq(nx)[:, None]
    ky = 2.0 * np.pi * np.fft.fftfreq(ny)[None, :]
    k2 = kx * kx + ky * ky

    phi_k = np.zeros_like(rho_k)
    nonzero = k2 > 0
    phi_k[nonzero] = -rho_k[nonzero] / k2[nonzero]
    phi_k[0, 0] = 0.0 + 0.0j

    phi = np.fft.ifftn(phi_k).real
    gx = np.fft.ifftn(1j * kx * phi_k).real
    gy = np.fft.ifftn(1j * ky * phi_k).real
    assert_finite("phi", phi)
    assert_finite("gx", gx)
    assert_finite("gy", gy)
    gmag = np.sqrt(gx * gx + gy * gy, dtype=np.float64)
    assert_finite("gmag", gmag)
    return PoissonResult2D(rho=rho, phi=phi, gx=gx, gy=gy, gmag=gmag)


def band_split_poisson_2d(
    rho01: np.ndarray,
    *,
    k0_frac: float = 0.15,
    validate_range: bool = True,
) -> BandSplit2D:
    """
    Split rho into low/high-k components in Fourier space, solve Poisson for each,
    and verify gx ~= gx_low + gx_high (same for gy).
    """
    rho01 = np.asarray(rho01, dtype=np.float64)
    if rho01.ndim != 2:
        raise ValueError("rho01 must be 2D")
    if validate_range:
        if (rho01.min() < -1e-6) or (rho01.max() > 1.0 + 1e-6):
            raise ValueError("rho01 must be in [0,1]")
    if not (0.0 < float(k0_frac) < 0.5):
        raise ValueError("k0_frac must be in (0,0.5)")

    nx, ny = rho01.shape
    rho_k = np.fft.fftn(rho01 - float(rho01.mean()))
    kx = 2.0 * np.pi * np.fft.fftfreq(nx)[:, None]
    ky = 2.0 * np.pi * np.fft.fftfreq(ny)[None, :]
    k = np.sqrt(kx * kx + ky * ky, dtype=np.float64)
    k_ny = np.pi  # in our 2*pi*fftfreq convention with unit spacing
    k0 = float(k0_frac) * k_ny
    mask_low = k <= k0

    rho_low_k = rho_k * mask_low
    rho_high_k = rho_k * (~mask_low)
    rho_low = np.fft.ifftn(rho_low_k).real
    rho_high = np.fft.ifftn(rho_high_k).real
    assert_finite("rho_low", rho_low)
    assert_finite("rho_high", rho_high)

    low = solve_poisson_periodic_fft_2d(rho_low)
    high = solve_poisson_periodic_fft_2d(rho_high)
    full = solve_poisson_periodic_fft_2d(rho_low + rho_high)

    denom_gx = float(np.mean(np.abs(full.gx))) + 1e-12
    denom_gy = float(np.mean(np.abs(full.gy))) + 1e-12
    rel_err_gx = float(np.mean(np.abs(full.gx - (low.gx + high.gx))) / denom_gx)
    rel_err_gy = float(np.mean(np.abs(full.gy - (low.gy + high.gy))) / denom_gy)
    return BandSplit2D(
        rho=rho01,
        rho_low=rho_low,
        rho_high=rho_high,
        low=low,
        high=high,
        full=full,
        rel_err_gx=rel_err_gx,
        rel_err_gy=rel_err_gy,
    )

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
    if alpha < 0:
        raise ValueError("alpha must be >= 0")

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
