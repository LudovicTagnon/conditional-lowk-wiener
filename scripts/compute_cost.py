#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path
import numpy as np


def timeit(fn, repeat: int) -> float:
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / repeat


def main() -> None:
    rng = np.random.RandomState(0)

    w = 193
    k = 800
    patch = rng.normal(size=(w, w))
    kernel = rng.normal(size=(w, w))

    def ceiling_op():
        return float(np.sum(patch * kernel))

    def sparsefft_op():
        freq = np.fft.fft2(patch)
        mag = np.abs(freq).ravel()
        idx = np.argpartition(mag, -k)[-k:]
        return float(np.sum(mag[idx]))

    d = 256
    n = 512
    X = rng.normal(size=(n, d))
    y = rng.normal(size=(n,))
    lam = 1e-3

    def wiener_op():
        xtx = X.T @ X
        xtx.flat[:: d + 1] += lam
        xty = X.T @ y
        _ = np.linalg.solve(xtx, xty)

    ceiling_ms = timeit(ceiling_op, repeat=200)
    sparse_ms = timeit(sparsefft_op, repeat=30)
    wiener_ms = timeit(wiener_op, repeat=10)

    out_path = Path("outputs/paper/compute_cost.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Compute snapshot (micro-benchmark)",
        "",
        "- note: synthetic arrays, single patch, not full training; no new fields generated",
        "",
        "| operation | size | repeats | avg_ms |",
        "| --- | --- | --- | --- |",
        f"| local ceiling (dot) | w={w} | 200 | {ceiling_ms:.4f} |",
        f"| sparseFFT (fft + topK) | w={w}, K={k} | 30 | {sparse_ms:.4f} |",
        f"| Wiener solve (linear system) | n={n}, d={d} | 10 | {wiener_ms:.4f} |",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
