from __future__ import annotations

import argparse
from pathlib import Path
import re


def _read_lines(path: Path) -> list[str]:
    return [line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()]


def _extract_line(lines: list[str], prefix: str) -> str:
    for line in lines:
        if line.strip().startswith(prefix):
            return line.strip()
    return ""


def _parse_threshold_policy(lines: list[str]) -> tuple[str, str]:
    line = _extract_line(lines, "4)")
    if not line:
        return "", ""
    main = ""
    sens = ""
    match = re.search(r"main thresh=([0-9.]+)", line)
    if match:
        main = match.group(1)
    match = re.search(r"sensitivity reported for (.+)$", line)
    if match:
        sens = match.group(1).strip().rstrip(".")
    return main, sens


def _parse_table(lines: list[str], header_key: str) -> list[dict[str, str]]:
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("|") and header_key in line:
            header_idx = i
            break
    if header_idx is None:
        return []
    header = [c.strip() for c in lines[header_idx].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for j in range(header_idx + 2, len(lines)):
        line = lines[j]
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != len(header):
            continue
        rows.append(dict(zip(header, cells)))
    return rows


def _extract_float(cell: str) -> str:
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cell or "")
    return match.group(0) if match else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="E69 paper draft generator.")
    parser.add_argument("--paper-dir", default="outputs/paper", help="Paper bundle directory.")
    parser.add_argument("--out", default=None, help="Output draft path.")
    args = parser.parse_args()

    paper_dir = Path(args.paper_dir)
    out_path = Path(args.out) if args.out else paper_dir / "draft.md"

    main_table_path = paper_dir / "main_table.md"
    short_results_path = paper_dir / "short_results.md"

    if not main_table_path.exists():
        raise FileNotFoundError(f"Missing main table: {main_table_path}")
    if not short_results_path.exists():
        raise FileNotFoundError(f"Missing short results: {short_results_path}")

    main_table = main_table_path.read_text(encoding="utf-8").strip()
    short_lines = _read_lines(short_results_path)

    sparse_table_path = paper_dir / "sparsefft_table.md"
    sparse_stability_path = paper_dir / "sparsefft_stability.md"
    sparse_curve_path = paper_dir / "sparsefft_curve.png"

    sparse_para = ""
    if sparse_table_path.exists():
        sparse_lines = _read_lines(sparse_table_path)
        rows = _parse_table(sparse_lines, "model")
        k800 = next((r for r in rows if "K=800" in r.get("model", "")), {})
        delta_w = _extract_float(k800.get("ΔrelRMSE vs wiener", ""))
        delta_c = _extract_float(k800.get("ΔrelRMSE vs ceiling", ""))
        stability_note = ""
        if sparse_stability_path.exists():
            stability_lines = _read_lines(sparse_stability_path)
            stability_note = " " + " ".join(stability_lines[:2]).strip()
        sparse_para = (
            "Spectral compressibility of Wiener weights: "
            f"the sparse-FFT sweep (Fig. sparsefft_curve.png, Table sparsefft_table.md) shows K≈800 as a trade-off "
            f"(ΔrelRMSE vs Wiener≈{delta_w}, ΔrelRMSE vs ceiling≈{delta_c})."
            + stability_note
            + " No new experiments were run for this appendix."
        )

    line1 = _extract_line(short_lines, "1)")
    line2 = _extract_line(short_lines, "2)")
    line3 = _extract_line(short_lines, "3)")
    line5 = _extract_line(short_lines, "5)")
    line6 = _extract_line(short_lines, "6)")
    line7 = _extract_line(short_lines, "7)")
    line8 = _extract_line(short_lines, "8)")
    line9 = _extract_line(short_lines, "9)")
    line12 = _extract_line(short_lines, "12)")
    line13 = _extract_line(short_lines, "13)")
    main_thresh, sens = _parse_threshold_policy(short_lines)

    abstract = (
        "We study whether local topological signals and conditional predictors improve gravitational-field "
        "estimation beyond mass-only baselines in controlled synthetic experiments. We compare a truncated-kernel "
        "ceiling, Wiener conditioning, and sparse-FFT approximations, and evaluate a two-channel low-k/high-k "
        "decomposition with strict out-of-sample fields. "
        f"{line1} {line2} "
        f"{line3} "
        f"With the main threshold policy (thresh={main_thresh}), "
        f"{line6} "
        "We provide a threshold-sensitivity appendix and a regime→magnitude model summary, and we package all "
        "results for reproducible post-processing."
    )

    intro = (
        "Predicting the gravitational field from local density information blends physics (Poisson/FFT) with "
        "statistical learning. The ceiling estimator captures the truncated-kernel contribution, while the Wiener "
        "estimator exploits conditional correlations and can, in principle, improve over the ceiling when outside "
        "support is predictable. Our goal is to quantify this improvement, characterize when it holds out-of-sample, "
        "and connect it to a regime→magnitude model that predicts when gains occur.\n\n"
        "Contributions:\n"
        "- We provide OOS benchmarks for low-k and full-g prediction using ceiling, Wiener, and sparse-FFT.\n"
        "- We document a regime classifier with high AUROC across families and a two-stage magnitude model.\n"
        "- We introduce a threshold policy and sensitivity appendix to avoid unstable strong-tail metrics.\n"
        "- We supply a paper-ready bundle and a draft generator for reproducible reporting.\n"
    )

    setup = (
        "We define three estimators for the low-k component of the field: (i) a truncated-kernel ceiling that "
        "applies the impulse-response kernel limited to a finite support, (ii) a Wiener estimator that solves a "
        "conditional linear prediction using the empirical covariance of the density field, and (iii) a sparse-FFT "
        "approximation that keeps the top-K Fourier modes of the Wiener-symmetrized kernel. We evaluate a two-channel "
        "decomposition in which g_full = g_low + g_high, with g_low predicted by ceiling/Wiener/sparse-FFT and g_high "
        "predicted by a fixed local block (kernel or pixels). This isolates how much the low-k predictor contributes "
        "to full-g accuracy.\n\n"
        "Compute: the ceiling convolution is O(w^2) per patch (or FFT for full fields), the Wiener solve uses a "
        "covariance-driven linear system, and the sparse-FFT variant reduces the representation to K modes with "
        "explicit truncation. All models use training-only information; test fields are untouched by fitting."
    )

    protocol = (
        "We use strict out-of-sample (OOS) splits with independent train/test fields. LOFO splits are used for "
        "regime modeling, and metrics include Pearson correlation and relRMSE. For two-stage models, we define "
        "a strong-tail threshold and report strong-subset metrics only when n_strong is sufficiently large. "
        "No-leakage is enforced by fitting on training fields only and evaluating exclusively on held-out fields. "
        "See method schematic (outputs/paper/method_schematic.png)."
    )

    results = (
        "Main results:\n\n"
        f"{main_table}\n\n"
        "Key quantified results (from the paper bundle):\n"
        + "\n".join(short_lines)
        + "\n\n"
        f"Threshold policy: main thresh={main_thresh}; sensitivity reported for {sens}."
    )
    if sparse_para:
        results += "\n\n" + sparse_para

    related = (
        "Related Work:\n\n"
        "Wiener filtering provides the classical linear MMSE estimator for stationary Gaussian fields and is the "
        "conceptual basis for the conditional low-k predictor used here. Gaussian process regression and kriging "
        "formalize conditional expectations in spatial settings, while spectral representations of kernels and "
        "filters connect to FFT-based accelerations. Sparse spectral approximations and truncation in Fourier space "
        "are standard compression tactics for smooth operators, and windowing/tapering are common for controlling "
        "edge artifacts in finite-support kernels. Our approach combines these ideas in a controlled synthetic setup "
        "with strict OOS evaluation.\n\n"
        "References:\n"
        "- Wiener, N. (1949). Extrapolation, Interpolation, and Smoothing of Stationary Time Series.\n"
        "- Rasmussen, C.E. & Williams, C.K.I. (2006). Gaussian Processes for Machine Learning.\n"
        "- Cressie, N. (1993). Statistics for Spatial Data (Kriging).\n"
        "- Kay, S.M. (1993). Fundamentals of Statistical Signal Processing, Vol. I.\n"
        "- Brigham, E.O. (1988). The Fast Fourier Transform and Its Applications.\n"
        "- Oppenheim, A.V. & Schafer, R.W. (2009). Discrete-Time Signal Processing.\n"
        "- Percival, D.B. & Walden, A.T. (1993). Spectral Analysis for Physical Applications.\n"
        "- Mallat, S. (2008). A Wavelet Tour of Signal Processing.\n"
    )

    regime = (
        "Regime→magnitude model:\n\n"
        "We use a two-stage model in which a regime classifier predicts whether the gain is strong, followed by "
        "a magnitude regressor on the strong subset. Reported AUROC and magnitude metrics are summarized here:\n"
        f"- {line3}\n"
        f"- {line5}\n"
        f"- {line6}\n"
        f"- {line7}\n"
        f"- {line8}\n"
        f"- {line9}\n"
        f"- {line12}\n"
        f"- {line13}\n"
        "Strong-subset metrics are only considered when n_strong>=30 to avoid unstable estimates."
    )

    limits = (
        "Limitations & scope:\n\n"
        "The Wiener gain depends on the correlation structure of the density field; when correlations are weak or "
        "the strong tail is rare, gains can be small or noisy. Threshold choice affects strong-tail sample sizes; "
        "we therefore report a sensitivity appendix and enforce a minimum n_strong before interpreting strong_R2. "
        "Sparse-FFT approximations inherit the bias/variance trade-off in K and are evaluated in the OOS setting."
    )

    repro = (
        "Reproducibility:\n\n"
        "The paper bundle is in outputs/paper/ (main_table.md, main_figure.png, short_results.md). The synthesis "
        "script is lab/paper.py, and the threshold policy is configured in configs/e68_threshold_policy.yaml. "
        "All referenced run directories are listed in the config file."
    )

    draft = "\n\n".join(
        [
            "# Draft (E69)",
            "## Abstract",
            abstract,
            "## 1 Introduction",
            intro,
            "## 2 Setup & estimators",
            setup,
            "## 3 Experimental protocol",
            protocol,
            "## 4 Results",
            results,
            "## Related Work",
            related,
            "## 5 Regime→magnitude model",
            regime,
            "## 6 Limitations & scope",
            limits,
            "## 7 Reproducibility",
            repro,
        ]
    )

    out_path.write_text(draft + "\n", encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
