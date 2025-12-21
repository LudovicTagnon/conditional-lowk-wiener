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
    main_table = (
        main_table.replace("wiener", "Wiener")
        .replace("highk_kernel", "high-k kernel")
        .replace("highk_pixels", "high-k pixels")
    )
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

    line1 = _extract_line(short_lines, "1)").replace("wiener", "Wiener")
    line2 = _extract_line(short_lines, "2)").replace("wiener", "Wiener")
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
        "We test whether conditional predictors improve gravitational-field estimation beyond a truncated-kernel "
        "ceiling in controlled synthetic data.\n"
        "Ceiling = a finite-support, truncated-kernel estimator (no conditional inference outside the window).\n"
        "Wiener = conditional linear prediction of g_low at the center from rho in a window, using train "
        "covariances.\n"
        "Mechanism: inside contains information about outside when the field is correlated; for alpha≈0 "
        "(white-ish), correlations vanish and gains disappear.\n"
        "OOS protocol: all metrics use independent train/test fields (relRMSE and Pearson).\n"
        f"Key OOS deltas: {line1} {line2}\n"
        f"Regime/magnitude summary: {line3}\n"
        f"Compressibility: sparseFFT with K≈800 gives ΔrelRMSE vs Wiener≈{delta_w} while beating ceiling by {delta_c}."
    )

    intro = (
        "Predicting the gravitational field from local density information blends physics (Poisson/FFT) with "
        "statistical learning. We compare a ceiling baseline (truncated kernel with finite support, no conditional "
        "inference beyond the window) to a Wiener predictor (conditional linear estimate from rho in a window). "
        "When rho is correlated, inside samples carry information about outside; this breaks down for alpha≈0, "
        "where correlations are near zero. Our goal is to quantify OOS gains, characterize when they occur, "
        "and connect them to a regime→magnitude model that predicts when gains appear.\n\n"
        "Contributions:\n"
        "1) OOS benchmarks for low-k and full-g using ceiling, Wiener, and sparseFFT.\n"
        "2) A regime→magnitude model with AUROC and strong-tail metrics under a defined threshold policy.\n"
        "3) A paper bundle and draft generator with sensitivity tables and reproducible artifacts.\n"
    )

    setup = (
        "We define three estimators for the low-k component of the field: (i) a truncated-kernel ceiling that "
        "applies the impulse-response kernel limited to a finite support, (ii) a Wiener estimator that solves a "
        "conditional linear prediction using the empirical covariance of the density field, and (iii) a sparse-FFT "
        "approximation that keeps the top-K Fourier modes of the Wiener-symmetrized kernel. We evaluate a two-channel "
        "decomposition in which g_full = g_low + g_high, with g_low predicted by ceiling/Wiener/sparseFFT and g_high "
        "predicted by a fixed local block (kernel or pixels). This isolates how much the low-k predictor contributes "
        "to full-g accuracy.\n\n"
        "Mini-box: What is Wiener here?\n"
        "It is the linear predictor of g_low at the patch center from rho in a finite window, estimated from "
        "training covariances. If correlations vanish (alpha≈0), the predictor collapses to the ceiling kernel.\n\n"
        "Compute: the ceiling convolution is O(w^2) per patch (or FFT for full fields), the Wiener solve uses a "
        "covariance-driven linear system, and the sparse-FFT variant reduces the representation to K modes with "
        "explicit truncation. All models use training-only information; test fields are untouched by fitting."
    )

    protocol = (
        "We use strict out-of-sample (OOS) splits with independent train/test fields. LOFO splits are used for "
        "regime modeling, and metrics include Pearson correlation and relRMSE (RMSE / std(y_test)). For two-stage "
        "models, we define a strong-tail threshold and report strong-subset metrics only when n_strong is "
        "sufficiently large. E62 is a single-seed dev baseline; E65/E66 are multi-seed runs that support the main "
        "claims and threshold policy. No-leakage is enforced by fitting on training fields only and evaluating "
        "exclusively on held-out fields. See method schematic (outputs/paper/method_schematic.png)."
    )

    results = (
        "Main results:\n\n"
        f"{main_table}\n\n"
        f"Compact synthesis (OOS): {line1} {line2} {line3}\n"
        f"Threshold policy: main thresh={main_thresh}; sensitivity reported for {sens}. "
        "Multi-seed results (E65/E66) support the main claims; E62 is a single-seed dev baseline.\n"
        "See outputs/paper/main_figure.png for deltas and ROC, and outputs/paper/sparsefft_table.md for the "
        "sparseFFT appendix."
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

    acknowledgments = (
        "Acknowledgment:\n\n"
        "Assistance from OpenAI ChatGPT (GPT-5.2 Thinking) and ChatGPT Codex (OpenAI) supported experimental "
        "orchestration, writing, and revision."
    )

    limits = (
        "Limitations & scope:\n\n"
        "All estimators are linear and tied to FFT/Poisson physics; nonlinear or nonstationary effects are not "
        "modeled.\n"
        "Results are on synthetic families (alpha, BBKS, BBKS-tilt) with controlled spectra; real data may violate "
        "these assumptions.\n"
        "The Wiener advantage depends on correlation structure, window choice, and covariance estimation; weakly "
        "correlated fields show little gain.\n"
        "Threshold choice affects strong-tail counts, so strong-subset metrics are only reported when n_strong>=30.\n"
        "SparseFFT compressibility trades bias/variance with K and may require recalibration across datasets."
    )

    repro = (
        "Reproducibility & Compute:\n\n"
        "This draft is generated from the paper bundle in outputs/paper/ and does not run new experiments. "
        "The synthesis scripts are lab/paper.py and lab/draft.py, with run paths specified in "
        "configs/e68_threshold_policy.yaml. To rebuild the bundle and draft, run scripts/reproduce_paper.sh. "
        "That script checks for required artifacts (tables, figures, schematic, sparseFFT appendix) and fails if "
        "any are missing. No new runs are executed for E70/E71; these steps only aggregate existing summaries. "
        "Runtime is dominated by file I/O and plotting, and the stored run directories encode all numeric values."
    )

    draft = "\n\n".join(
        [
            "# Conditional low-k prediction beyond a truncated-kernel ceiling",
            "Authors: Ludovic Tagnon",
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
            "## Acknowledgment",
            acknowledgments,
            "## 6 Limitations & scope",
            limits,
            "## 7 Reproducibility & Compute",
            repro,
        ]
    )

    out_path.write_text(draft + "\n", encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
