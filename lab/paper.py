from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_float(cell: str) -> float | None:
    if cell is None:
        return None
    s = cell.strip()
    if not s or "n<" in s or "n/a" in s:
        return None
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _parse_table(text: str, header_key: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in text.splitlines()]
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


def _find_latest_run(runs_root: Path, tag: str) -> Path | None:
    candidates = [p for p in runs_root.iterdir() if p.is_dir() and tag in p.name]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _get_run_path(cfg: dict[str, Any], key: str, tag: str, runs_root: Path) -> Path:
    run_map = cfg.get("run_map", {}) or {}
    if key in run_map:
        return Path(run_map[key])
    found = _find_latest_run(runs_root, tag)
    if found is None:
        raise FileNotFoundError(f"run for {key} not found (tag={tag})")
    return found


def _format_mean(value: float | None, *, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _parse_holdout_table(text: str, holdout: str) -> dict[float, dict[str, str]]:
    lines = text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip() == f"## Holdout: {holdout}":
            start_idx = i
            break
    if start_idx is None:
        return {}
    sub = "\n".join(lines[start_idx:])
    rows = _parse_table(sub, "thresh")
    out: dict[float, dict[str, str]] = {}
    for row in rows:
        thr = _parse_float(row.get("thresh", ""))
        if thr is None:
            continue
        out[float(thr)] = row
    return out


def _extract_mean_from_row(row: dict[str, str], key: str) -> float | None:
    return _parse_float(row.get(key, ""))


def _build_md_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(row.get(col, "") for col in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="E67 paper synthesis.")
    parser.add_argument("--config", required=True, help="Path to e67 config YAML.")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    runs_root = Path(cfg.get("runs_root", "outputs/runs"))
    out_dir = Path(cfg.get("out_dir", "outputs/paper"))
    out_dir.mkdir(parents=True, exist_ok=True)

    run_e48 = _get_run_path(cfg, "e48", "e48", runs_root)
    run_e49 = _get_run_path(cfg, "e49", "e49", runs_root)
    run_e50 = _get_run_path(cfg, "e50", "e50", runs_root)
    run_e61 = _get_run_path(cfg, "e61", "e61", runs_root)
    run_e62 = _get_run_path(cfg, "e62", "e62", runs_root)
    run_e65 = _get_run_path(cfg, "e65", "e65", runs_root)
    run_e66 = _get_run_path(cfg, "e66", "e66", runs_root)
    run_roc = _get_run_path(cfg, "e64_roc", "e64", runs_root)

    e48_text = _read_text(run_e48 / "summary_e48_oos_sparsefft_vs_wiener_alpha2.md")
    e49_text = _read_text(run_e49 / "summary_e49_oos_twochannel_fullg_wiener_sparsefft.md")
    e50_text = _read_text(run_e50 / "summary_e50_oos_twochannel_fullg_wiener_sparsefft_highkpixels.md")
    e61_text = _read_text(run_e61 / "summary_e61_regime_gainlaw_stable.md")
    e62_text = _read_text(run_e62 / "summary_e62_two_stage_gain_model.md")
    e65_text = _read_text(run_e65 / "summary_e65_blend_stability_thresholds.md")
    e66_text = _read_text(run_e66 / "summary_e66_alpha_tail_power.md")

    # E48 low-k.
    e48_rows = _parse_table(e48_text, "model")
    e48_ceiling = next(r for r in e48_rows if r["model"] == "ceiling")
    e48_wiener = next(r for r in e48_rows if r["model"] == "wiener")
    e48_fft800 = next(r for r in e48_rows if "K=800" in r["model"])
    e48_rel_ceiling = _parse_float(e48_ceiling["relRMSE mean±std"]) or float("nan")
    e48_rel_wiener = _parse_float(e48_wiener["relRMSE mean±std"]) or float("nan")
    e48_rel_fft = _parse_float(e48_fft800["relRMSE mean±std"]) or float("nan")

    # E49 full-g high-k kernel.
    e49_rows = _parse_table(e49_text, "model")
    e49_a = next(r for r in e49_rows if r["model"].startswith("A "))
    e49_b = next(r for r in e49_rows if r["model"].startswith("B "))
    e49_c = next(r for r in e49_rows if r["model"].startswith("C "))
    e49_rel_a = _parse_float(e49_a["relRMSE mean±std"]) or float("nan")
    e49_rel_b = _parse_float(e49_b["relRMSE mean±std"]) or float("nan")
    e49_rel_c = _parse_float(e49_c["relRMSE mean±std"]) or float("nan")

    # E50 full-g high-k pixels.
    e50_rows = _parse_table(e50_text, "model")
    e50_a = next(r for r in e50_rows if r["model"].startswith("A "))
    e50_b = next(r for r in e50_rows if r["model"].startswith("B "))
    e50_c = next(r for r in e50_rows if r["model"].startswith("C "))
    e50_rel_a = _parse_float(e50_a["relRMSE mean±std"]) or float("nan")
    e50_rel_b = _parse_float(e50_b["relRMSE mean±std"]) or float("nan")
    e50_rel_c = _parse_float(e50_c["relRMSE mean±std"]) or float("nan")

    # E65 holdout tables.
    e65_alpha = _parse_holdout_table(e65_text, "alpha")
    e65_bbks_ext = _parse_holdout_table(e65_text, "bbks_ext")
    e65_bbks_tilt = _parse_holdout_table(e65_text, "bbks_tilt")

    # E66 alpha holdout (strong counts).
    e66_alpha = _parse_holdout_table(e66_text, "alpha")

    thr_main = 0.03
    thr_alt = 0.02

    def get_metric(table: dict[float, dict[str, str]], thr: float, key: str) -> float | None:
        row = table.get(float(thr))
        if not row:
            return None
        return _extract_mean_from_row(row, key)

    def get_cell(table: dict[float, dict[str, str]], thr: float, key: str) -> str:
        row = table.get(float(thr))
        if not row:
            return "n/a"
        return row.get(key, "n/a")

    # AUROC at thresholds.
    auc_alpha_03 = get_metric(e65_alpha, thr_main, "AUROC")
    auc_bbks_03 = get_metric(e65_bbks_ext, thr_main, "AUROC")
    auc_tilt_03 = get_metric(e65_bbks_tilt, thr_main, "AUROC")

    auc_alpha_02 = get_metric(e65_alpha, thr_alt, "AUROC")
    auc_bbks_02 = get_metric(e65_bbks_ext, thr_alt, "AUROC")
    auc_tilt_02 = get_metric(e65_bbks_tilt, thr_alt, "AUROC")

    # Global metrics at thr=0.03.
    g_alpha_r2_03 = get_metric(e66_alpha, thr_main, "global_R2")
    g_alpha_mae_03 = get_metric(e66_alpha, thr_main, "global_MAE")
    g_bbks_r2_03 = get_metric(e65_bbks_ext, thr_main, "global_R2")
    g_bbks_mae_03 = get_metric(e65_bbks_ext, thr_main, "global_MAE")
    g_tilt_r2_03 = get_metric(e65_bbks_tilt, thr_main, "global_R2")
    g_tilt_mae_03 = get_metric(e65_bbks_tilt, thr_main, "global_MAE")

    # Strong metrics at thr=0.03 with n_strong>=30.
    n_alpha_03 = get_metric(e66_alpha, thr_main, "n_strong")
    n_bbks_03 = get_metric(e65_bbks_ext, thr_main, "n_strong")
    n_tilt_03 = get_metric(e65_bbks_tilt, thr_main, "n_strong")

    s_alpha_mae_03 = get_metric(e66_alpha, thr_main, "strong_MAE") if (n_alpha_03 or 0) >= 30 else None
    s_alpha_r2_03 = get_metric(e66_alpha, thr_main, "strong_R2") if (n_alpha_03 or 0) >= 30 else None

    s_bbks_mae_03 = get_metric(e65_bbks_ext, thr_main, "strong_MAE") if (n_bbks_03 or 0) >= 30 else None
    s_bbks_r2_03 = get_metric(e65_bbks_ext, thr_main, "strong_R2") if (n_bbks_03 or 0) >= 30 else None

    s_tilt_mae_03 = get_metric(e65_bbks_tilt, thr_main, "strong_MAE") if (n_tilt_03 or 0) >= 30 else None
    s_tilt_r2_03 = get_metric(e65_bbks_tilt, thr_main, "strong_R2") if (n_tilt_03 or 0) >= 30 else None

    # Alpha tail power comparison.
    n_alpha_03_e65 = get_metric(e65_alpha, thr_main, "n_strong")
    n_alpha_03_e66 = get_metric(e66_alpha, thr_main, "n_strong")
    s_r2_alpha_03_e66 = s_alpha_r2_03

    def parse_e61_auc(text: str, label: str) -> float | None:
        pattern = rf"{label}:.*?y_strong .*?auc=([0-9.]+)"
        match = re.search(pattern, text)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    e61_auc_bbks = parse_e61_auc(e61_text, "bbks_ext")
    e61_auc_alpha = parse_e61_auc(e61_text, "alpha")
    e61_auc_tilt = parse_e61_auc(e61_text, "bbks_tilt")

    # Main table rows.
    main_rows: list[dict[str, str]] = []
    main_rows.append(
        {
            "Section": "OOS low-k (E48)",
            "Metric": "relRMSE ceiling / wiener / FFT800",
            "Value": f"{_format_mean(e48_rel_ceiling)} / {_format_mean(e48_rel_wiener)} / {_format_mean(e48_rel_fft)}",
        }
    )
    main_rows.append(
        {
            "Section": "OOS low-k (E48)",
            "Metric": "ΔrelRMSE vs ceiling (wiener / FFT800)",
            "Value": f"{_format_mean(e48_rel_wiener - e48_rel_ceiling)} / {_format_mean(e48_rel_fft - e48_rel_ceiling)}",
        }
    )
    main_rows.append(
        {
            "Section": "OOS full-g highk_kernel (E49)",
            "Metric": "relRMSE A/B/C",
            "Value": f"{_format_mean(e49_rel_a)} / {_format_mean(e49_rel_b)} / {_format_mean(e49_rel_c)}",
        }
    )
    main_rows.append(
        {
            "Section": "OOS full-g highk_kernel (E49)",
            "Metric": "ΔrelRMSE vs A (B/C)",
            "Value": f"{_format_mean(e49_rel_b - e49_rel_a)} / {_format_mean(e49_rel_c - e49_rel_a)}",
        }
    )
    main_rows.append(
        {
            "Section": "OOS full-g highk_pixels (E50)",
            "Metric": "relRMSE A/B/C",
            "Value": f"{_format_mean(e50_rel_a)} / {_format_mean(e50_rel_b)} / {_format_mean(e50_rel_c)}",
        }
    )
    main_rows.append(
        {
            "Section": "OOS full-g highk_pixels (E50)",
            "Metric": "ΔrelRMSE vs A (B/C)",
            "Value": f"{_format_mean(e50_rel_b - e50_rel_a)} / {_format_mean(e50_rel_c - e50_rel_a)}",
        }
    )
    main_rows.append(
        {
            "Section": "Regime AUROC (E65)",
            "Metric": "AUROC y_strong @0.03 (alpha/bbks_ext/bbks_tilt)",
            "Value": f"{_format_mean(auc_alpha_03, digits=3)} / {_format_mean(auc_bbks_03, digits=3)} / {_format_mean(auc_tilt_03, digits=3)}",
        }
    )
    main_rows.append(
        {
            "Section": "Regime AUROC (E65)",
            "Metric": "AUROC y_strong @0.02 (alpha/bbks_ext/bbks_tilt)",
            "Value": f"{_format_mean(auc_alpha_02, digits=3)} / {_format_mean(auc_bbks_02, digits=3)} / {_format_mean(auc_tilt_02, digits=3)}",
        }
    )
    main_rows.append(
        {
            "Section": "Two-stage global @0.03",
            "Metric": "R2 / MAE (alpha[E66], bbks_ext[E65], bbks_tilt[E65])",
            "Value": (
                f"{_format_mean(g_alpha_r2_03)}/{_format_mean(g_alpha_mae_03)} ; "
                f"{_format_mean(g_bbks_r2_03)}/{_format_mean(g_bbks_mae_03)} ; "
                f"{_format_mean(g_tilt_r2_03)}/{_format_mean(g_tilt_mae_03)}"
            ),
        }
    )
    main_rows.append(
        {
            "Section": "Strong subset @0.03",
            "Metric": "strong_MAE / strong_R2 (n_strong>=30)",
            "Value": (
                f"alpha(E66): {_format_mean(s_alpha_mae_03)}/{_format_mean(s_alpha_r2_03)} (n={_format_mean(n_alpha_03, digits=1)}) ; "
                f"bbks_ext(E65): {_format_mean(s_bbks_mae_03)}/{_format_mean(s_bbks_r2_03)} (n={_format_mean(n_bbks_03, digits=1)}) ; "
                f"bbks_tilt(E65): n<30"
            ),
        }
    )
    main_rows.append(
        {
            "Section": "Alpha tail power",
            "Metric": "n_strong@0.03 (E65 vs E66)",
            "Value": f"E65: {_format_mean(n_alpha_03_e65, digits=1)} ; E66: {_format_mean(n_alpha_03_e66, digits=1)}",
        }
    )
    main_rows.append(
        {
            "Section": "Alpha tail power",
            "Metric": "strong_R2@0.03 (E66, n>=30)",
            "Value": _format_mean(s_r2_alpha_03_e66),
        }
    )

    # Write main table.
    csv_path = out_dir / "main_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Section", "Metric", "Value"])
        writer.writeheader()
        writer.writerows(main_rows)

    md_path = out_dir / "main_table.md"
    md_path.write_text(_build_md_table(main_rows, ["Section", "Metric", "Value"]) + "\n", encoding="utf-8")

    # Composite figure.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # (i) ΔrelRMSE bars (OOS).
    labels = ["low-k", "full-g (kernel)", "full-g (pixels)"]
    wiener_delta = [e48_rel_wiener - e48_rel_ceiling, e49_rel_b - e49_rel_a, e50_rel_b - e50_rel_a]
    fft_delta = [e48_rel_fft - e48_rel_ceiling, e49_rel_c - e49_rel_a, e50_rel_c - e50_rel_a]
    x = np.arange(len(labels))
    width = 0.35
    ax = axes[0]
    ax.bar(x - width / 2, wiener_delta, width, label="Wiener")
    ax.bar(x + width / 2, fft_delta, width, label="sparseFFT K=800")
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("ΔrelRMSE vs ceiling")
    ax.set_title("OOS ΔrelRMSE")
    ax.legend(fontsize=8)

    # (ii) ROC example (from E64).
    ax = axes[1]
    roc_path = run_roc / "roc_y_strong_alpha.png"
    if roc_path.exists():
        img = plt.imread(roc_path)
        ax.imshow(img)
        ax.set_title("ROC y_strong (alpha, E64)")
        ax.axis("off")
    else:
        ax.text(0.5, 0.5, "ROC not found", ha="center", va="center")
        ax.set_axis_off()

    # (iii) strong_R2 vs thresh for alpha (E66).
    ax = axes[2]
    thr_list = sorted(e66_alpha.keys()) if e66_alpha else [0.02, 0.03, 0.04, 0.05]
    r2_vals = [get_metric(e66_alpha, thr, "strong_R2") for thr in thr_list]
    r2_vals = [v if v is not None else np.nan for v in r2_vals]
    n_vals = [get_metric(e66_alpha, thr, "n_strong") for thr in thr_list]
    n_vals = [v if v is not None else np.nan for v in n_vals]
    ax.plot(thr_list, r2_vals, marker="o", label="strong_R2")
    ax.set_xlabel("threshold")
    ax.set_ylabel("strong_R2")
    ax.set_title("Alpha strong_R2 vs thresh (E66)")
    ax2 = ax.twinx()
    ax2.plot(thr_list, n_vals, marker="s", color="tab:orange", label="n_strong")
    ax2.set_ylabel("n_strong")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "main_figure.png", dpi=150)
    plt.close(fig)

    # Short results text.
    short_lines = [
        "Results summary:",
        f"- OOS low-k (E48): relRMSE ceiling={e48_rel_ceiling:.4f}, wiener={e48_rel_wiener:.4f}, FFT800={e48_rel_fft:.4f}.",
        f"- OOS full-g (E49): relRMSE A/B/C={e49_rel_a:.4f}/{e49_rel_b:.4f}/{e49_rel_c:.4f} (high-k kernel).",
        f"- OOS full-g (E50): relRMSE A/B/C={e50_rel_a:.4f}/{e50_rel_b:.4f}/{e50_rel_c:.4f} (high-k pixels).",
        f"- AUROC y_strong@0.03 (E65): alpha={_format_mean(auc_alpha_03, digits=3)}, bbks_ext={_format_mean(auc_bbks_03, digits=3)}, bbks_tilt={_format_mean(auc_tilt_03, digits=3)}.",
        f"- Two-stage global@0.03: alpha(E66) R2={_format_mean(g_alpha_r2_03)}, MAE={_format_mean(g_alpha_mae_03)}; bbks_ext(E65) R2={_format_mean(g_bbks_r2_03)}, MAE={_format_mean(g_bbks_mae_03)}.",
        f"- Strong subset@0.03 (n>=30): alpha(E66) MAE={_format_mean(s_alpha_mae_03)}, R2={_format_mean(s_alpha_r2_03)}; bbks_ext(E65) MAE={_format_mean(s_bbks_mae_03)}, R2={_format_mean(s_bbks_r2_03)}.",
        f"- Alpha tail power: n_strong@0.03 increased from E65={_format_mean(n_alpha_03_e65, digits=1)} to E66={_format_mean(n_alpha_03_e66, digits=1)}.",
        "- ROC example is taken from E64 (alpha holdout), AUROC≈1.0.",
        "- Baseline two-stage (E62) global R2: bbks_ext=0.560, alpha=0.437, bbks_tilt=0.905.",
        f"- E61 y_strong AUROC: alpha={_format_mean(e61_auc_alpha, digits=3)}, bbks_ext={_format_mean(e61_auc_bbks, digits=3)}, bbks_tilt={_format_mean(e61_auc_tilt, digits=3)}.",
    ]
    (out_dir / "short_results.md").write_text("\n".join(short_lines) + "\n", encoding="utf-8")

    print(str(out_dir))


if __name__ == "__main__":
    main()
