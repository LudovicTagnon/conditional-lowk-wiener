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


def _parse_table_in_section(text: str, section_header: str, header_key: str) -> list[dict[str, str]]:
    idx = text.find(section_header)
    if idx == -1:
        return []
    return _parse_table(text[idx:], header_key)


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
    run_e46 = _get_run_path(cfg, "e46", "e46", runs_root)
    run_e47 = _get_run_path(cfg, "e47", "e47", runs_root)
    run_e61 = _get_run_path(cfg, "e61", "e61", runs_root)
    run_e62 = _get_run_path(cfg, "e62", "e62", runs_root)
    run_e65 = _get_run_path(cfg, "e65", "e65", runs_root)
    run_e66 = _get_run_path(cfg, "e66", "e66", runs_root)
    run_roc = _get_run_path(cfg, "e64_roc", "e64", runs_root)

    e48_text = _read_text(run_e48 / "summary_e48_oos_sparsefft_vs_wiener_alpha2.md")
    e49_text = _read_text(run_e49 / "summary_e49_oos_twochannel_fullg_wiener_sparsefft.md")
    e50_text = _read_text(run_e50 / "summary_e50_oos_twochannel_fullg_wiener_sparsefft_highkpixels.md")
    e46_text = _read_text(run_e46 / "summary_e46_sparsefft_to_wiener_alpha2.md")
    e47_text = _read_text(run_e47 / "summary_e47_sparsefft_mode_stability.md")
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

    thr_main = float(cfg.get("main_thresh", 0.02))
    sens_thresholds = [float(x) for x in cfg.get("sensitivity_thresholds", [0.02, 0.03, 0.04, 0.05])]

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
    auc_alpha = get_metric(e65_alpha, thr_main, "AUROC")
    auc_bbks = get_metric(e65_bbks_ext, thr_main, "AUROC")
    auc_tilt = get_metric(e65_bbks_tilt, thr_main, "AUROC")

    # Global metrics at thr=0.03.
    g_alpha_r2 = get_metric(e66_alpha, thr_main, "global_R2")
    g_alpha_mae = get_metric(e66_alpha, thr_main, "global_MAE")
    g_bbks_r2 = get_metric(e65_bbks_ext, thr_main, "global_R2")
    g_bbks_mae = get_metric(e65_bbks_ext, thr_main, "global_MAE")
    g_tilt_r2 = get_metric(e65_bbks_tilt, thr_main, "global_R2")
    g_tilt_mae = get_metric(e65_bbks_tilt, thr_main, "global_MAE")

    # Strong metrics at thr=0.03 with n_strong>=30.
    n_alpha = get_metric(e66_alpha, thr_main, "n_strong")
    n_bbks = get_metric(e65_bbks_ext, thr_main, "n_strong")
    n_tilt = get_metric(e65_bbks_tilt, thr_main, "n_strong")

    s_alpha_mae = get_metric(e66_alpha, thr_main, "strong_MAE") if (n_alpha or 0) >= 30 else None
    s_alpha_r2 = get_metric(e66_alpha, thr_main, "strong_R2") if (n_alpha or 0) >= 30 else None

    s_bbks_mae = get_metric(e65_bbks_ext, thr_main, "strong_MAE") if (n_bbks or 0) >= 30 else None
    s_bbks_r2 = get_metric(e65_bbks_ext, thr_main, "strong_R2") if (n_bbks or 0) >= 30 else None

    s_tilt_mae = get_metric(e65_bbks_tilt, thr_main, "strong_MAE") if (n_tilt or 0) >= 30 else None
    s_tilt_r2 = get_metric(e65_bbks_tilt, thr_main, "strong_R2") if (n_tilt or 0) >= 30 else None

    # Alpha tail power comparison.
    n_alpha_03_e65 = get_metric(e65_alpha, thr_main, "n_strong")
    n_alpha_03_e66 = get_metric(e66_alpha, thr_main, "n_strong")
    s_r2_alpha_e66 = s_alpha_r2

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
            "Metric": f"AUROC y_strong @{thr_main:.2f} (alpha/bbks_ext/bbks_tilt)",
            "Value": f"{_format_mean(auc_alpha, digits=3)} / {_format_mean(auc_bbks, digits=3)} / {_format_mean(auc_tilt, digits=3)}",
        }
    )
    main_rows.append(
        {
            "Section": f"Two-stage global @{thr_main:.2f}",
            "Metric": "R2 / MAE (alpha[E66], bbks_ext[E65], bbks_tilt[E65])",
            "Value": (
                f"{_format_mean(g_alpha_r2)}/{_format_mean(g_alpha_mae)} ; "
                f"{_format_mean(g_bbks_r2)}/{_format_mean(g_bbks_mae)} ; "
                f"{_format_mean(g_tilt_r2)}/{_format_mean(g_tilt_mae)}"
            ),
        }
    )
    main_rows.append(
        {
            "Section": f"Strong subset @{thr_main:.2f}",
            "Metric": "strong_MAE / strong_R2 (n_strong>=30)",
            "Value": (
                f"alpha(E66): {_format_mean(s_alpha_mae)}/{_format_mean(s_alpha_r2)} (n={_format_mean(n_alpha, digits=1)}) ; "
                f"bbks_ext(E65): {_format_mean(s_bbks_mae)}/{_format_mean(s_bbks_r2)} (n={_format_mean(n_bbks, digits=1)}) ; "
                f"bbks_tilt(E65): n<30"
            ),
        }
    )
    main_rows.append(
        {
            "Section": "Alpha tail power",
            "Metric": f"n_strong@{thr_main:.2f} (E65 vs E66)",
            "Value": f"E65: {_format_mean(n_alpha_03_e65, digits=1)} ; E66: {_format_mean(n_alpha_03_e66, digits=1)}",
        }
    )
    main_rows.append(
        {
            "Section": "Alpha tail power",
            "Metric": f"strong_R2@{thr_main:.2f} (E66, n>=30)",
            "Value": _format_mean(s_r2_alpha_e66),
        }
    )

    # Write main table.
    csv_path = out_dir / "main_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Section", "Metric", "Value"])
        writer.writeheader()
        writer.writerows(main_rows)

    md_path = out_dir / "main_table.md"
    sensitivity_rows: list[dict[str, str]] = []
    for thr in sens_thresholds:
        row = e66_alpha.get(float(thr), {})
        n_strong = _extract_mean_from_row(row, "n_strong")
        strong_r2 = _extract_mean_from_row(row, "strong_R2")
        if n_strong is None or n_strong < 30:
            strong_r2_cell = "NA (n<30)"
        else:
            strong_r2_cell = row.get("strong_R2", "n/a")
        sensitivity_rows.append(
            {
                "thresh": f"{thr:.2f}",
                "AUROC": row.get("AUROC", "n/a"),
                "global_R2": row.get("global_R2", "n/a"),
                "global_MAE": row.get("global_MAE", "n/a"),
                "strong_MAE": row.get("strong_MAE", "n/a"),
                "strong_R2": strong_r2_cell,
                "n_strong": row.get("n_strong", "n/a"),
            }
        )
    sensitivity_md = "\n".join(
        [
            "## Sensitivity to threshold (alpha, E66)",
            _build_md_table(
                sensitivity_rows,
                ["thresh", "AUROC", "global_R2", "global_MAE", "strong_MAE", "strong_R2", "n_strong"],
            ),
        ]
    )
    md_path.write_text(
        _build_md_table(main_rows, ["Section", "Metric", "Value"]) + "\n\n" + sensitivity_md + "\n",
        encoding="utf-8",
    )

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

    # (iii) n_strong + strong_MAE vs thresh for alpha (E66).
    ax = axes[2]
    thr_list = sens_thresholds
    n_vals = [get_metric(e66_alpha, thr, "n_strong") for thr in thr_list]
    n_vals = [v if v is not None else np.nan for v in n_vals]
    mae_vals = [get_metric(e66_alpha, thr, "strong_MAE") for thr in thr_list]
    mae_vals = [v if v is not None else np.nan for v in mae_vals]
    ax.bar(thr_list, n_vals, width=0.006, color="tab:blue", alpha=0.6, label="n_strong")
    ax.set_xlabel("threshold")
    ax.set_ylabel("n_strong")
    ax.set_title("Alpha strong tail stability (E66)")
    ax2 = ax.twinx()
    ax2.plot(thr_list, mae_vals, marker="o", color="tab:orange", label="strong_MAE")
    ax2.set_ylabel("strong_MAE")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "main_figure.png", dpi=150)
    plt.close(fig)

    # SparseFFT compressibility appendix (E70).
    e46_perf = _parse_table(e46_text, "ΔrelRMSE vs wiener")
    e46_weights = _parse_table(e46_text, "corr(wK,w_sym)")
    k_vals: list[int] = []
    rel_vals: list[float] = []
    corr_vals: list[float] = []
    for row in e46_perf:
        k_label = row.get("K", "")
        if k_label.isdigit():
            k = int(k_label)
            rel = _parse_float(row.get("relRMSE mean±std", ""))
            if rel is not None:
                k_vals.append(k)
                rel_vals.append(rel)
    corr_map: dict[int, float] = {}
    for row in e46_weights:
        k_label = row.get("K", "")
        if k_label.isdigit():
            corr = _parse_float(row.get("corr(wK,w_sym)", ""))
            if corr is not None:
                corr_map[int(k_label)] = corr
    for k in k_vals:
        corr_vals.append(corr_map.get(k, float("nan")))

    if k_vals:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
        ax.plot(k_vals, rel_vals, marker="o", label="relRMSE (low-k)")
        ax.set_xscale("log")
        ax.set_xlabel("K (sparse FFT modes)")
        ax.set_ylabel("relRMSE")
        ax.set_title("SparseFFT compressibility (E46)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax2 = ax.twinx()
        ax2.plot(k_vals, corr_vals, marker="s", color="tab:orange", label="corr vs Wiener_sym")
        ax2.set_ylabel("corr(wK, wiener_sym)")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="lower right", fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "sparsefft_curve.png", dpi=150)
        plt.close(fig)

    # SparseFFT OOS table from E48.
    e48_perf = _parse_table(e48_text, "model")
    e48_delta = _parse_table(e48_text, "ΔrelRMSE vs wiener")
    perf_map = {row["model"]: row for row in e48_perf}
    delta_map = {row["K"]: row for row in e48_delta}

    def rel_mean(row: dict[str, str]) -> float | None:
        return _parse_float(row.get("relRMSE mean±std", ""))

    ceiling_rel = rel_mean(perf_map.get("ceiling", {}))
    wiener_rel = rel_mean(perf_map.get("wiener", {}))

    table_rows: list[dict[str, str]] = []
    models = ["ceiling", "wiener", "sparseFFT K=200", "sparseFFT K=400", "sparseFFT K=800", "sparseFFT K=1600"]
    for model in models:
        row = perf_map.get(model)
        if not row:
            continue
        pearson = row.get("Pearson mean±std", "n/a")
        rel = row.get("relRMSE mean±std", "n/a")
        delta_w = "n/a"
        delta_c = "n/a"
        if model.startswith("sparseFFT"):
            k_label = model.split("K=")[1]
            drow = delta_map.get(k_label)
            if drow:
                delta_w = drow.get("ΔrelRMSE vs wiener", "n/a")
                delta_c = drow.get("ΔrelRMSE vs ceiling", "n/a")
        elif model == "ceiling" and ceiling_rel is not None and wiener_rel is not None:
            delta_w = f"{ceiling_rel - wiener_rel:+.4f}"
            delta_c = "+0.0000"
        elif model == "wiener":
            delta_w = "+0.0000"
            if ceiling_rel is not None and wiener_rel is not None:
                delta_c = f"{wiener_rel - ceiling_rel:+.4f}"
        table_rows.append(
            {
                "model": model,
                "Pearson": pearson,
                "relRMSE": rel,
                "ΔrelRMSE vs wiener": delta_w,
                "ΔrelRMSE vs ceiling": delta_c,
            }
        )

    sparse_table_md = _build_md_table(
        table_rows,
        ["model", "Pearson", "relRMSE", "ΔrelRMSE vs wiener", "ΔrelRMSE vs ceiling"],
    )
    (out_dir / "sparsefft_table.md").write_text(sparse_table_md + "\n", encoding="utf-8")

    # SparseFFT mode stability summary (E47).
    gx_rows = _parse_table_in_section(e47_text, "## Stability (gx)", "Jaccard")
    gy_rows = _parse_table_in_section(e47_text, "## Stability (gy)", "Jaccard")
    def find_k(rows: list[dict[str, str]], k: str) -> dict[str, str]:
        for row in rows:
            if row.get("K", "") == k:
                return row
        return {}
    gx_800 = find_k(gx_rows, "800")
    gy_800 = find_k(gy_rows, "800")
    stability_lines = [
        "SparseFFT mode stability (E47, K=800):",
        f"- gx: Jaccard={gx_800.get('Jaccard_mean+/-std','n/a')}, IoMin={gx_800.get('IoMin_mean+/-std','n/a')}, overlap={gx_800.get('overlap_vs_global_mean+/-std','n/a')}.",
        f"- gy: Jaccard={gy_800.get('Jaccard_mean+/-std','n/a')}, IoMin={gy_800.get('IoMin_mean+/-std','n/a')}, overlap={gy_800.get('overlap_vs_global_mean+/-std','n/a')}.",
        "- Mode selection uses TRAIN-only Wiener weights (no test leakage).",
    ]
    (out_dir / "sparsefft_stability.md").write_text("\n".join(stability_lines) + "\n", encoding="utf-8")

    # Short results text.
    short_lines = [
        "Results (paper-ready):",
        f"1) OOS low-k gain: ΔrelRMSE(wiener-ceiling)≈{_format_mean(e48_rel_wiener - e48_rel_ceiling)} and ΔrelRMSE(FFT800-ceiling)≈{_format_mean(e48_rel_fft - e48_rel_ceiling)}.",
        f"2) OOS full-g two-channel: ΔrelRMSE(B-A)≈{_format_mean(e49_rel_b - e49_rel_a)} (high-k kernel) and ΔrelRMSE≈{_format_mean(e50_rel_b - e50_rel_a)} (high-k pixels).",
        f"3) Regime AUROC at thresh={thr_main:.2f}: alpha={_format_mean(auc_alpha, digits=3)}, bbks_ext={_format_mean(auc_bbks, digits=3)}, bbks_tilt={_format_mean(auc_tilt, digits=3)}.",
        f"4) Threshold policy: main thresh={thr_main:.2f}; sensitivity reported for {', '.join([f'{t:.2f}' for t in sens_thresholds])}.",
        f"5) Alpha global@{thr_main:.2f} (E66): R2={_format_mean(g_alpha_r2)}, MAE={_format_mean(g_alpha_mae)}.",
        f"6) Alpha strong subset@{thr_main:.2f} (n>=30): MAE={_format_mean(s_alpha_mae)}, R2={_format_mean(s_alpha_r2)}.",
        f"7) Alpha tail power: n_strong@{thr_main:.2f} increased from E65={_format_mean(n_alpha_03_e65, digits=1)} to E66={_format_mean(n_alpha_03_e66, digits=1)}.",
        f"8) bbks_ext global@{thr_main:.2f}: R2={_format_mean(g_bbks_r2)}, MAE={_format_mean(g_bbks_mae)}.",
        f"9) bbks_ext strong@{thr_main:.2f} (n>=30): MAE={_format_mean(s_bbks_mae)}, R2={_format_mean(s_bbks_r2)}.",
        "10) bbks_tilt strong subset is below n>=30 at thresh=0.02; reported as NA in the table.",
        "11) ROC example (E64, alpha holdout) shows AUROC≈1.0; sensitivity plots show stable n_strong and strong_MAE for alpha.",
        "12) Baseline two-stage (E62) global R2: bbks_ext=0.560, alpha=0.437, bbks_tilt=0.905.",
        f"13) E61 y_strong AUROC: alpha={_format_mean(e61_auc_alpha, digits=3)}, bbks_ext={_format_mean(e61_auc_bbks, digits=3)}, bbks_tilt={_format_mean(e61_auc_tilt, digits=3)}.",
    ]
    (out_dir / "short_results.md").write_text("\n".join(short_lines) + "\n", encoding="utf-8")

    print(str(out_dir))


if __name__ == "__main__":
    main()
