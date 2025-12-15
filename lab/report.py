from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunSummary:
    run_dir: Path
    exp_name: str
    experiment: str
    metrics: dict[str, Any]
    metrics_by_mass_bin: list[dict[str, Any]] | None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _read_metrics_by_mass_bin(path: Path) -> list[dict[str, Any]]:
    import csv

    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out: dict[str, Any] = dict(row)
            for k, v in list(out.items()):
                if v is None:
                    continue
                s = str(v)
                try:
                    if s.strip() == "":
                        continue
                    if "." in s or "e" in s.lower():
                        out[k] = float(s)
                    else:
                        out[k] = int(s)
                except ValueError:
                    out[k] = s
            rows.append(out)
    return rows


def _format_md_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        return "| " + " | ".join(cells[i].ljust(widths[i]) for i in range(len(cells))) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    out = [fmt_row(headers), sep]
    out.extend(fmt_row(r) for r in rows)
    return "\n".join(out)


def _fmt_float(x: Any, nd: int = 4) -> str:
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if xf != xf:  # NaN
        return "nan"
    return f"{xf:.{nd}f}"


def _summarize_run(run_dir: Path) -> RunSummary:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json in {run_dir}")

    metrics = _read_json(metrics_path)
    exp_name = str(metrics.get("exp_name", run_dir.name))
    experiment = str(metrics.get("experiment", ""))
    mb_path = run_dir / "metrics_by_mass_bin.csv"
    mb = _read_metrics_by_mass_bin(mb_path) if mb_path.exists() else None
    return RunSummary(run_dir=run_dir, exp_name=exp_name, experiment=experiment, metrics=metrics, metrics_by_mass_bin=mb)


def _print_model_table(summary: RunSummary) -> None:
    models = summary.metrics.get("models", {})
    deltas = summary.metrics.get("deltas_vs_A", {})
    if not isinstance(models, dict):
        raise ValueError("metrics.json: expected 'models' mapping")

    headers = ["model", "pearson", "rmse", "relRMSE", "Δpearson vs A", "ΔrelRMSE vs A"]
    rows: list[list[str]] = []
    for m in ["A", "B", "C"]:
        mm = models.get(m, {})
        if not isinstance(mm, dict):
            mm = {}
        d = deltas.get(m, {}) if m != "A" else {"delta_pearson": 0.0, "delta_relRMSE": 0.0}
        if not isinstance(d, dict):
            d = {"delta_pearson": "?", "delta_relRMSE": "?"}
        rows.append(
            [
                m,
                _fmt_float(mm.get("pearson")),
                _fmt_float(mm.get("rmse")),
                _fmt_float(mm.get("relRMSE")),
                _fmt_float(d.get("delta_pearson")),
                _fmt_float(d.get("delta_relRMSE")),
            ]
        )

    print(f"\n## {summary.run_dir.name}")
    print(f"- experiment: {summary.experiment}")
    print(_format_md_table(headers, rows))


def _mass_bin_summary(rows: list[dict[str, Any]]) -> str:
    import numpy as np

    def col(name: str) -> np.ndarray:
        vals: list[float] = []
        for r in rows:
            v = r.get(name)
            try:
                vals.append(float(v))
            except Exception:
                continue
        return np.asarray(vals, dtype=np.float64)

    pearson_c = col("pearson_C")
    rel_c = col("relRMSE_C")
    if pearson_c.size == 0 or rel_c.size == 0:
        return "E2 summary: missing columns (expected pearson_C and relRMSE_C)."
    if np.all(~np.isfinite(pearson_c)) or np.all(~np.isfinite(rel_c)):
        return "E2 summary: bins contain only non-finite values."

    best_p = int(np.nanargmax(pearson_c))
    worst_p = int(np.nanargmin(pearson_c))
    best_r = int(np.nanargmin(rel_c))
    worst_r = int(np.nanargmax(rel_c))

    s = (
        "E2 bins (model C, test): "
        f"pearson[min/med/max]={_fmt_float(np.nanmin(pearson_c))}/"
        f"{_fmt_float(np.nanmedian(pearson_c))}/{_fmt_float(np.nanmax(pearson_c))}, "
        f"best_bin={rows[best_p].get('bin')} worst_bin={rows[worst_p].get('bin')}; "
        f"relRMSE[min/med/max]={_fmt_float(np.nanmin(rel_c))}/"
        f"{_fmt_float(np.nanmedian(rel_c))}/{_fmt_float(np.nanmax(rel_c))}, "
        f"best_bin={rows[best_r].get('bin')} worst_bin={rows[worst_r].get('bin')}."
    )
    return s


def _iter_latest_runs(runs_dir: Path, latest: int) -> list[Path]:
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs dir not found: {runs_dir}")
    dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.name)
    if latest <= 0:
        return dirs
    return dirs[-latest:]


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--path", type=str, help="Path to a single run directory.")
    g.add_argument("--runs", type=str, help="Path to outputs/runs directory.")
    p.add_argument("--latest", type=int, default=5, help="When using --runs, report only the latest N runs.")
    args = p.parse_args()

    run_dirs: list[Path]
    if args.path:
        run_dirs = [Path(args.path)]
    else:
        run_dirs = _iter_latest_runs(Path(args.runs), int(args.latest))

    for rd in run_dirs:
        summary = _summarize_run(rd)
        _print_model_table(summary)
        if summary.metrics_by_mass_bin is not None:
            print(f"- {_mass_bin_summary(summary.metrics_by_mass_bin)}")


if __name__ == "__main__":
    main()
