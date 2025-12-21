#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt


def parse_table(path: Path) -> list[dict[str, str]]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.startswith("| ")]
    if len(lines) < 3:
        raise ValueError(f"Missing table in {path}")
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:
        parts = [c.strip() for c in line.strip("|").split("|")]
        if len(parts) != len(header):
            continue
        rows.append(dict(zip(header, parts)))
    return rows


def build_dataset(rows: list[dict[str, str]], thresh: float) -> tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    for r in rows:
        gain = float(r["gain"])
        y.append(1 if gain > thresh else 0)
        X.append(
            [
                float(r["outside_var_frac"]),
                float(r["Pearson_inside_out"]),
                float(r["R2_inside_out"]),
                float(r["pred1"]),
                float(r["pred2"]),
            ]
        )
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError("Only one class in y_strong; cannot compute AUROC.")
    return X, y


def crossval_auc(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], prob))
    return float(np.mean(aucs)), float(np.std(aucs))


def main() -> None:
    parser = argparse.ArgumentParser(description="AUROC sanity checks from existing summary tables.")
    parser.add_argument(
        "--summary",
        default="outputs/runs/20251220_175718_e56_unified_gain_law/summary_e56_unified_gain_law.md",
        help="Path to summary table with condition-level features.",
    )
    parser.add_argument("--threshold", type=float, default=0.02, help="y_strong threshold on gain.")
    parser.add_argument("--out", default="outputs/paper/sanity_checks.md", help="Output markdown path.")
    parser.add_argument("--plot", default="outputs/paper/sanity_checks.png", help="Output plot path.")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    rows = parse_table(summary_path)
    X, y = build_dataset(rows, args.threshold)

    rng = np.random.RandomState(0)
    auc_nom, auc_nom_std = crossval_auc(X, y, seed=0)

    shuf_aucs = []
    perm_aucs = []
    for rep in range(20):
        y_shuf = rng.permutation(y)
        auc_shuf, _ = crossval_auc(X, y_shuf, seed=10 + rep)
        shuf_aucs.append(auc_shuf)
        X_perm = X.copy()
        for j in range(X_perm.shape[1]):
            rng.shuffle(X_perm[:, j])
        auc_perm, _ = crossval_auc(X_perm, y, seed=30 + rep)
        perm_aucs.append(auc_perm)

    auc_shuf, auc_shuf_std = float(np.mean(shuf_aucs)), float(np.std(shuf_aucs))
    auc_perm, auc_perm_std = float(np.mean(perm_aucs)), float(np.std(perm_aucs))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sanity checks (AUROC)",
        "",
        f"- source: `{summary_path}`",
        f"- y_strong threshold: gain > {args.threshold:.2f}",
        "- split: 5-fold stratified CV over condition-level rows (no new fields)",
        "",
        "| setting | AUROC_mean | AUROC_std |",
        "| --- | --- | --- |",
        f"| nominal | {auc_nom:.3f} | {auc_nom_std:.3f} |",
        f"| label-shuffled | {auc_shuf:.3f} | {auc_shuf_std:.3f} |",
        f"| feature-permuted | {auc_perm:.3f} | {auc_perm_std:.3f} |",
        "",
        "Note: this check uses existing condition-level feature tables (no new simulations).",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    plot_path = Path(args.plot)
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    labels = ["nominal", "label-shuffled", "feature-permuted"]
    means = [auc_nom, auc_shuf, auc_perm]
    errs = [auc_nom_std, auc_shuf_std, auc_perm_std]
    ax.bar(labels, means, yerr=errs, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("AUROC")
    ax.set_title("Sanity checks (condition-level)")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)

    if auc_shuf > 0.7 or auc_perm > 0.7:
        raise SystemExit("Sanity check failed: shuffled/permuted AUROC too high")


if __name__ == "__main__":
    main()
