#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

submission_dir="outputs/paper/submission"
mkdir -p "$submission_dir"

python3 - <<'PY'
from pathlib import Path

paper_dir = Path("outputs/paper")
draft = paper_dir / "draft.md"
out_path = paper_dir / "SUBMISSION_METADATA.md"

def extract_abstract(text: str) -> str:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "## abstract":
            start = i + 1
            break
    if start is None:
        return ""
    end = len(lines)
    for j in range(start, len(lines)):
        if lines[j].strip().startswith("## "):
            end = j
            break
    block = "\n".join(lines[start:end]).strip()
    return block

text = draft.read_text(encoding="utf-8")
abstract = extract_abstract(text)

content = """# Submission Metadata

Title: Conditional low-k prediction beyond a truncated-kernel ceiling
Short title: Conditional low-k prediction
Authors: [add authors here]
Keywords: Wiener filtering; gravitational field; sparse FFT; out-of-sample evaluation; regime modeling; spectral estimation; Poisson solver
Suggested categories: signal processing / statistical inference / computational physics

Abstract:
{abstract}

Acknowledgment:
Generative AI assistance was used for workflow support and editorial revision. Scientific choices, validation, and claims were reviewed by the author.

Plain-language summary:
We estimate a gravitational field from density using a physics-derived kernel baseline and a conditional (Wiener) predictor
that exploits correlations. We test out-of-sample generalization and show a consistent relRMSE gain over the baseline,
with a sparse FFT compression that keeps most of the gain. We also build a regime classifier that predicts when gains
are strong and report sensitivity to the strong-tail threshold.

Notes:
- Main threshold policy = 0.02 (see sensitivity appendix).
- SparseFFT trade-off: K≈800 retains most of the Wiener gain while beating the ceiling.
"""

out_path.write_text(content.format(abstract=abstract), encoding="utf-8")
print(str(out_path))
PY

draft_pdf="outputs/paper/draft.pdf"
draft_html="outputs/paper/draft.html"
draft_format=""

if [[ -s "$draft_pdf" ]]; then
  draft_format="pdf"
  draft_file="$draft_pdf"
elif [[ -s "$draft_html" ]]; then
  draft_format="html"
  draft_file="$draft_html"
else
  echo "missing: $draft_pdf or $draft_html"
  exit 1
fi

files=(
  "outputs/paper/draft.md"
  "$draft_file"
  "outputs/paper/main_table.md"
  "outputs/paper/main_figure.png"
  "outputs/paper/method_schematic.png"
  "outputs/paper/method_schematic.svg"
  "outputs/paper/sparsefft_curve.png"
  "outputs/paper/sparsefft_table.md"
  "outputs/paper/sparsefft_stability.md"
  "outputs/paper/sanity_checks.md"
  "outputs/paper/sanity_checks.png"
  "outputs/paper/compute_cost.md"
  "outputs/paper/short_results.md"
  "outputs/paper/SUBMISSION_METADATA.md"
  "references.bib"
  "CITATION.cff"
  "LICENSE"
  "README.md"
  "CHECKLIST_REVIEW.md"
  "requirements.lock"
  "scripts/reproduce_paper.sh"
  "scripts/build_pdf.sh"
  "scripts/pre_submission_audit.sh"
  "scripts/check_pdf_provenance.sh"
)

missing=0
for f in "${files[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "missing: $f"
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "submission bundle incomplete"
  exit 1
fi

SUBMISSION_FORMAT="$draft_format" python3 - <<'PY'
from pathlib import Path
import os

meta_path = Path("outputs/paper/SUBMISSION_METADATA.md")
fmt = os.environ.get("SUBMISSION_FORMAT", "")
format_line = f"Bundle format: draft.{fmt}" if fmt else "Bundle format: unknown"
lines = meta_path.read_text(encoding="utf-8").splitlines()
lines = [ln for ln in lines if not ln.startswith("Bundle format:")]
lines.append(format_line)
meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

mkdir -p "$submission_dir/scripts"
for f in "${files[@]}"; do
  if [[ "$f" == scripts/* ]]; then
    cp -f "$f" "$submission_dir/scripts/"
  else
    cp -f "$f" "$submission_dir/"
  fi
done

essential=(
  "$submission_dir/draft.md"
  "$submission_dir/$(basename "$draft_file")"
  "$submission_dir/main_figure.png"
)
for f in "${essential[@]}"; do
  if [[ ! -s "$f" ]]; then
    echo "empty: $f"
    exit 1
  fi
done

echo "submission bundle:"
for f in "${files[@]}"; do
  if [[ "$f" == scripts/* ]]; then
    base="$submission_dir/$f"
  else
    base="$submission_dir/$(basename "$f")"
  fi
  if [[ -f "$base" ]]; then
    size_bytes="$(wc -c < "$base" | tr -d ' ')"
    echo "- $base ($size_bytes bytes)"
  fi
done
