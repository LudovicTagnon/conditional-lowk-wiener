#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

scripts/reproduce_paper.sh

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

files=(
  "outputs/paper/draft.md"
  "outputs/paper/main_figure.png"
  "outputs/paper/method_schematic.png"
  "outputs/paper/method_schematic.svg"
  "outputs/paper/sparsefft_curve.png"
  "outputs/paper/main_table.md"
  "outputs/paper/sparsefft_table.md"
  "outputs/paper/sparsefft_stability.md"
  "outputs/paper/short_results.md"
  "outputs/paper/SUBMISSION_METADATA.md"
  "references.bib"
  "CITATION.cff"
  "LICENSE"
  "README_paper.md"
  "CHECKLIST_REVIEW.md"
)

missing=0
optional_files=()
if [[ -f "outputs/paper/draft.pdf" ]]; then
  optional_files+=("outputs/paper/draft.pdf")
fi

for f in "${files[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "missing: $f"
    missing=1
  fi
done
for f in "${optional_files[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "optional missing: $f"
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo "submission bundle incomplete"
  exit 1
fi

for f in "${files[@]}"; do
  cp -f "$f" "$submission_dir/"
done
for f in "${optional_files[@]}"; do
  cp -f "$f" "$submission_dir/"
done

echo "submission bundle:"
for f in "${files[@]}" "${optional_files[@]}"; do
  base="$(basename "$f")"
  if [[ -f "$submission_dir/$base" ]]; then
    size_bytes="$(wc -c < "$submission_dir/$base" | tr -d ' ')"
    echo "- $submission_dir/$base ($size_bytes bytes)"
  fi
done
