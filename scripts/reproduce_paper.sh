#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$root_dir"

python3 -m lab.paper --config configs/e68_threshold_policy.yaml
python3 -m lab.draft --paper-dir outputs/paper
python3 scripts/sanity_checks.py
python3 scripts/compute_cost.py

required_files=(
  "outputs/paper/main_table.md"
  "outputs/paper/main_figure.png"
  "outputs/paper/draft.md"
  "outputs/paper/method_schematic.png"
  "outputs/paper/method_schematic.svg"
  "outputs/paper/sparsefft_table.md"
  "outputs/paper/sparsefft_curve.png"
  "outputs/paper/sparsefft_stability.md"
  "outputs/paper/sanity_checks.md"
  "outputs/paper/sanity_checks.png"
  "outputs/paper/compute_cost.md"
)

missing=0
for f in "${required_files[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "missing: $f"
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo "paper bundle incomplete"
  exit 1
fi

echo "paper bundle OK:"
for f in "${required_files[@]}"; do
  size_bytes="$(wc -c < "$f" | tr -d ' ')"
  echo "- $f ($size_bytes bytes)"
done
