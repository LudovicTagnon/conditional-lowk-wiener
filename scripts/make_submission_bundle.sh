#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

scripts/reproduce_paper.sh

submission_dir="outputs/paper/submission"
mkdir -p "$submission_dir"

files=(
  "outputs/paper/draft.md"
  "outputs/paper/main_figure.png"
  "outputs/paper/method_schematic.svg"
  "outputs/paper/sparsefft_curve.png"
  "outputs/paper/main_table.md"
  "outputs/paper/sparsefft_table.md"
  "outputs/paper/short_results.md"
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

for f in "${files[@]}"; do
  cp -f "$f" "$submission_dir/"
done

echo "submission bundle:"
for f in "${files[@]}"; do
  base="$(basename "$f")"
  size_bytes="$(wc -c < "$submission_dir/$base" | tr -d ' ')"
  echo "- $submission_dir/$base ($size_bytes bytes)"
done
