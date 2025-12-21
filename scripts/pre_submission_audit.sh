#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

scripts/reproduce_paper.sh
scripts/make_submission_bundle.sh
scripts/build_pdf.sh

submission_dir="outputs/paper/submission"

required_files=(
  "draft.md"
  "main_table.md"
  "main_figure.png"
  "sparsefft_table.md"
  "sparsefft_curve.png"
  "sparsefft_stability.md"
  "references.bib"
  "CITATION.cff"
  "LICENSE"
  "README_paper.md"
  "CHECKLIST_REVIEW.md"
)

missing=0
for f in "${required_files[@]}"; do
  path="$submission_dir/$f"
  if [[ ! -s "$path" ]]; then
    echo "missing or empty: $path"
    missing=1
  fi
done

schematic_svg="$submission_dir/method_schematic.svg"
schematic_png="$submission_dir/method_schematic.png"
if [[ ! -s "$schematic_svg" && ! -s "$schematic_png" ]]; then
  echo "missing: $schematic_svg or $schematic_png"
  missing=1
fi

if [[ "$missing" -ne 0 ]]; then
  echo "pre-submission audit failed"
  exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
  hash_cmd="sha256sum"
elif command -v shasum >/dev/null 2>&1; then
  hash_cmd="shasum -a 256"
else
  hash_cmd=""
fi

echo "submission audit:"
print_file() {
  local file_path="$1"
  local size_bytes
  size_bytes="$(wc -c < "$file_path" | tr -d ' ')"
  if [[ -n "$hash_cmd" ]]; then
    local hash
    hash="$($hash_cmd "$file_path" | awk '{print $1}')"
    echo "- $file_path ($size_bytes bytes) $hash"
  else
    echo "- $file_path ($size_bytes bytes) hash_unavailable"
  fi
}

for f in "${required_files[@]}"; do
  print_file "$submission_dir/$f"
done
if [[ -s "$schematic_png" ]]; then
  print_file "$schematic_png"
fi
if [[ -s "$schematic_svg" ]]; then
  print_file "$schematic_svg"
fi
if [[ -s "$submission_dir/draft.pdf" ]]; then
  print_file "$submission_dir/draft.pdf"
else
  html_path="outputs/paper/draft.html"
  if [[ -s "$html_path" ]]; then
    print_file "$html_path"
  else
    echo "missing: $html_path"
    exit 1
  fi
fi
if [[ -s "$submission_dir/SUBMISSION_METADATA.md" ]]; then
  print_file "$submission_dir/SUBMISSION_METADATA.md"
fi

echo "pre-submission audit OK"
