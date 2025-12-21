#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

pdf_path="outputs/paper/submission/draft.pdf"
md_path="outputs/paper/draft.md"
meta_path="outputs/paper/SUBMISSION_METADATA.md"
submission_meta="outputs/paper/submission/SUBMISSION_METADATA.md"

if [[ ! -s "$pdf_path" ]]; then
  echo "missing: $pdf_path"
  exit 1
fi
if [[ ! -s "$md_path" ]]; then
  echo "missing: $md_path"
  exit 1
fi
if [[ ! -s "$meta_path" ]]; then
  echo "missing: $meta_path"
  exit 1
fi

python3 - <<'PY'
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

pdf_path = Path("outputs/paper/submission/draft.pdf")
md_path = Path("outputs/paper/draft.md")
meta_path = Path("outputs/paper/SUBMISSION_METADATA.md")
submission_meta = Path("outputs/paper/submission/SUBMISSION_METADATA.md")

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def extract_text_first_page(pdf: Path) -> str:
    if shutil.which("pdftotext"):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            subprocess.run(
                ["pdftotext", "-f", "1", "-l", "1", str(pdf), str(tmp_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return tmp_path.read_text(encoding="utf-8", errors="ignore")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    try:
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError:
            from PyPDF2 import PdfReader  # type: ignore
    except Exception as exc:
        raise RuntimeError("pdftotext or pypdf/PyPDF2 required for PDF text extraction") from exc

    reader = PdfReader(str(pdf))
    if not reader.pages:
        return ""
    text = reader.pages[0].extract_text() or ""
    return text

text = extract_text_first_page(pdf_path)
if not text.strip():
    raise SystemExit("PDF provenance check failed: empty text extracted from first page")

norm = re.sub(r"\s+", " ", text).strip().lower()
required = ["ceiling =", "wiener ="]
missing = [m for m in required if m not in norm]
has_dup = "key quantified results" in norm

pass_check = (not missing) and (not has_dup)

md_hash = sha256(md_path)
pdf_hash = sha256(pdf_path)

lines = meta_path.read_text(encoding="utf-8").splitlines()
lines = [
    ln
    for ln in lines
    if not ln.startswith("Draft source hash")
    and not ln.startswith("Draft PDF hash")
    and not ln.startswith("PDF provenance check:")
]
lines.append(f"Draft source hash (draft.md): {md_hash}")
lines.append(f"Draft PDF hash (submission/draft.pdf): {pdf_hash}")
lines.append(f"PDF provenance check: {'PASS' if pass_check else 'FAIL'}")
meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

if submission_meta.exists():
    submission_meta.write_text(meta_path.read_text(encoding="utf-8"), encoding="utf-8")

if not pass_check:
    msg = "PDF seems obsolete"
    if missing:
        msg += f"; missing markers: {', '.join(missing)}"
    if has_dup:
        msg += "; contains 'Key quantified results'"
    raise SystemExit(msg)

print("PDF provenance check: PASS")
PY
