#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

md="outputs/paper/draft.md"
pdf="outputs/paper/draft.pdf"
html="outputs/paper/draft.html"
sanitized="outputs/paper/draft_sanitized.md"
submission_dir="outputs/paper/submission"

if [[ ! -f "$md" ]]; then
  echo "missing: $md"
  exit 1
fi

generated=0

if command -v pandoc >/dev/null 2>&1 && command -v pdflatex >/dev/null 2>&1; then
  if python3 scripts/sanitize_md_for_latex.py "$md" "$sanitized"; then
    if pandoc --from=markdown+raw_tex "$sanitized" --pdf-engine=pdflatex -o "$pdf"; then
      generated=1
    fi
  else
    echo "sanitize failed; skipping pdflatex path"
  fi
fi

if [[ "$generated" -eq 0 ]]; then
  if command -v pandoc >/dev/null 2>&1; then
    if pandoc "$md" -o "$html"; then
      if command -v wkhtmltopdf >/dev/null 2>&1; then
        if wkhtmltopdf "$html" "$pdf"; then
          generated=1
        fi
      fi
    fi
  fi
fi

if [[ "$generated" -eq 0 ]]; then
  python3 - <<'PY'
from pathlib import Path
import html

md_path = Path("outputs/paper/draft.md")
html_path = Path("outputs/paper/draft.html")
text = md_path.read_text(encoding="utf-8")
try:
    import markdown  # type: ignore
    body = markdown.markdown(text)
    content = f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>draft</title></head><body>{body}</body></html>"
except Exception:
    escaped = html.escape(text)
    content = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>draft</title></head>"
        "<body><pre style='white-space:pre-wrap;font-family:monospace;'>"
        + escaped
        + "</pre></body></html>"
    )
html_path.write_text(content, encoding="utf-8")
print(str(html_path))
PY
  if command -v wkhtmltopdf >/dev/null 2>&1; then
    if wkhtmltopdf "$html" "$pdf"; then
      generated=1
    fi
  fi
fi

if [[ -f "$pdf" ]]; then
  mkdir -p "$submission_dir"
  cp -f "$pdf" "$submission_dir/"
  echo "generated: $pdf"
else
  echo "PDF not generated; fallback HTML at $html"
fi

exit 0
