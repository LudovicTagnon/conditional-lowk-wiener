#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys


REPLACEMENTS = {
    "Δ": r"\ensuremath{\Delta}",
    "±": r"\ensuremath{\pm}",
    "≥": r"\ensuremath{\ge}",
    "≤": r"\ensuremath{\le}",
    "≈": r"\ensuremath{\approx}",
    "→": r"\ensuremath{\to}",
    "−": "-",
    "–": "-",
    "—": "-",
    "“": '"',
    "”": '"',
    "’": "'",
    "…": "...",
}


def sanitize_text(text: str) -> str:
    for src, dst in REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text


def report_non_ascii(text: str) -> int:
    counts: Counter[str] = Counter()
    line_hits: dict[str, list[int]] = defaultdict(list)
    for idx, line in enumerate(text.splitlines(), start=1):
        for ch in line:
            if ord(ch) > 127:
                counts[ch] += 1
                if idx not in line_hits[ch]:
                    line_hits[ch].append(idx)
    if not counts:
        return 0
    print("non-ascii report (top 20):")
    for ch, count in counts.most_common(20):
        cp = f"U+{ord(ch):04X}"
        lines = ", ".join(str(i) for i in line_hits[ch][:5])
        more = "" if len(line_hits[ch]) <= 5 else "..."
        print(f"- {cp} '{ch}': {count} hits; lines {lines}{more}")
    return 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanitize Markdown for pdflatex.")
    parser.add_argument("input", help="Input markdown path")
    parser.add_argument("output", help="Output markdown path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    text = in_path.read_text(encoding="utf-8")
    sanitized = sanitize_text(text)
    out_path.write_text(sanitized, encoding="utf-8")
    return report_non_ascii(sanitized)


if __name__ == "__main__":
    sys.exit(main())
