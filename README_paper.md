# Paper Bundle Reproduction

This repository includes a paper-ready bundle under `outputs/paper/` and a script to regenerate it from stored run summaries (no new experiments).

## Setup

1) Create a Python 3.11+ environment.
2) Install dependencies:
   - `python3 -m pip install -r requirements.txt`
3) (Optional) Pin versions:
   - `python3 -m pip install -r requirements.lock`

## Reproduce the paper bundle

Run:
```
scripts/reproduce_paper.sh
```

This regenerates:
- `outputs/paper/main_table.md`
- `outputs/paper/main_figure.png`
- `outputs/paper/method_schematic.png` and `.svg`
- `outputs/paper/sparsefft_table.md`, `sparsefft_curve.png`, `sparsefft_stability.md`
- `outputs/paper/draft.md`

All run directories used for the synthesis are listed in `configs/e68_threshold_policy.yaml`.

## Build PDF

Run:
```
scripts/build_pdf.sh
```

This attempts to generate `outputs/paper/draft.pdf` using pandoc + pdflatex with a Unicode sanitizer step.
If pdflatex is not available or sanitization fails, the script falls back to HTML (`outputs/paper/draft.html`).
If a PDF is created, it is also copied into `outputs/paper/submission/`.
No new experiments are run during PDF generation.

## Pre-submission audit

Run:
```
scripts/pre_submission_audit.sh
```

This regenerates the paper bundle, rebuilds the submission pack, and validates that required files exist and are non-empty.
It prints file sizes and SHA256 hashes for traceability.

## License

Code and generated figures are released under the MIT License (see `LICENSE`).
