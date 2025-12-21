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
