# Scripts Guide

These scripts are the reproducibility and submission layer of the repository.

## Stable paper path

Run in this order when the goal is to rebuild the paper bundle from stored artifacts:

1. `scripts/reproduce_paper.sh`
2. `scripts/run_sanity_checks.sh`
3. `scripts/build_pdf.sh`
4. `scripts/make_submission_bundle.sh`
5. `scripts/pre_submission_audit.sh`

## What each script does

- `reproduce_paper.sh`: rebuilds `outputs/paper/` from stored experiment outputs, including sanity-check and compute-support notes
- `run_sanity_checks.sh`: regenerates the sanity-check note and figure
- `build_pdf.sh`: renders `outputs/paper/draft.pdf` when tooling is available
- `make_submission_bundle.sh`: assembles `outputs/paper/submission/` from the current paper outputs
- `check_pdf_provenance.sh`: checks that the bundled PDF comes from the current paper directory
- `pre_submission_audit.sh`: end-to-end audit of the submission directory
- `compute_cost.sh` / `compute_cost.py`: cost/accounting helper for the paper bundle

These scripts are the cleanest operational interface of the repository for publication-facing work.
