# Outputs Index

Date: 2026-03-11

`outputs/` is ignored by Git and contains several kinds of artifacts. This note is the versioned map of that space.

## 1. Stable paper nucleus

Start here for the most mature scientific line:

- `outputs/paper/`
- `outputs/paper/draft.md`
- `outputs/paper/main_table.md`
- `outputs/paper/short_results.md`
- `outputs/paper/submission/`

This is the clean publication-facing bundle.

## 2. Runner-generated local experiments

- `outputs/runs/`

This folder contains timestamped directories created by `python3 -m lab.run --config ...`.

Naming pattern:

- `outputs/runs/<YYYYMMDD_HHMMSS>_<exp_name>/`

These are the raw local run artifacts behind many of the paper and extension summaries.

## 3. Named extension snapshots

The `E70+` families also appear as top-level directories directly under `outputs/`, for example:

- `E76_20251223_145149_e76_sparc_catalog_ylib`
- `E78_20251223_150851_e78_verlinde_weak_lensing`
- `E161_20260102_001542_e161_fixed_conservative`
- `E163_20260102_002212_e163_bulk_audit`

These are best interpreted using `EXPERIMENT_TRACKS_2026-03-11.md`.

## 4. Practical reading order

1. `outputs/paper/`
2. representative extension checkpoints (`E76`, `E78`, `E161`, `E163`)
3. only then the broader `E70+` directory set
