# Extension Reproduction

Date: 2026-03-12
Project: `topological_gravity_minilab`

This note records the practical reading and reproduction path for the strongest post-paper extension families.

## Scope

These tracks are promoted because they have the cleanest local evidence bundles beyond the stable paper nucleus:

- SPARC / RAR
- weak lensing
- safety / abstention audits

## Recommended reading order

1. `SPARC_RAR_EVIDENCE_MAP_2026-03-12.md`
2. `WEAK_LENSING_EVIDENCE_MAP_2026-03-12.md`
3. `SAFETY_AUDIT_EVIDENCE_MAP_2026-03-12.md`

## Canonical saved artifacts

### SPARC / RAR

- `outputs/E76_summary.md`
- `outputs/E76_20251223_145149_e76_sparc_catalog_ylib/E76_table_models_fixedY.md`
- `outputs/E76_20251223_145149_e76_sparc_catalog_ylib/E76_residual_correlations.md`

### Weak lensing

- `outputs/E78_summary.md`
- `outputs/E78_20251223_150851_e78_verlinde_weak_lensing/E78_table.md`
- `outputs/E78_20251223_150851_e78_verlinde_weak_lensing/E78_overlay.png`

### Safety / abstention

- `outputs/E160_20260102_001116_e160_fixed_threshold_audit/E160_summary.md`
- `outputs/E161_20260102_001542_e161_fixed_conservative/E161_summary.md`
- `outputs/E161_20260102_001542_e161_fixed_conservative/E161_best_params.json`
- `outputs/E163_20260102_002212_e163_bulk_audit/E163_summary.md`
- `outputs/E163_20260102_002212_e163_bulk_audit/E163_rule_card_bulk_only.txt`

## Practical note

At this stage, the most reliable local workflow for these extensions is artifact-first, not rerun-first:

- the outputs already exist;
- they are more stable than the surrounding exploratory code path;
- the local `lab/run.py` has active work in progress and should not be treated as a frozen reproduction target for these extension families.

So the clean reproduction strategy is:

1. read the evidence maps
2. inspect the saved summaries and tables
3. only then decide whether one track deserves its own dedicated rerun / paper / repository
