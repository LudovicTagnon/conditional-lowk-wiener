# Paper Verification

Date: 2026-03-11
Project: `topological_gravity_minilab`

This note records the verification pass performed on the stable paper line without changing the scientific content of the repository.

## Commands run

- `scripts/pre_submission_audit.sh`
- `python3 -m py_compile lab/*.py scripts/*.py`
- `python3 -m lab.run --config configs/e0_1overf_alpha2.yaml --override n_patches=500 --override grid_size=24 --override exp_name=e0_smoke_codex_20260311`
- `python3 -m lab.report --path outputs/runs/20260311_175809_e0_smoke_codex_20260311`

## Paper bundle status

- `pre_submission_audit.sh`: PASS
- PDF provenance: PASS
- submission bundle regenerated under `outputs/paper/submission/`
- `draft.pdf` regenerated and copied into the submission bundle

## Stable paper metrics confirmed

From `outputs/paper/main_table.md` and `outputs/paper/short_results.md`:

- OOS low-k relRMSE: ceiling `0.1549`, Wiener `0.1229`, sparseFFT `0.1244`
- OOS low-k gains vs ceiling: Wiener `-0.0320`, sparseFFT `-0.0305`
- OOS full-g two-channel gains vs baseline A: `-0.0308` (high-k kernel), `-0.0316` (high-k pixels)
- regime AUROC at threshold `0.02`: alpha `0.989`, bbks_ext `1.000`, bbks_tilt `1.000`

These are the numbers that appear fixed by the paper bundle and remain the strongest scientific core of the repository.

## Smoke run result

Run:

- `outputs/runs/20260311_175809_e0_smoke_codex_20260311`

Summary:

- baseline A: `pearson=0.0654`, `relRMSE=1.0217`
- model B: `pearson=0.3291`, `relRMSE=0.9513`
- model C: `pearson=0.4392`, `relRMSE=0.9024`

This was only a lightweight runner smoke test, not a paper-grade rerun.

## Practical verdict

- The stable paper nucleus is operational and auditable from stored local artifacts.
- The submission tooling is coherent end-to-end.
- The large `E70+` extension line remains scientifically interesting but should still be read as separate tracks rather than as one single unified paper narrative.
