# Scientific Status

Date: 2026-03-11
Project: `topological_gravity_minilab`

## Scope

This is the most mature scientific project in the workspace, but it already contains two distinct layers:

- a stable paper line on conditional low-k prediction beyond a finite-support local ceiling;
- a broader local exploration line (`E70+`) covering SPARC, weak lensing, effective laws, OOD gating, and safety audits.

The first line is publication-ready. The second line is active and interesting, but much less consolidated.

## Important pivot

Historically, the repository presents itself as a "topological gravity" mini-lab centered on local `b0`-style features. The strongest current scientific line is no longer that broad claim.

The actual core contribution has shifted toward:

- conditional low-k prediction;
- Wiener-style linear inference from local windows;
- sparse spectral compression;
- clean out-of-sample benchmarking against a finite-support local ceiling.

So the repository name preserves the origin story, but the paper-grade scientific center is now much closer to a controlled signal-processing / conditional-inference benchmark than to a general topological-gravity theory.

## Main artifacts

- [outputs/paper/draft.md](/home/kudovic/Desktop/RECHERCHE/projects_active/topological_gravity_minilab/outputs/paper/draft.md): canonical manuscript source for the stable line.
- [outputs/paper/main_table.md](/home/kudovic/Desktop/RECHERCHE/projects_active/topological_gravity_minilab/outputs/paper/main_table.md): main result table.
- [PAPER_VERIFICATION_2026-03-11.md](/home/kudovic/Desktop/RECHERCHE/projects_active/topological_gravity_minilab/PAPER_VERIFICATION_2026-03-11.md): local verification record for the stable line.
- [PAPER_EVIDENCE_MAP_2026-03-11.md](/home/kudovic/Desktop/RECHERCHE/projects_active/topological_gravity_minilab/PAPER_EVIDENCE_MAP_2026-03-11.md): provenance map from paper claims to exact runs and generated artifacts.
- [OUTPUTS_INDEX_2026-03-11.md](/home/kudovic/Desktop/RECHERCHE/projects_active/topological_gravity_minilab/OUTPUTS_INDEX_2026-03-11.md): versioned guide to the ignored `outputs/` tree.
- [scripts/reproduce_paper.sh](/home/kudovic/Desktop/RECHERCHE/projects_active/topological_gravity_minilab/scripts/reproduce_paper.sh): rebuilds the paper bundle from stored artifacts.
- [scripts/pre_submission_audit.sh](/home/kudovic/Desktop/RECHERCHE/projects_active/topological_gravity_minilab/scripts/pre_submission_audit.sh): audit layer for submission hygiene.
- [outputs/E76_20251223_145149_e76_sparc_catalog_ylib/E76_table_models_fixedY.md](/home/kudovic/Desktop/RECHERCHE/projects_active/topological_gravity_minilab/outputs/E76_20251223_145149_e76_sparc_catalog_ylib/E76_table_models_fixedY.md): one of the strongest SPARC-side extension artifacts.
- [outputs/E78_20251223_150851_e78_verlinde_weak_lensing/E78_table.md](/home/kudovic/Desktop/RECHERCHE/projects_active/topological_gravity_minilab/outputs/E78_20251223_150851_e78_verlinde_weak_lensing/E78_table.md): weak-lensing extension snapshot.
- [outputs/E161_20260102_001542_e161_fixed_conservative/E161_summary.md](/home/kudovic/Desktop/RECHERCHE/projects_active/topological_gravity_minilab/outputs/E161_20260102_001542_e161_fixed_conservative/E161_summary.md) and [outputs/E163_20260102_002212_e163_bulk_audit/E163_summary.md](/home/kudovic/Desktop/RECHERCHE/projects_active/topological_gravity_minilab/outputs/E163_20260102_002212_e163_bulk_audit/E163_summary.md): latest safety-audit checkpoints.

## Stable scientific core

### Current thesis

The stable claim is narrow and strong:

- conditional linear prediction of the low-k field beats a finite-support local/truncated-kernel baseline out of sample;
- the gain persists when folded back into full-g via a two-channel decomposition;
- sparse spectral compression preserves most of the Wiener gain.

### Quantitative points that appear fixed in the paper bundle

- OOS low-k relRMSE:
  - local ceiling: `0.1549`;
  - Wiener: `0.1229`;
  - sparseFFT K≈800: `0.1244`.
- OOS low-k gain vs local ceiling:
  - Wiener: `-0.0320`;
  - sparseFFT: `-0.0305`.
- OOS full-g two-channel gain vs baseline A:
  - high-k kernel branch: `-0.0308`;
  - high-k pixels branch: `-0.0316`.
- Regime classification at threshold `0.02`:
  - AUROC `0.989` for alpha;
  - AUROC `1.000` for bbks_ext and bbks_tilt.

This is the project's cleanest and most defensible scientific contribution.

## Extension line beyond the paper

### Empirical extensions that look non-trivial

- SPARC/RAR side:
  - in [E76_table_models_fixedY.md](/home/kudovic/Desktop/RECHERCHE/projects_active/topological_gravity_minilab/outputs/E76_20251223_145149_e76_sparc_catalog_ylib/E76_table_models_fixedY.md), several effective-law models (`H2_mond`, `H3_fit`, `H1_iso`) decisively outperform the baryonic baseline `B0` on `nll_test`, `mad`, and `relRMSE_log`.
- Weak lensing side:
  - in [E78_table.md](/home/kudovic/Desktop/RECHERCHE/projects_active/topological_gravity_minilab/outputs/E78_20251223_150851_e78_verlinde_weak_lensing/E78_table.md), `H2_verlinde` and `H1_powerlaw_fit` strongly beat the baryonic baseline on `chi2` and `relRMSE`.

These results are interesting, but they are not yet integrated into the stable paper line with the same level of framing or audit.

### Recent safety / abstention line

- The 2026 line is no longer about raw gain alone; it is about making the extension layer safer under domain shift.
- [E160_summary.md](/home/kudovic/Desktop/RECHERCHE/projects_active/topological_gravity_minilab/outputs/E160_20260102_001116_e160_fixed_threshold_audit/E160_summary.md) shows the pre-conservative state:
  - `FAIL: unsafe survivors in TEST-only autopsy (n=3)`;
  - mean coverage around `0.87`.
- [E161_summary.md](/home/kudovic/Desktop/RECHERCHE/projects_active/topological_gravity_minilab/outputs/E161_20260102_001542_e161_fixed_conservative/E161_summary.md) and [E163_summary.md](/home/kudovic/Desktop/RECHERCHE/projects_active/topological_gravity_minilab/outputs/E163_20260102_002212_e163_bulk_audit/E163_summary.md) show the current status:
  - `PASS: no unsafe survivors in TEST-only autopsy`;
  - max `worst_delta_test` around `0.0280`;
  - mean coverage around `0.84`;
  - holdout coverage is uneven, with very low acceptance on some groups.

This is scientifically valuable, but it belongs to a different project phase than the low-k paper.

## Reproducibility state

- The paper line is highly reproducible from local artifacts:
  - manuscript sources are generated from stored outputs;
  - rebuild and audit scripts are present;
  - the bundle is already shaped for submission.
- The `E70+` line is only partially consolidated:
  - many configs, post-processors, and summaries exist;
  - the scientific story spanning them is still fragmented by topic.

## Main structural problems

- One repository now hosts several research programs:
  - synthetic low-k prediction;
  - empirical galaxy-law fitting;
  - weak-lensing checks;
  - safety / abstention / OOD control.
- The paper line is clear; the extension line is rich but conceptually diffuse.
- Without an explicit split, it is easy to over-read the whole repo as one unified theory arc when it is really a stack of partially connected programs.

## Recommended next content work

1. Keep the low-k paper as the canonical scientific nucleus of the repository.
2. Split `E70+` into named thematic tracks instead of one long experiment chain.
3. Promote only the strongest extension families into paper-ready narratives:
   - SPARC/RAR,
   - weak lensing,
   - safety audits.
