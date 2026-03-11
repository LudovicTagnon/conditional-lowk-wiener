# Experiment Tracks

Date: 2026-03-11

This note groups the main experiment families living inside the `E70+` extension line so the repository can be read as a set of tracks rather than one long undifferentiated chain.

## Track A — Stable paper nucleus

This is the cleanest line and should stay the canonical center of the repository.

Main artifacts:

- `outputs/paper/`
- `outputs/paper/draft.md`
- `outputs/paper/main_table.md`
- `scripts/reproduce_paper.sh`
- `scripts/pre_submission_audit.sh`

Scientific focus:

- conditional low-k prediction;
- Wiener vs finite-support local ceiling;
- sparse spectral compression;
- regime-to-magnitude modeling with strict OOS evaluation.

## Track B — Galaxy-law / SPARC / RAR extensions

Representative runs:

- `outputs/E70_20251223_133517_e70_h1_h2_mond`
- `outputs/E73_20251223_141154_e73_sparc_rar`
- `outputs/E74_20251223_141854_e74_sparc_scatter`
- `outputs/E75_20251223_143115_e75_sparc_ylib`
- `outputs/E76_20251223_145149_e76_sparc_catalog_ylib`
- `outputs/E77_20251223_145854_e77_sparc_freeY_grid`

What this track does:

- baryonic-to-effective-law fits on SPARC-like data;
- RAR-style comparisons;
- catalog-level evaluation and model ranking.

Current value:

- probably the strongest empirical extension after the paper line;
- still not framed with the same reproducibility and narrative discipline as Track A.

## Track C — Weak lensing / effective-law bridge

Representative runs:

- `outputs/E78_20251223_150851_e78_verlinde_weak_lensing`
- `outputs/E79_20251223_152109_e79_parametric_kernel_to_rar`
- `outputs/E81_20251223_154656_e81_efe_emergent`

What this track does:

- checks whether effective kernels / Verlinde-like variants can improve lensing-side fits;
- probes bridges between empirical galaxy laws and lensing summaries.

Current value:

- promising;
- clearly an extension track, not yet part of the paper-grade center.

## Track D — Synthetic law search and residual structures

Representative runs:

- `outputs/E80_20251223_153734_e80_kband_ablation`
- `outputs/E82_20251223_160725_e82_lcdm_rk`
- `outputs/E83_20251223_172903_e83_lcdm_kernel_theory`
- `outputs/E84_20251223_174159_e84_micro2macro_rar`
- `outputs/E85_20251223_230715_e85_nongaussian`
- `outputs/E86_20251223_231928_e86_renorm_kernel`
- `outputs/E87_20251223_233636_e87_renorm_predict`
- `outputs/E94_20251226_231207_e94_residual_basis`
- `outputs/E95_20251226_232243_e95_multiplicative_ratio`
- `outputs/E96_20251226_233417_e96_multiplicative_ratio_stable`
- `outputs/E97_20251226_234535_e97_minimal_law_extreme`
- `outputs/E98_20251226_235357_e98_stable_regime`

What this track does:

- synthetic field families and ablations;
- residual-basis decompositions;
- multiplicative or minimal law searches;
- renormalized-kernel style hypotheses.

Current value:

- broad exploratory reservoir;
- scientifically useful, but internally heterogeneous.

## Track E — Safety / OOD / abstention program

This is the largest late-stage extension family. It is best read as a separate research program about safe deployment of the extension models under domain shift.

Representative checkpoints:

- `outputs/E88_20251224_121009_e88_tail_invariants`
- `outputs/E90_20251224_155008_e90_distill_ood`
- `outputs/E99_20251227_000454_e99_safe_moe`
- `outputs/E100_20251227_010251_e100_hier_safe_moe`
- `outputs/E101_20251227_073659_e101_safe_gate_full`
- `outputs/E102_20251227_074325_e102_risk_gate_conformal`
- `outputs/E103_20251227_091012_e103_tail_invariants_gate`
- `outputs/E104_20251227_091623_e104_gate_distillation`
- `outputs/E105_20251227_093717_e105_stresstest_altfields`
- `outputs/E106_20251227_095019_e106_aniso_hardening`
- `outputs/E107_20251227_202026_e107_multiseed_domain`
- `outputs/E108_20251227_204821_e108_multifamily_conformal`
- `outputs/E110_20251227_224532_e110_mondrian_cqr_gate`
- `outputs/E111_20251227_235713_e111_cluster_mondrian_veto`
- `outputs/E114_20251228_124800_e114_safety_hardening`
- `outputs/E116_20251229_083737_e116_safety_margin_gate`
- `outputs/E119_20251229_111649_e119_ubeps_veto`
- `outputs/E123_20251229_162329_e123_conformal_veto`
- `outputs/E125_20251229_163304_e125_survivor_veto`
- `outputs/E139_ood_veto_20251230_005515_e139_ood_veto`
- `outputs/E142_20251230_110559_e142_holdout_strict`
- `outputs/E147_20251230_124439_e147_tail_veto_tree`
- `outputs/E148_20251230_131931_e148_energycap_gate`
- `outputs/E157_20260101_222804_e157_residual_spike_veto`
- `outputs/E160_20260102_001116_e160_fixed_threshold_audit`
- `outputs/E161_20260102_001542_e161_fixed_conservative`
- `outputs/E163_20260102_002212_e163_bulk_audit`

What this track does:

- tail-risk diagnostics;
- gating, veto, conformal and Mondrian-style abstention;
- multi-family and holdout audits;
- safety hardening under anisotropy and domain shift.

Current reading:

- `E160` is a useful failure checkpoint;
- `E161` and `E163` are the current conservative pass points.

## How to use this split

- If the goal is publication or external communication, start from Track A.
- If the goal is the strongest empirical extension, inspect Track B first, then Track C.
- If the goal is safe deployment / abstention research, jump directly to Track E.
- Track D is best treated as exploratory support material unless one sub-family is promoted into its own focused paper.
- For the raw directory layout inside the ignored `outputs/` tree, pair this note with `OUTPUTS_INDEX_2026-03-11.md`.
