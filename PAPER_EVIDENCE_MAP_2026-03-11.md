# Paper Evidence Map

Date: 2026-03-11
Project: `topological_gravity_minilab`

This note maps the stable paper claims to the exact stored runs, synthesis config, and generated artifacts that support them.

## Stable scope

The publication-facing nucleus of this repository is the conditional low-k / Wiener / sparseFFT / threshold-policy line synthesized by:

- `configs/e68_threshold_policy.yaml`
- `lab/paper.py`
- `lab/draft.py`
- `scripts/reproduce_paper.sh`

The paper line is supported by stored local runs. No new heavy experiments are needed to rebuild `outputs/paper/`.

## Synthesis config

`configs/e68_threshold_policy.yaml` pins the canonical run map:

- `e46`: `outputs/runs/20251220_121206_e46_sparsefft_to_wiener_alpha2`
- `e47`: `outputs/runs/20251220_122530_e47_sparsefft_mode_stability_alpha2`
- `e48`: `outputs/runs/20251220_123326_e48_oos_sparsefft_vs_wiener_alpha2`
- `e49`: `outputs/runs/20251220_124104_e49_oos_twochannel_fullg_wiener_sparsefft`
- `e50`: `outputs/runs/20251220_125130_e50_oos_twochannel_fullg_wiener_sparsefft_highkpixels`
- `e61`: `outputs/runs/20251220_215420_e61_regime_gainlaw_stable`
- `e62`: `outputs/runs/20251220_222548_e62_two_stage_gain_model`
- `e65`: `outputs/runs/20251221_004201_e65_blend_stability_thresholds`
- `e66`: `outputs/runs/20251221_021252_e66_alpha_tail_power`
- `e64_roc`: `outputs/runs/20251221_001325_e64_capped_weak_blend`

## Claim-to-artifact map

### 1. OOS low-k beats the local ceiling

Primary evidence:

- `outputs/runs/20251220_123326_e48_oos_sparsefft_vs_wiener_alpha2/summary_e48_oos_sparsefft_vs_wiener_alpha2.md`
- `outputs/paper/main_table.md`
- `outputs/paper/short_results.md`

Numbers carried into the paper bundle:

- local ceiling relRMSE: `0.1549`
- Wiener relRMSE: `0.1229`
- sparseFFT K=800 relRMSE: `0.1244`
- gains vs local ceiling: `-0.0320` and `-0.0305`

### 2. The gain survives in full-g two-channel reconstruction

High-k kernel branch:

- `outputs/runs/20251220_124104_e49_oos_twochannel_fullg_wiener_sparsefft/summary_e49_oos_twochannel_fullg_wiener_sparsefft.md`

High-k pixels branch:

- `outputs/runs/20251220_125130_e50_oos_twochannel_fullg_wiener_sparsefft_highkpixels/summary_e50_oos_twochannel_fullg_wiener_sparsefft_highkpixels.md`

Paper-facing outputs:

- `outputs/paper/main_table.md`
- `outputs/paper/short_results.md`
- `outputs/paper/main_figure.png`

Numbers carried into the paper bundle:

- kernel branch `B-A`: `-0.0308`
- pixels branch `B-A`: `-0.0316`

### 3. SparseFFT preserves most of the Wiener gain

Compression sweep:

- `outputs/runs/20251220_121206_e46_sparsefft_to_wiener_alpha2/summary_e46_sparsefft_to_wiener_alpha2.md`

OOS sparse table:

- `outputs/runs/20251220_123326_e48_oos_sparsefft_vs_wiener_alpha2/summary_e48_oos_sparsefft_vs_wiener_alpha2.md`
- `outputs/paper/sparsefft_table.md`
- `outputs/paper/sparsefft_curve.png`

Mode-stability check:

- `outputs/runs/20251220_122530_e47_sparsefft_mode_stability_alpha2/summary_e47_sparsefft_mode_stability.md`
- `outputs/paper/sparsefft_stability.md`

Important reading:

- `E46` supports the compression trade-off.
- `E47` supports the train-only mode-selection stability claim.
- `E48` confirms that K≈800 still beats the local ceiling out of sample.

### 4. Regime classification is near-perfect at the main threshold

Primary holdout tables:

- `outputs/runs/20251221_004201_e65_blend_stability_thresholds/summary_e65_blend_stability_thresholds.md`

Supplementary regime baseline:

- `outputs/runs/20251220_215420_e61_regime_gainlaw_stable/summary_e61_regime_gainlaw_stable.md`

Paper-facing outputs:

- `outputs/paper/main_table.md`
- `outputs/paper/short_results.md`

Numbers carried into the paper bundle at `thresh=0.02`:

- alpha AUROC: `0.989`
- bbks_ext AUROC: `1.000`
- bbks_tilt AUROC: `1.000`

### 5. The alpha strong-tail model becomes usable only after E66

Threshold-stability and tail-power evidence:

- `outputs/runs/20251221_004201_e65_blend_stability_thresholds/summary_e65_blend_stability_thresholds.md`
- `outputs/runs/20251221_021252_e66_alpha_tail_power/summary_e66_alpha_tail_power.md`

Paper-facing outputs:

- `outputs/paper/main_table.md`
- `outputs/paper/short_results.md`
- `outputs/paper/main_figure.png`

Numbers carried into the paper bundle at `thresh=0.02`:

- `E65` alpha `n_strong = 5.0`
- `E66` alpha `n_strong = 50.4`
- `E66` alpha global `R2 = 0.7940`, `MAE = 0.0131`
- `E66` alpha strong subset `R2 = 0.6140`, `MAE = 0.0146`

### 6. ROC figure provenance

The ROC panel embedded in `outputs/paper/main_figure.png` is sourced from:

- `outputs/runs/20251221_001325_e64_capped_weak_blend/roc_y_strong_alpha.png`

This is a figure-provenance source, not the main quantitative source for the AUROC table.

### 7. Sanity checks and compute note

Sanity checks are regenerated from stored summary tables by:

- `scripts/sanity_checks.py`
- source table: `outputs/runs/20251220_175718_e56_unified_gain_law/summary_e56_unified_gain_law.md`
- outputs: `outputs/paper/sanity_checks.md`, `outputs/paper/sanity_checks.png`

Compute note is regenerated by:

- `scripts/compute_cost.py`
- output: `outputs/paper/compute_cost.md`

These are publication-support artifacts, not central scientific claims.

## Practical verdict

- The stable paper line is tightly pinned to `E46-E50`, `E61-E66`, plus the `E64` ROC figure.
- The strongest claims in the manuscript are recoverable from stored local summaries and regenerated paper outputs.
- The broader `E70+` line is scientifically separate and should not be read as evidence for this same paper narrative.
