# Weak Lensing Evidence Map

Date: 2026-03-12
Project: `topological_gravity_minilab`

This note promotes the weak-lensing extension into a readable evidence bundle.

## Canonical artifact set

- `outputs/E78_summary.md`
- `outputs/E78_20251223_150851_e78_verlinde_weak_lensing/E78_table.md`
- `outputs/E78_20251223_150851_e78_verlinde_weak_lensing/E78_overlay.png`
- `outputs/E78_20251223_150851_e78_verlinde_weak_lensing/config_used.yaml`

## Local scope

This track uses the Brouwer 2021 RAR weak-lensing tables and compares:

- `B0_baryons`
- `H2_verlinde`
- `H1_powerlaw_fit`

The local note explicitly says that a GR+NFW baseline is not provided in the tables, so the benchmark is internal to the saved artifact set.

## Main supported claims

### 1. The baryonic baseline performs very poorly on the local lensing benchmark

From `E78_table.md`:

- `B0_baryons`: `chi2 = 1809.05`, `chi2_dof = 120.60`, `pearson = 0.885`, `relRMSE = 1.298`

### 2. Effective-law variants drastically improve the fit

From `E78_table.md`:

- `H2_verlinde`: `chi2 = 137.62`, `chi2_dof = 9.17`, `pearson = 0.971`, `relRMSE = 0.303`
- `H1_powerlaw_fit`: `chi2 = 35.67`, `chi2_dof = 2.74`, `pearson = 0.983`, `relRMSE = 0.327`

Interpretation:

- both extension models strongly outperform `B0_baryons`
- within this saved local benchmark, `H1_powerlaw_fit` has the best chi-square, while `H2_verlinde` remains the more physics-branded effective-law candidate

## Scientific reading

- This is a real local fit improvement, not just a rhetorical extension.
- The benchmark remains narrower than a full cosmology-grade lensing validation because the local tables do not expose a proper GR+NFW reference inside the saved artifact set.
- The track is therefore promising, but still one step below a paper-grade external claim.
