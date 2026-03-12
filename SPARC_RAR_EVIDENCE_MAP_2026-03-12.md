# SPARC / RAR Evidence Map

Date: 2026-03-12
Project: `topological_gravity_minilab`

This note promotes the strongest galaxy-law extension into a readable evidence bundle.

## Canonical artifact set

- `outputs/E76_summary.md`
- `outputs/E76_20251223_145149_e76_sparc_catalog_ylib/E76_table_models_fixedY.md`
- `outputs/E76_20251223_145149_e76_sparc_catalog_ylib/E76_table_models_freeY_prior.md`
- `outputs/E76_20251223_145149_e76_sparc_catalog_ylib/E76_residual_correlations.md`
- `outputs/E76_20251223_145149_e76_sparc_catalog_ylib/config_used.yaml`

## Local scope

This is not the low-k synthetic paper line. It is an empirical galaxy-law benchmark on the SPARC catalog with 175 galaxies and 5 folds.

## Main supported claims

### 1. The baryonic baseline is decisively outperformed

From `E76_table_models_fixedY.md`:

- `B0`: `nll_test_mean = 0.7773`, `mad_mean = 0.2141`, `relRMSE_log_mean = 0.9571`
- `H2_mond`: `-0.3219`, `0.0924`, `0.3664`
- `H3_fit`: `-0.3247`, `0.0942`, `0.3690`
- `H1_iso`: `-0.3153`, `0.0968`, `0.3704`

Interpretation:

- the effective-law family is not marginally better than `B0`; it is better by a large margin on every main error metric recorded locally.

### 2. The improvement is not just overfitting one nuisance parameter choice

From `E76_table_models_freeY_prior.md`:

- the free-Υ-with-priors table preserves essentially the same ranking as the fixed-Υ table in the local outputs

Interpretation:

- the gain does not disappear when the project switches from fixed mass-to-light ratios to the local prior-controlled free-Υ view.

### 3. Residual structure is strongly reduced relative to the baryonic baseline

From `E76_residual_correlations.md`:

- `B0` residuals correlate strongly with `gas_frac_proxy` (`0.608`) and `median_log_gbar` (`-0.656`)
- the best effective-law models collapse these correlations close to zero

Interpretation:

- in the local benchmark, the extension models do not only lower the scalar error; they also remove obvious residual systematics that remain in `B0`.

## Scientific reading

- This is the strongest empirical extension after the paper nucleus.
- The local outputs support model-ranking and residual-diagnostic claims.
- What is still missing is a paper-grade narrative and an explicit bridge from these effective laws back to the synthetic/paper line.
