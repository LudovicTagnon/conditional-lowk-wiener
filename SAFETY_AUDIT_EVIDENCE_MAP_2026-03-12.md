# Safety Audit Evidence Map

Date: 2026-03-12
Project: `topological_gravity_minilab`

This note promotes the 2026 abstention / safety line into a readable evidence bundle.

## Canonical artifact set

- `outputs/E160_20260102_001116_e160_fixed_threshold_audit/E160_summary.md`
- `outputs/E161_20260102_001542_e161_fixed_conservative/E161_summary.md`
- `outputs/E161_20260102_001542_e161_fixed_conservative/E161_best_params.json`
- `outputs/E163_20260102_002212_e163_bulk_audit/E163_summary.md`
- `outputs/E163_20260102_002212_e163_bulk_audit/E163_rule_card_bulk_only.txt`

## Local scope

This is not a raw-gain track. It is a safety-under-domain-shift track for deciding when to veto the calibrated model back to Wiener.

## Main supported claims

### 1. The fixed-threshold audit fails in its pre-conservative form

From `E160_summary.md`:

- `max worst_delta = 0.0365`
- `mean coverage = 0.8657`
- `FAIL: unsafe survivors in TEST-only autopsy (n=3)`

Interpretation:

- the earlier gate is not deployable as-is because unsafe survivors remain in the test autopsy.

### 2. Conservative percentile thresholds remove unsafe survivors

From `E161_summary.md` and `E161_best_params.json`:

- mode: `AND`
- percentiles: `p_std = 0.98`, `p_q99 = 0.98`, `p_shape = 0.99`
- `max worst_delta_test = 0.0280`
- `mean coverage_test = 0.8333`
- `PASS: no unsafe survivors in TEST-only autopsy`

Interpretation:

- the pass point is achieved by tightening the thresholds, not by improving raw coverage.

### 3. The deployable rule is simple and survives the bulk-only audit

From `E163_summary.md` and `E163_rule_card_bulk_only.txt`:

- `safe_bulk_only` and `safe_full` have identical mean metrics in the saved audit
- deployable rule:
  - use CAL unless `(bulk_std >= -2.6155752642788253) AND (bulk_q99 >= -2.6106704728174304)`
  - then veto to Wiener

Interpretation:

- the active repository does contain a compact, readable deployable rule card
- the best local safety result is therefore not just a table; it is a decision rule with explicit thresholds

### 4. Holdout coverage remains highly uneven

From `E161_summary.md` and `E163_summary.md`:

- holdout `A`: coverage around `0.39–0.42`
- holdout `B`: coverage around `0.0185`
- holdout `C`: coverage around `0.9444`

Interpretation:

- the gate is safer, but not uniformly usable across holdout groups
- the right claim is conservative safety, not broad acceptance

## Scientific reading

- `E160` is the failure checkpoint.
- `E161` is the conservative pass checkpoint.
- `E163` is the compact deployable summary.
- This is the strongest late-stage extension line if the research goal is abstention and OOD safety rather than physical law fitting.
