# Paper Reproduction

Date: 2026-03-11
Project: `topological_gravity_minilab`

This note records the clean reproduction path for the stable paper line from stored local artifacts.

## Recommended commands

### Rebuild paper-facing outputs

```bash
scripts/reproduce_paper.sh
```

This regenerates:

- `outputs/paper/main_table.md`
- `outputs/paper/main_figure.png`
- `outputs/paper/draft.md`
- `outputs/paper/method_schematic.png`
- `outputs/paper/method_schematic.svg`
- `outputs/paper/sparsefft_table.md`
- `outputs/paper/sparsefft_curve.png`
- `outputs/paper/sparsefft_stability.md`
- `outputs/paper/sanity_checks.md`
- `outputs/paper/sanity_checks.png`
- `outputs/paper/compute_cost.md`

### Render the draft PDF

```bash
scripts/build_pdf.sh
```

Preferred output:

- `outputs/paper/draft.pdf`

Fallback:

- `outputs/paper/draft.html`

### Assemble the submission directory

```bash
scripts/make_submission_bundle.sh
```

This populates:

- `outputs/paper/submission/`
- `outputs/paper/SUBMISSION_METADATA.md`

### Run the final audit

```bash
scripts/pre_submission_audit.sh
```

This is the end-to-end verification path for the paper bundle and PDF provenance.

## Scientific inputs

The paper bundle is synthesized from stored runs pinned in:

- `configs/e68_threshold_policy.yaml`

See:

- `PAPER_EVIDENCE_MAP_2026-03-11.md`
- `PAPER_VERIFICATION_2026-03-11.md`

## Important scope note

This reproduction path rebuilds the stable paper line only. It does not rerun the later `E70+` extensions, SPARC/lensing branches, or the 2026 safety audits.
