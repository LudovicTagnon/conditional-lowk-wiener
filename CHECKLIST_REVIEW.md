# Review Checklist (E73)

- Claim: OOS low-k improves over ceiling by ~0.03 relRMSE; support: `outputs/paper/main_table.md` (E48).
- Claim: OOS full-g two-channel improves over ceiling by ~0.03 relRMSE; support: `outputs/paper/main_table.md` (E49/E50).
- Claim: Regime classifier AUROC ~1 at main thresh=0.02; support: `outputs/paper/main_table.md` (E65).
- Claim: Two-stage global R2/MAE at thresh=0.02; support: `outputs/paper/main_table.md` (E66/E65).
- Claim: Strong subset metrics only when n_strong>=30; support: `outputs/paper/main_table.md` and sensitivity appendix.
- Claim: Alpha tail power increases n_strong from E65 to E66; support: `outputs/paper/main_table.md` (E65/E66).
- Claim: SparseFFT K≈800 trade-off; support: `outputs/paper/sparsefft_table.md` (E48).
- Claim: SparseFFT modes stable across folds; support: `outputs/paper/sparsefft_stability.md` (E47).
- Figure: OOS deltas + ROC + tail stability; support: `outputs/paper/main_figure.png`.
- Figure: Method schematic; support: `outputs/paper/method_schematic.png` and `.svg`.
- Runs referenced in synthesis: see `configs/e68_threshold_policy.yaml` (E48/E49/E50/E61/E62/E65/E66 + E64 ROC).
- Note: no new experiments since E72; E70/E71/E73 are synthesis-only steps.
