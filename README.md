# Mini-lab topo → gravité (CLI)

Objectif: tester si des features topologiques locales (`b0` multi-seuils) améliorent la prédiction du champ gravitationnel `g = ||∇Φ||` au-delà de la masse, sur données synthétiques (3D) et via un loader générique (2D).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancer une expérience (commande unique)

```bash
python3 -m lab.run --config configs/e0_1overf_alpha2.yaml
```

Chaque run crée `outputs/runs/<YYYYMMDD_HHMMSS>_<expname>/` avec:
- `config_used.yaml`, `metrics.json`, `pred_vs_true.csv`
- `pred_vs_true.png`, `residuals_hist.png`
- + fichiers additionnels selon l’expérience (ex: E2).

## Analyse rapide des runs

Synthèse console (markdown-friendly):

```bash
python3 -m lab.report --runs outputs/runs --latest 5
# ou
python3 -m lab.report --path outputs/runs/<YYYYMMDD_HHMMSS>_<expname>
```

## Expériences fournies

- **E0** (synthétique 3D 1/f): génère `rho` 3D (bruit 1/f), résout Poisson périodique via FFT (`Φ_k = -ρ_k/k^2`, dérivées spectrales), sample des patches `9×9×9`, entraîne Ridge sur A/B/C.
- **E1** (placebo): identique à E0 mais shuffle des colonnes `b0_*` avant entraînement (le gain attendu doit disparaître).
- **E2** (mass-binning): identique à E0 (alpha=2.0) + métriques par quantiles de masse sur le test (`metrics_by_mass_bin.csv`, `metric_vs_mass_bin.png`).
- **E3** (loader réel 2D): charge `Sigma_norm.npy` et `g_proj.npy`, sample des patches `9×9`, même features/modèles.

## Run sequence (recommandé)

```bash
python3 -m lab.run --config configs/e0_1overf_alpha2.yaml --override n_patches=2000 --override grid_size=32 --override exp_name=e0_smoke
python3 -m lab.run --config configs/e1_placebo_shuffle.yaml --override n_patches=2000 --override grid_size=32 --override exp_name=e1_smoke
python3 -m lab.run --config configs/e2_mass_binning.yaml --override n_patches=5000 --override grid_size=48 --override exp_name=e2_smoke
python3 -m lab.run --config configs/e0_1overf_alpha15.yaml
python3 -m lab.run --config configs/e0_1overf_alpha2.yaml
python3 -m lab.run --config configs/e3_real2d_example.yaml  # nécessite de renseigner les paths .npy
```

## Paramètres utiles (dans les configs)

- `seed` (reproductibilité), `n_patches`, `patch_size`, `grid_size`, `alpha`
- `thresholds_b0: [0.4, 0.5, 0.6, 0.7]`
- `ridge_alpha` (régularisation), `test_frac` (split)

## Notes

- Les features sont standardisées avant Ridge.
- `pred_vs_true.csv` contient `y_true`, `split`, `mass`, et `y_pred_A/B/C` (une ligne par patch).
