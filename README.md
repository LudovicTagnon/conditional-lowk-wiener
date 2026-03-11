# Mini-lab topo → gravité (CLI)

Objectif: tester si des features topologiques locales (`b0` multi-seuils) améliorent la prédiction du champ gravitationnel `g = ||∇Φ||` au-delà de la masse, sur données synthétiques (3D) et via un loader générique (2D).

## Lecture rapide du projet

- **Ligne stabilisée / papier**: bundle reproductible sous `outputs/paper/` et scripts de soumission dans `scripts/`.
- **Ligne expérimentale locale**: nombreuses configs additionnelles (`E70+`, SPARC, lensing, OOD/safety gates) et sorties d'audit plus récentes dans `outputs/`.
- **Point d'entrée scientifique**: `SCIENTIFIC_STATUS_2026-03-11.md` sépare la ligne papier stable des extensions locales.
- **Cartographie des tracks**: `EXPERIMENT_TRACKS_2026-03-11.md` et `OUTPUTS_INDEX_2026-03-11.md`.
- **Vérification papier**: `PAPER_VERIFICATION_2026-03-11.md`.
- **Reproduction papier**: `PAPER_REPRODUCTION_2026-03-11.md` détaille la chaîne de reconstruction propre du bundle.
- **Métadonnées locales**: les fichiers de conversation et de statut locaux sont des artefacts de workspace, pas des sources scientifiques.
- **Provenance papier**: `PAPER_EVIDENCE_MAP_2026-03-11.md` relie chaque claim stable aux runs et scripts exacts.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optionnel (versions figées):

```bash
python3 -m pip install -r requirements.lock
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

## Pistes présentes au-delà du noyau initial

Le dépôt local contient aussi des branches d'exploration supplémentaires, non couvertes par la synthèse minimale ci-dessus:

- comparaisons SPARC / RAR et variantes de lois effectives;
- tests de weak lensing (`data/lensing/`);
- familles OOD, conformal, veto, risk-gating et audits holdout plus récents;
- scripts de post-traitement `postprocess_e1xx.py` pour agréger ces campagnes.

En pratique: la ligne "papier" est la partie la plus propre et la plus stable; la ligne `E70+` est active mais plus mouvante.

## Index utiles par dossier

- `lab/README.md`: rôle des modules Python et des post-processors.
- `configs/README.md`: regroupement des familles de configs par phase scientifique.
- `data/README.md`: description des jeux SPARC et lensing.
- `scripts/README.md`: ordre recommandé des scripts de reproduction / audit.

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

## Artefacts de soumission (release + bundle)

- **Release GitHub**: `submission-v1` contient l’archive `submission_bundle.zip`.
- **Artefacts locaux**: `outputs/paper/` (bundle) et `outputs/paper/submission/` (pack prêt à upload).

## Régénérer le bundle papier (sans nouvelles expériences)

```bash
scripts/reproduce_paper.sh
scripts/build_pdf.sh
scripts/make_submission_bundle.sh
scripts/pre_submission_audit.sh
```

Ces scripts:
- régénèrent le bundle dans `outputs/paper/`,
- régénèrent aussi `sanity_checks.md` et `compute_cost.md`,
- génèrent `outputs/paper/draft.pdf` (ou HTML fallback),
- construisent le pack `outputs/paper/submission/`,
- vérifient les empreintes + la provenance PDF.

## Hygiène du dépôt local

- Les `outputs/` sont volumineux et ignorés par Git.
- Les fichiers de conversation et de statut locaux sont ignorés.
- Le fichier parasite `0.03.` était un artefact local accidentel; il n'est pas utilisé par le projet.
