# Config Guide

This folder is best read in experiment families, not as one flat list.

## Early mini-lab baseline

- `e0` to `e3`
- synthetic 3D baseline, placebo, mass-binning, and 2D loader examples

## Locality / kernel / Fourier paper line

- roughly `e10` to `e50`
- low-k locality tests, kernel comparisons, Wiener reconstruction, sparse FFT, and two-channel full-g reconstruction

## Gain-law / regime paper synthesis

- roughly `e51` to `e68`
- BBKS families, gain-law stabilization, threshold policy, and the paper synthesis path

## Empirical extension line

- `e70` to `e87`
- SPARC, RAR, weak lensing, and related empirical-law experiments

## Safety / OOD / abstention line

- `e88` and later
- tail diagnostics, conformal gating, veto rules, holdout audits, and conservative abstention

## Reading rule

If the goal is the stable paper contribution, start from `e48` to `e68`. If the goal is extension work, use `EXPERIMENT_TRACKS_2026-03-11.md` to map the later config families to their output directories.
