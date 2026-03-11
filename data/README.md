# Data Guide

This repository currently carries two main non-synthetic data families.

## `data/sparc/`

SPARC rotation-curve files and catalog material used by the galaxy-law / RAR extension track.

Typical contents:

- `*_rotmod.dat` galaxy rotation-curve files
- `MassModels_Lelli2016c.mrt` catalog-level source table

## `data/lensing/`

Weak-lensing / RAR-side inputs used by the lensing extension track.

Current active subfolder:

- `brouwer2021_rar/`

Note:

- this folder contains some macOS resource-fork files named `._*`; they are workspace noise and should be ignored.

## What is not here

The stable paper nucleus is primarily synthetic and does not depend on large raw observational datasets inside `data/` beyond these extension-track inputs.
