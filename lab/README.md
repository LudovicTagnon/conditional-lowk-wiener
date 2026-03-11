# Lab Module Guide

This package contains the executable logic of the repository.

## Core runner path

- `run.py`: main experiment entry point used by `python3 -m lab.run --config ...`
- `data_synth.py`: synthetic field generation and patch sampling helpers
- `features.py`: feature builders
- `models.py`: model definitions and evaluation helpers
- `report.py`: lightweight markdown-friendly run summarizer
- `utils.py`: shared utilities

## Paper bundle synthesis

- `paper.py`: aggregates stored experiment outputs into `outputs/paper/`
- `draft.py`: renders the manuscript markdown from the paper bundle

These two modules are tied to the frozen paper structure and should be treated as the stable publication path.

## Extension-track helpers

- `postprocess_e118.py` through `postprocess_e163.py`: post-processors for the later `E70+` research line
- `e144_fast.py`: specialized helper for one late safety-track branch

In practice, the repository contains one stable paper path and a large extension layer. The `postprocess_e*.py` modules belong to that extension layer rather than to the paper nucleus.
