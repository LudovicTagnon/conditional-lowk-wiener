# Repository Guidelines

This repository directory is currently empty (no source files or Git history detected). Use the conventions below as the starting point for adding code, notes, and tooling so contributors can work consistently.

## Project Structure & Module Organization

- Keep top-level organization predictable:
  - `src/` for application/library code.
  - `tests/` for automated tests.
  - `docs/` for long-form documentation (design notes, theory, references).
  - `assets/` for figures and other static files referenced by docs.
- Prefer small, focused modules and clear boundaries (e.g., `src/<domain>/…` rather than one large file).

## Build, Test, and Development Commands

No standard build/test commands are configured yet. If you introduce tooling, add a single entry point and document it:

- `make test` (or `just test`): run the full test suite.
- `make fmt`: auto-format code.
- `make lint`: run linters/static checks.
- `make run`: run the primary executable/script locally.

Example (recommended): add a `Makefile` that wraps language-specific commands so contributors don’t need to memorize them.

## Coding Style & Naming Conventions

- Use UTF-8 text, Unix newlines (`\n`), and avoid generated files in commits unless required.
- Follow the idiomatic formatter/linter for the chosen language (e.g., `black`/`ruff` for Python, `prettier` for JS/TS, `rustfmt` for Rust).
- Name files and modules descriptively; avoid abbreviations. Use `kebab-case` for Markdown filenames and `snake_case`/`PascalCase` following language norms.

## Testing Guidelines

- Add tests for new behavior and bug fixes; keep them deterministic and fast.
- Mirror the source layout (e.g., `tests/<module>/test_<feature>.*`) and prefer clear, behavior-based test names.

## Commit & Pull Request Guidelines

No commit conventions can be inferred yet. Until a history exists, use Conventional Commits:
- `feat: …`, `fix: …`, `docs: …`, `test: …`, `refactor: …`, `chore: …`

PRs should include: a short summary, rationale, how to verify (commands), and screenshots for doc/asset changes when relevant.

## Agent-Specific Instructions

- Keep patches minimal and scoped; avoid drive-by refactors.
- Update `docs/` (or add it) when introducing new concepts, assumptions, or workflows.
