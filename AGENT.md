# Agent Instructions

Read `CLAUDE.md` for the repository's detailed architecture, development commands,
and contribution guidance. The instructions below add the current project-specific
conventions and take precedence where they are more specific.

## Development workflow

- Use `uv` for dependency resolution and execution. Prefer `uv run just <recipe>` so tests use the versions in `uv.lock`.
- The repository `justfile` is cross-platform and works on Windows, macOS, and Linux/WSL.
- Run the relevant checks after changes:
  - `uv run just format`
  - `uv run just check`
  - `uv run just test`
- New or changed implementation code should have 100% test coverage. Check it with `uv run coverage report -m`.

## Code style

- Follow nearby repository code and tests before introducing new patterns.
- Use a maximum line length of 119 characters and single-quoted strings.
- Use the repository's existing typing style. Do not add `from __future__ import annotations` or `TYPE_CHECKING` imports unless the surrounding code requires them.
- Optimizers inherit from `BaseOptimizer`, implement `init_group()` and `step()`, and reuse its validation and update helpers where applicable.
- Keep implementations focused; remove redundant compatibility layers, comments, and abstractions.

## Adding an optimizer

- Add the implementation under `pytorch_optimizer/optimizer/`.
- Export it from the optimizer package and register it in `OPTIMIZER_LIST` so it is available through `OPTIMIZERS` and `load_optimizer()`.
- Add a training recipe to `tests/constants.py`; the parametrized tests in `tests/test_optimizers.py` will exercise it.
- Add focused tests for optimizer-specific behavior, state handling, validation, wrappers, and edge cases as needed.
- Update the relevant documentation and a versioned file under `docs/changelogs/`.
- Never edit the root `CHANGELOG.md`; it is maintained automatically.

## Commits and pull requests

- Use conventional commit prefixes without square brackets. Prefixes describe the kind of
  change, not just the files touched:
  - `feature: ...` — add user-visible functionality, such as the historical `feature: implement Magma optimizer` commit.
  - `fix: ...` — correct a bug or compatibility issue, such as `fix: prevent NaN in AdamP rsqrt`.
  - `docs: ...` — change documentation, such as `docs: update documentation`.
  - `style: ...` — make formatting or lint-only changes, such as `style: fix F401`.
  - `build(ci): ...` — change CI or release automation, such as `build(ci): fix release title`.
  - `build(deps): ...` — update dependencies or lockfiles, such as `build(deps): packages`.
- Feature PR titles follow the repository convention, such as `[Feature] Implement `Magma` optimizer`.
- Keep commits focused, verify the branch and working tree before pushing, and use `--force-with-lease` only when rewriting a pushed branch is necessary.
