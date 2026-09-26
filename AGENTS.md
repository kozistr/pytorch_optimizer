# Agent Instructions

## Development workflow

- Use `uv` for dependency resolution and execution. Prefer `uv run just <recipe>` so tests use the versions in `uv.lock`.
- The repository `justfile` is cross-platform and works on Windows, macOS, and Linux/WSL.
- Run the relevant checks after changes:
  - `uv run just format`
  - `uv run just check`
  - `uv run just test`
- New or changed implementation code should have 100% test coverage. Check it with `uv run coverage report -m`.
- If coverage drops, add focused tests using the existing patterns in `tests/constants.py` and `tests/test_*.py`.
- Run a focused test with `uv run pytest tests/test_optimizers.py::test_name -sv -vv`.
- Use `uv run just docs` to serve the Zensical documentation and `uv run just docs-build` to build it with strict validation.
- Run `uv run just update-docs` after changing public exports to regenerate the API reference.

## Architecture

- `pytorch_optimizer/base/` contains `BaseOptimizer`, scheduler bases, shared types, and exceptions.
- `pytorch_optimizer/optimizer/` contains optimizers, wrappers, and shared gradient and update utilities.
- `pytorch_optimizer/lr_scheduler/` and `pytorch_optimizer/loss/` contain schedulers and loss functions.
- The public API exports components from `pytorch_optimizer/__init__.py`. Use `load_optimizer()` to load a class by name
  and `create_optimizer()` to configure an optimizer with common options and wrappers.
- Optional integrations include `bitsandbytes`, `q-galore-torch`, and `torchao`.
- `tests/test_optimizers.py` uses recipes from `tests/constants.py` for training tests. Parameter validation, variants,
  wrappers, losses, and schedulers have separate test modules. `tests/conftest.py` supplies the training data fixture.
- `zensical.toml` owns documentation navigation and theme settings. Keep the home page separate from `README.md`.

## Code style

- Follow nearby repository code and tests before introducing new patterns.
- Add comments only when code is difficult to understand; avoid redundant, meaningless, or excessive comments.
- Use blank lines to separate logical steps and improve readability.
- Do not add introductory comments or docstrings to scripts.
- Use a maximum line length of 119 characters and single-quoted strings.
- Use the repository's existing typing style. Do not add `from __future__ import annotations` or `TYPE_CHECKING` imports unless the surrounding code requires them.
- Optimizers inherit from `BaseOptimizer`, implement `init_group()` and `step()`, and reuse its validation and update helpers where applicable.
- Follow Google-style docstrings. Reuse helpers such as `apply_weight_decay()`, `apply_cautious()`, `debias()`,
  `validate_learning_rate()`, `validate_betas()`, and `validate_range()` before adding update or validation logic.
- Keep implementations focused; remove redundant compatibility layers and abstractions.

## Adding an optimizer

- Add the implementation under `pytorch_optimizer/optimizer/`.
- Export it from the optimizer package and register it in `OPTIMIZER_LIST` so it is available through `OPTIMIZERS` and `load_optimizer()`.
- Add a training recipe to `tests/constants.py`; the parametrized tests in `tests/test_optimizers.py` will exercise it.
- Add focused tests for optimizer-specific behavior, state handling, validation, wrappers, and edge cases as needed.
- Update the relevant documentation and algorithm table in `README.md`.
- Write clear PR titles and descriptions; the release workflow generates notes from merged PRs and syncs them to
  `CHANGELOG.md`, `docs/changelogs/<tag>.md`, and the changelog index through an automated PR.
- Register new loss functions and schedulers in their package exports and add them to the corresponding tests and README tables.
- Never edit the root `CHANGELOG.md`; it is maintained automatically.

## Commits and pull requests

Every commit subject must start with one of these prefixes:

| Prefix | Use for |
| --- | --- |
| `feat:` | Add a feature or public API. |
| `fix:` | Correct a bug or unintended behavior. |
| `perf:` | Improve execution speed or resource use. |
| `style:` | Change code formatting without changing behavior. |
| `refactor:` | Restructure code without changing public behavior. |
| `docs:` | Add, revise, or reorganize documentation. |
| `chore:` | Maintain development tools, automation, or repository housekeeping. |
| `build:` | Change build, packaging, or dependency configuration. |
| `update:` | Refresh project content or metadata when no more specific prefix applies. |

Keep the text after the prefix imperative, concise, and specific to the committed change.
Do not use commit prefixes outside this list.

- Feature PR titles follow the existing format, such as `[Feature] Implement Magma optimizer`.
- Keep commits focused, verify the branch and working tree before pushing, and use `--force-with-lease` only when rewriting a pushed branch is necessary.
