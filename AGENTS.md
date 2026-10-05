# Agent Instructions

## Development workflow

- Use `uv` to resolve dependencies and run commands.
- Prefer `uv run just <recipe>` to use the versions in `uv.lock`.
- The repository `justfile` is cross-platform and works on Windows, macOS, and Linux/WSL.
- Run the relevant checks after changes:
  - `uv run just format`
  - `uv run just check`
  - `uv run just test`
- Cover new and changed implementation code at 100%. Check coverage with `uv run coverage report -m`.
- Use `uv run coverage report -m` to find uncovered lines.
- Add focused cases in the appropriate test module.
- Run focused optimizer tests with `uv run pytest tests/optimizers/test_muon.py -sv -vv`.
- Select a training recipe with `uv run pytest tests/test_optimizers.py::TestOptimizerTraining -k adammini -sv -vv`.
- Use `uv run just docs` to serve the Zensical documentation.
- Use `uv run just docs-build` to build the documentation with strict validation.
- Run `uv run just update-docs` after changing public exports to regenerate the API reference.

## Architecture

- `pytorch_optimizer/base/` contains `BaseOptimizer`, scheduler bases, shared types, and exceptions.
- `pytorch_optimizer/optimizer/` contains optimizers, wrappers, and shared gradient and update utilities.
- Put schedulers in `pytorch_optimizer/lr_scheduler/` and loss functions in `pytorch_optimizer/loss/`.
- Export public components from `pytorch_optimizer/__init__.py`.
- Use `load_optimizer()` to load a class by name.
- Use `create_optimizer()` to configure an optimizer with common options and wrappers.
- Optional integrations include `bitsandbytes`, `q-galore-torch`, and `torchao`.
- `tests/test_optimizers.py` runs shared training and interface tests using `OPTIMIZER_RECIPES` from
  `tests/recipes.py`. Focused optimizer and algorithm-helper cases live in `tests/optimizers/test_<module>.py`.
  Parameter validation, variants, wrappers, losses, and schedulers have separate shared test modules.
- `tests/fixtures.py` contains the shared training model and parameter builders.
- `tests/utils.py` contains helpers to construct optimizers and run training.
- `tests/conftest.py` contains pytest fixtures, including training data.
- `tests/optimizer_cases.py` derives constructor options and APIs that accept models from signatures.
  It detects sparse and complex support with cached CPU step probes.
  Explicit exclusions cover specialized protocols and numerical limitations.
- `zensical.toml` defines documentation navigation and theme settings.
- Keep the home page separate from `README.md`.

## Code style

- Follow nearby repository code and tests before introducing new patterns.
- Add comments only when code is difficult to understand.
- Avoid redundant, meaningless, or excessive comments.
- Use blank lines to separate logical steps and improve readability.
- Do not add introductory comments or docstrings to scripts.
- Use a maximum line length of 119 characters and single-quoted strings.
- Use the repository's existing typing style.
- Add `from __future__ import annotations` or `TYPE_CHECKING` imports only if the surrounding code requires them.
- Inherit from `BaseOptimizer` when you implement an optimizer.
- Implement `init_group()` and `step()`.
- Reuse the base class validation and update helpers where applicable.
- All optimizer implementations must follow the algorithms and update rules in their original papers.
- Follow Google-style docstrings.
- Reuse helpers before you add update or validation logic.
  Examples include `apply_weight_decay()`, `apply_cautious()`, `debias()`, `validate_learning_rate()`,
  `validate_betas()`, and `validate_range()`.
- Keep implementations focused.
- Remove redundant compatibility layers and abstractions.

## Testing

- Add or update minimal recipes in `OPTIMIZER_RECIPES` for optimizer changes, including affected variants.
  Each recipe contains an optimizer name, options, and an iteration count.
- Use the shared runner for optimizers that accept models if their update protocol fits.
- Avoid separate training or smoke tests that a recipe can replace.
- Add focused cases for behavior that training recipes do not check.
  Examples include numerical updates, checkpoint restoration, validation, wrappers, and edge cases.
- Follow nearby patterns and preserve 100% implementation coverage.
- Give focused cases an observable assertion.
- Compare numerical results with `torch.testing.assert_close()`.
- Check expected exceptions with `pytest.raises()`.
- Use stable inputs and a trusted reference for iterative algorithms.
- Reuse `TrainingModel`, `build_model()`, and parameter builders from `tests/fixtures.py`.
- Use a parameter if a test only calls optimizer methods.
- Create additional shapes or model features only if the behavior requires them.
- Use `tests.utils.build_optimizer()` for ordinary optimizer setup.
- Use `load_optimizer()` for constructor validation or APIs that accept an optimizer class.
- Use `create_optimizer()` to test the public factory.
- Use pytest classes to group related behavior where useful.
- Parametrize only applicable combinations.
- Run shared interface checks once per optimizer.
- Reuse capability detection from `tests/optimizer_cases.py` instead of maintaining optimizer name lists.
- Keep exclusions for specific protocols explicit.
- Preserve random state in probes and propagate unexpected failures.
- Keep variant, scheduler, and loss case data beside their tests.
- Copy recipe options before you attach callbacks or parameter groups.
- Avoid retaining model instances or optimizer state in shared case data.
- Keep tensors and training iteration counts small while retaining meaningful assertions.
- Remove redundant cases without losing behavior checks or coverage.
- Avoid session-wide performance overrides.

## Adding an optimizer

- Add the implementation under `pytorch_optimizer/optimizer/`.
- Export the optimizer from the optimizer package.
- Register it in `OPTIMIZER_LIST` to make it available through `OPTIMIZERS` and `load_optimizer()`.
- Add a training recipe and focused cases following the testing guidance above.
- Update the relevant documentation and the algorithm table in `README.md`.
- Write clear pull request titles and descriptions.
  The release workflow generates notes from merged pull requests.
  It updates `CHANGELOG.md`, `docs/changelogs/<tag>.md`, and the changelog index through an automated pull request.
- Register new loss functions and schedulers in their package exports.
- Add the components to the corresponding tests and README tables.
- Leave the root `CHANGELOG.md` to the release automation.

## Commits and pull requests

Start commit subjects with one of these prefixes:

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

- Write concise, specific pull request titles that describe the actual change.
- Use a title tag only if it adds useful context and matches the work.
- Do not derive a title tag from the commit prefix alone.
- Feature PR titles follow the existing format, such as `[Feature] Implement Magma optimizer`.
- Keep commits focused.
- Check the branch and working tree before you push.
- Use `--force-with-lease` when you need to rewrite a pushed branch.
