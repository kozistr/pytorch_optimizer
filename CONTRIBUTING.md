# Contributing

Contributions to `pytorch-optimizer` for code, documentation, and tests are always welcome!

## Setup

```bash
# Clone the repository
git clone https://github.com/kozistr/pytorch_optimizer.git
cd pytorch_optimizer

# Install dependencies using uv (recommended)
uv sync
```

## Development Commands

```bash
# Format code
uv run just format

# Lint code
uv run just lint

# Full check (lint + type checking)
uv run just check

# Run tests
uv run just test

# Run a specific test
uv run pytest tests/test_optimizers.py::test_name -sv -vv

# Serve documentation locally
uv run just docs

# Build documentation with warnings treated as errors
uv run just docs-build
```

## Code Style

- Line length: **119** characters
- Use **single quotes** for strings (not double quotes)
- Formatter and linter: **ruff**
- Docstring style: **Google style** ([example](https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/adamp.py#L14))

Run `uv run just format` and `uv run just check` before submitting a PR.

## Documentation

We use [Zensical](https://zensical.org/) with `mkdocstrings` for the API reference.
The `docs` and `docs-build` recipes install documentation dependencies in an isolated Python 3.12 environment.

- Edit navigation and theme settings in `zensical.toml`.
- Keep the documentation home page in `docs/index.md` and usage examples in `docs/getting-started.md`.
- Run `uv run just update-docs` after changing public exports to regenerate the optimizer, scheduler, and loss references.
- Indent nested changelog bullets by four spaces per level. Use `#123` for issue and PR references; the site links them to GitHub.
- Run `uv run just docs-build` before submitting documentation changes.

### Release notes

When a maintainer pushes a `vMAJOR.MINOR.PATCH` tag, the publish workflow asks GitHub to generate release notes from merged PRs.
It creates the GitHub release, then opens a PR to sync that release's notes into `CHANGELOG.md`,
`docs/changelogs/<tag>.md`, and the changelog index. The changelog PR uses the existing `automerge` label.
Rerunning the workflow updates the existing version section instead of adding it twice.

Write PR titles that explain the change to users. You do not need to maintain a separate changelog entry before a release.
If a maintainer edits the published release notes, rerun the changelog job to sync those edits.

## Adding New Optimizers, Loss Functions, or LR Schedulers

Reference existing implementations:
- Optimizers: `pytorch_optimizer/optimizer/`
- Loss functions: `pytorch_optimizer/loss/`
- LR schedulers: `pytorch_optimizer/lr_scheduler/`

### Checklist

1. Create a new file in the appropriate directory
2. For optimizers: inherit from `BaseOptimizer`, implement `init_group()` and `step()`
3. Utilize existing `BaseOptimizer` methods instead of reimplementing:
   - `apply_weight_decay()`, `apply_ams_bound()`, `apply_adam_debias()`
   - `debias()`, `debias_beta()`, `get_rectify_step_size()`
   - `apply_cautious()`, `get_adanorm_gradient()`
   - `validate_learning_rate()`, `validate_betas()`, `validate_range()`
4. Register in the corresponding `__init__.py` files
5. Run `uv run just format` and `uv run just check`
6. Add tests with **100% coverage** requirement
7. For new optimizers: add a minimal training recipe to `tests/constants.py` (see `OPTIMIZERS` list)
8. Describe the user-visible change in the PR title and description for the generated release notes.
9. Update `README.md`:
   - Update the count of optimizers/loss functions/schedulers
   - Add entry to the appropriate markdown table

## Testing

Tests are in `tests/` directory:
- `test_optimizers.py` - Main optimizer tests
- `test_optimizer_parameters.py` - Parameter validation tests
- `test_optimizer_variants.py` - Variant tests (Cautious, AdamD, etc.)
- `test_loss_functions.py` - Loss function tests
- `test_lr_schedulers.py` - Scheduler tests

100% test coverage is required.

## Questions

If you have any questions about contribution, please ask in the Issues, Discussions, or just in PR :)

Thank you!
