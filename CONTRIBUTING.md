# Contributing

Contributions to `pytorch-optimizer` for code, documentation, and tests are always welcome!

## Setup

```bash
# Clone the repository
git clone https://github.com/kozistr/pytorch_optimizer.git
cd pytorch_optimizer

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e ".[dev]"
```

## Development Commands

```bash
# Format code
make format

# Lint code
make lint

# Full check (lint + type checking)
make check

# Run tests
make test

# Run a specific test
python -m pytest tests/test_optimizers.py::test_name -sv -vv

# Serve documentation locally
make docs
```

## Code Style

- Line length: **119** characters
- Use **single quotes** for strings (not double quotes)
- Formatter: **black** with `-S -l 119` flags
- Linter: **ruff**
- Docstring style: **Google style** ([example](https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/adamp.py#L14))

Run `make format` and `make check` before submitting a PR.

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
5. Run `make format` and `make check`
6. Add tests with **100% coverage** requirement
7. For new optimizers: add a minimal training recipe to `tests/constants.py` (see `OPTIMIZERS` list)
8. Add a short description to the latest changelog in `docs/changelogs/`
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
