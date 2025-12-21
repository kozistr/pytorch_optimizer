# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

```bash
# Install dependencies (using poetry)
poetry install

# Run tests with coverage
make test
# Or directly:
python -m pytest -p no:pastebin -p no:nose -p no:doctest --disable-warnings -sv -vv --cov=pytorch_optimizer --cov-report=xml ./tests

# Run a single test file
python -m pytest tests/test_optimizers.py -sv -vv

# Run a specific test
python -m pytest tests/test_optimizers.py::test_name -sv -vv

# Format code
make format

# Lint code
make lint

# Full check (lint + type checking)
make check

# Update documentation
make update-docs
# Or: python scripts/update_docs.py

# Serve documentation locally
make docs
```

## Code Style

- Line length: **119** characters
- Use **single quotes** for strings (not double quotes)
- Formatter: **black** with `-S -l 119` flags
- Linter: **ruff**
- Docstring style: **Google style**, not reST or NumPy style

## Architecture Overview

### Package Structure

- `pytorch_optimizer/` - Main package
  - `base/` - Base classes and types
    - `optimizer.py` - `BaseOptimizer` class that all optimizers inherit from
    - `type.py` - Type definitions (Parameters, Closure, Loss, etc.)
    - `exception.py` - Custom exceptions
    - `scheduler.py` - Base scheduler class
  - `optimizer/` - All optimizer implementations (130+ optimizers)
    - Each optimizer is in its own file (e.g., `adamp.py`, `lion.py`, `adopt.py`)
    - `utils.py` - Shared utilities for optimizers
    - `gradient_centralization.py` - GC implementation
    - `agc.py` - Adaptive Gradient Clipping
    - `lookahead.py` - Lookahead wrapper
    - `experimental/` - Experimental optimizers
  - `lr_scheduler/` - Learning rate schedulers
  - `loss/` - Loss function implementations

### Key Base Class: `BaseOptimizer`

All optimizers inherit from `BaseOptimizer` (in `base/optimizer.py`), which provides:
- Static methods for common operations: `apply_weight_decay`, `debias`, `get_rectify_step_size`, `apply_cautious`
- Validation methods: `validate_range`, `validate_betas`, `validate_learning_rate`
- AMSBound, AdaNorm, and other variant implementations

When implementing a new optimizer:
1. Inherit from `BaseOptimizer`
2. Implement `init_group(self, group: ParamGroup, **kwargs) -> None` for state initialization
3. Implement `step(self, closure: Closure = None) -> Loss`

### Adding New Optimizers, Loss Functions, or LR Schedulers

Reference existing implementations for patterns:
- Optimizers: `pytorch_optimizer/optimizer/`
- Loss functions: `pytorch_optimizer/loss/`
- LR schedulers: `pytorch_optimizer/lr_scheduler/`

**Checklist:**

1. Create a new file in the appropriate directory
2. For optimizers: inherit from `BaseOptimizer`, implement `init_group()` and `step()`
3. Utilize existing `BaseOptimizer` methods instead of reimplementing:
   - `apply_weight_decay()`, `apply_ams_bound()`, `apply_adam_debias()`
   - `debias()`, `debias_beta()`, `get_rectify_step_size()`
   - `apply_cautious()`, `get_adanorm_gradient()`
   - `validate_learning_rate()`, `validate_betas()`, `validate_range()`
4. Register in the corresponding `__init__.py` files
5. Run `make format` and `make check` to ensure strict style compliance
6. Add tests with **100% coverage** requirement
7. For new optimizers: add a minimal training recipe to `tests/constants.py` (see `OPTIMIZERS` list)
8. Add a short description to the latest changelog in `docs/changelogs/`
9. Update `README.md`:
   - Update the count of optimizers/loss functions/schedulers
   - Add entry to the appropriate markdown table with format:
     `| Name | Paper full title | [github](link) | [paper](arxiv_link) | [cite](citation_link) |`

### Public APIs

Load optimizers dynamically:
```python
from pytorch_optimizer import load_optimizer, get_supported_optimizers
optimizer = load_optimizer('adamp')(model.parameters())
```

Create optimizer with common options:
```python
from pytorch_optimizer import create_optimizer
optimizer = create_optimizer(model, 'adamp', lr=1e-3, use_gc=True, use_lookahead=True)
```

## Testing

Tests are in `tests/` directory:
- `test_optimizers.py` - Main optimizer tests
- `test_optimizer_parameters.py` - Parameter validation tests
- `test_optimizer_variants.py` - Variant tests (Cautious, AdamD, etc.)
- `test_loss_functions.py` - Loss function tests
- `test_lr_schedulers.py` - Scheduler tests

The `conftest.py` provides a `environment` fixture with sample data for training tests.

## External Optimizer Support

The package wraps optimizers from external libraries:
- `bitsandbytes` - 8-bit optimizers (`load_bnb_optimizer`)
- `q-galore-torch` - Q-GaLore optimizers (`load_q_galore_optimizer`)
- `torchao` - TorchAO optimizers (`load_ao_optimizer`)
