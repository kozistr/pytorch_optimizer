# Contributing

You can contribute code, tests, or documentation. Follow the existing implementations and keep your PR focused on
the problem you are solving.

## Setup and checks

Install [uv](https://docs.astral.sh/uv/), clone the repository, and install its development dependencies:

```shell
uv sync
```

Before submitting a pull request, run:

```shell
uv run just format
uv run just check
uv run just test
uv run coverage report -m
```

Cover new and changed implementation code at **100%**. Run tests on CPU by default. For GPU testing, use
`python -m pytest --device=cuda` with a CUDA-enabled PyTorch interpreter.

## Code

- Use single-quoted strings and a maximum line length of 119 characters. Separate logical steps with blank lines.
- Follow the original paper when implementing an optimizer and reuse existing `BaseOptimizer` helpers.
- Inherit from `BaseOptimizer` when adding an optimizer and register it in `OPTIMIZER_LIST`. Register new components
  in the appropriate package exports and update their README tables.

## Tests

Add `(optimizer_name, options, iterations)` to `OPTIMIZER_RECIPES` in [tests/recipes.py](tests/recipes.py)
for optimizer convergence tests. Add focused cases in `tests/optimizers/test_<module>.py` for numerical updates,
checkpoint restoration, or edge cases that the training recipes do not verify. Keep shared validation, variant,
wrapper, loss, and scheduler cases in their existing `tests/test_*.py` modules.

Reuse the model and parameter builders in [tests/fixtures.py](tests/fixtures.py) and helpers in
[tests/utils.py](tests/utils.py). Use a parameter for tests that call optimizer methods; use the shared model for
forward passes or model-based APIs. Group related cases in pytest classes where they share setup.

Use small, deterministic inputs and compare results with known values or a trusted reference. A focused update test
can use a parameter:

```python
import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def test_sgd_update():
    param = make_parameter((2,), grad=1.0)
    optimizer = build_optimizer('sgd', [param], lr=0.1)

    optimizer.step()

    torch.testing.assert_close(param, torch.tensor([-0.1, -0.1]))
```

Use `load_optimizer()` for constructor validation and `create_optimizer()` to test its public API. Check expected
exceptions with `pytest.raises`, and parametrize the options each optimizer supports. Check existing cases before
adding a test to avoid duplicating their assertions.

## Documentation

- Write API docstrings in Google style alongside the implementation.
- Write documentation pages in `docs/`, with usage examples in [docs/getting-started.md](docs/getting-started.md).
  Update `README.md` when adding components or changing public usage.
- Run `uv run just docs-build` and resolve build errors and warnings before submitting changes to the site.

## Pull requests

Describe the problem, the resulting behavior, and how you tested the change. Include the paper or reference
implementation for new optimizers and explain any algorithm differences. Report checks you could not run.

Start commit subjects with `feat:`, `fix:`, `perf:`, `style:`, `refactor:`, `docs:`, `chore:`, `build:`, or `update:`.
Use concise, specific commit subjects and PR titles.

If you use a coding agent or LLM, review its output and make sure you understand the implementation and tests.
You are responsible for the changes you submit.

For questions, open an issue or discussion, or ask in your pull request.
