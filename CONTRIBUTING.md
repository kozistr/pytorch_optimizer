# Contributing

Contribute code, tests, or documentation. Follow the existing implementations.
Keep each pull request focused on one problem.

## Setup and checks

1. Install [uv](https://docs.astral.sh/uv/).
2. Clone the repository.
3. Install the development dependencies from the repository root:

```shell
uv sync
```

Run these checks before you submit a pull request:

```shell
uv run just format
uv run just check
uv run just test
uv run coverage report -m
```

Cover new and changed implementation code at **100%**.
Run tests on CPU by default.
For GPU tests, run `python -m pytest --device=cuda` with a CUDA-enabled PyTorch interpreter.

## Code

- Use single-quoted strings and a maximum line length of 119 characters.
- Separate logical steps with blank lines.
- Follow the original paper when you implement an optimizer.
- Reuse existing `BaseOptimizer` helpers.
- Inherit from `BaseOptimizer` when you add an optimizer.
- Register the optimizer in `OPTIMIZER_LIST`.
- Register new components in the appropriate package exports.
- Update the corresponding README tables.

## Tests

Add `(optimizer_name, options, iterations)` to `OPTIMIZER_RECIPES` in [tests/recipes.py](tests/recipes.py) to test
optimizer convergence.
Add focused cases in `tests/optimizers/test_<module>.py` for behavior that training recipes do not check.
These cases include numerical updates, checkpoint restoration, and edge cases.
Keep shared validation, variant, wrapper, loss, and scheduler cases in their existing `tests/test_*.py` modules.

Reuse the model and parameter builders in [tests/fixtures.py](tests/fixtures.py).
Reuse the helpers in [tests/utils.py](tests/utils.py).
Use a parameter for tests that call optimizer methods.
Use the shared model for forward passes or APIs that accept a model.
Group related cases in pytest classes if they share setup.

Use small, deterministic inputs.
Compare results with known values or a trusted reference.
A focused update test can use a parameter:

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

- Use `load_optimizer()` for constructor validation.
- Use `create_optimizer()` to test the public factory API.
- Check expected exceptions with `pytest.raises`.
- Parametrize the options that each optimizer supports.
- Check existing cases before you add a test to avoid duplicate assertions.

## Documentation

- Write API docstrings in Google style alongside the implementation.
- Write documentation pages in `docs/`.
- Add usage examples to [docs/getting-started.md](docs/getting-started.md).
- Update `README.md` when you add components or change public usage.
- Run `uv run just docs-build`.
- Resolve build errors and warnings before you submit changes to the site.

## Pull requests

Describe the problem, the resulting behavior, and how you tested the change.
Include the paper or reference implementation for each new optimizer.
Explain any differences from the original algorithm.
Report checks you could not run.

Start commit subjects with `feat:`, `fix:`, `perf:`, `style:`, `refactor:`, `docs:`, `chore:`, `build:`, or `update:`.
Use concise, specific commit subjects and pull request titles.

Review code and tests from coding agents or large language models before you submit them.
Make sure you understand the implementation and tests.
You are responsible for the changes you submit.

Ask questions in an issue, a discussion, or your pull request.
