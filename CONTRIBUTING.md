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

**Add a test or recipe only for uncovered implementation changes or a demonstrated regression.**
An implementation change does not require a new test or recipe.
Reuse existing tests for refactors and performance changes when they check the behavior and preserve 100% coverage.

### Decide whether to add a case

Before editing tests:

1. Read the relevant shared tests, focused tests, and recipes.
2. Run the existing suite with `uv run just test` against the implementation change.
3. Run `uv run coverage report -m` to locate uncovered implementation lines.
4. Identify any incorrect result or failure that existing assertions would miss.

For a regression, demonstrate that the case fails without the fix and passes with it.
Line coverage alone does not prove correctness.
A demonstrated regression can justify a case without increasing coverage.

Identify the uncovered lines or regression for each proposed addition.
Explain why existing tests do not already cover the gap.
Extend an existing case when its purpose remains clear.
Otherwise, add the smallest focused case that closes the gap.
Leave tests and recipes unchanged when you find no gap.

### Use shared tests and recipes

The shared tests already cover training, validation, variants, checkpoints, wrappers, and optimizer interfaces.
Check their assertions before adding those checks to `tests/optimizers/test_<module>.py`.
Keep shared cases in their existing `tests/test_*.py` modules.
Run shared interface checks once per optimizer.
Reuse capability detection from [tests/optimizer_cases.py](tests/optimizer_cases.py).
Do not maintain separate optimizer name lists for shared checks.

Add one baseline `(optimizer_name, options, iterations)` recipe for a new optimizer in
[tests/recipes.py](tests/recipes.py).
For an existing optimizer, change or add a recipe only when a specific coverage gap or regression requires training.
Do not add a recipe for each modified optimizer, option, dtype, or variant.
Inspect `TRAINING_CASES` in [tests/test_optimizers.py](tests/test_optimizers.py) first.
Existing recipes already generate dtype and foreach variants.
Use the shared runner for model-based optimizers when their update protocol fits.
Do not add standalone convergence or smoke tests that the shared runner already covers.

### Keep focused cases small

- Give each case an assertion that detects the identified failure.
- Test observable results.
  Do not add assertions about helper call counts or private batch layouts to test an optimization.
- Measure speed with benchmarks. Keep timing thresholds out of ordinary unit tests.
- Parametrize only combinations that expose distinct gaps.
  Avoid Cartesian products without a reason for each combination.
- Reuse models and parameter builders from [tests/fixtures.py](tests/fixtures.py).
  Reuse helpers from [tests/utils/](tests/utils/).
- Use `build_optimizer()` for ordinary setup.
  Use `load_optimizer()` for constructor validation.
  Use `create_optimizer()` for factory tests.
- Use a parameter when the test only calls optimizer methods.
  Use a model when the tested API requires one.
- Add shapes or model features only when the identified gap requires them.
- Use small, deterministic inputs and the minimum meaningful training iterations.
- Compare numerical results with `torch.testing.assert_close()`. Use known values or a trusted reference.
- Check expected exceptions with `pytest.raises()`.
- Group related cases in pytest classes when they share setup.
- Keep variant, scheduler, and loss case data beside their tests.
- Copy recipe options before modification.
  Do not retain model instances or optimizer state in shared case data.
- Preserve random state in probes and propagate unexpected failures.
  Keep protocol exclusions explicit.
- Avoid session-wide performance overrides.

### Review test additions

Run the required checks and inspect the final coverage report.
Remove additions that repeat existing checks without detecting a distinct failure
or covering missing implementation code.
Preserve existing regression assertions and 100% implementation coverage.
Report test results and coverage in the pull request.
For each added case or recipe, identify the gap it closes and why the existing suite was insufficient.

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
