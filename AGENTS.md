# Agent Instructions

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidance.

## Workflow

- Use `uv`; prefer `uv run just <recipe>`.
- After code changes, run `uv run just format`, `uv run just check`, and `uv run just test`.
- Require 100% coverage for changed implementation code; check `uv run coverage report -m`.

## Implementation

- Follow existing code and typing style. Keep comments and abstractions minimal, and reuse helpers.
- Implement optimizer formulas from the original papers using `BaseOptimizer`, `init_group()`, and `step()`.
- Register and export new components; add one baseline training recipe for each new optimizer.
- Do not add implementation details or experiment results to documentation.
- Leave changelog updates to release automation.

## Testing

- Run existing tests and inspect coverage before adding cases or recipes.
- Add only for uncovered implementation or demonstrated regressions; explain the gap and prefer extending existing cases.
- Reuse shared fixtures, runners, and capability checks; avoid duplicate tests.
- Test observable results with deterministic inputs. Benchmark speed and report test results and coverage.

## Commits and PRs

- Keep commits focused and commit subjects and PR titles concise, specific, and imperative.
- Use `feat:`, `fix:`, `perf:`, `style:`, `refactor:`, `docs:`, `chore:`, `build:`, or `update:` for commit subjects.
- Check the branch and working tree before pushing; use `--force-with-lease` for pushed history rewrites.
