set windows-shell := ['powershell.exe', '-NoLogo', '-Command']

files := 'pytorch_optimizer examples tests scripts hubconf.py'

format:
    ruff check --fix {{files}}

lint:
    ruff check {{files}}

check: lint
    pyright pytorch_optimizer examples

test:
    pytest -p no:pastebin -p no:nose -p no:doctest --disable-warnings --cov=pytorch_optimizer --cov-report=xml ./tests

requirements:
    uv export --no-dev > requirements.txt
    uv export --group dev > requirements-dev.txt

visualize:
    python -m examples.visualize_optimizers

docs:
    uv run --no-project --python 3.12 --with-requirements requirements-docs.txt zensical serve

docs-build:
    uv run --no-project --python 3.12 --with-requirements requirements-docs.txt zensical build --strict

update-docs:
    python scripts/update_docs.py

update-deps:
    uv sync --upgrade
