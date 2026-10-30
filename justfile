set windows-shell := ['powershell.exe', '-NoLogo', '-Command']

files := 'pytorch_optimizer examples tests scripts hubconf.py'

format:
    ruff check --fix {{files}}

lint:
    ruff check {{files}}

check: lint
    ty check

test:
    pytest

requirements:
    uv export --no-dev > requirements.txt
    uv export --group dev > requirements-dev.txt

visualize:
    python -m examples.visualize_optimizers

docs:
    uv run --no-project --python 3.12 --with-requirements docs/requirements-docs.txt zensical serve

docs-build:
    uv run --no-project --python 3.12 --with-requirements docs/requirements-docs.txt zensical build --strict

update-docs:
    python scripts/update_docs.py

update-deps:
    uv sync --upgrade
