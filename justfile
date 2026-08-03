set windows-shell := ['powershell.exe', '-NoLogo', '-Command']

files := 'pytorch_optimizer examples tests scripts hubconf.py'
black-flags := '-S -l 119'

format:
    ruff check --fix {{files}}
    black {{black-flags}} {{files}}

lint:
    black {{black-flags}} --check {{files}}
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
    mkdocs serve

update-docs:
    python scripts/update_docs.py

update-deps:
    uv sync --upgrade
