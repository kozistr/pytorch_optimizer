.PHONY: format lint test check requirements visualize docs update-docs

FILES := pytorch_optimizer examples tests scripts hubconf.py
BLACK_FLAGS := -S -l 119

format:
	ruff check --fix $(FILES)
	black $(BLACK_FLAGS) $(FILES)

lint:
	black $(BLACK_FLAGS) --check $(FILES)
	ruff check $(FILES)

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
