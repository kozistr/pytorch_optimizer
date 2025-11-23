.PHONY: format lint test check requirements visualize docs

FILES := pytorch_optimizer examples tests hubconf.py
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
	python -m pytest -p no:pastebin -p no:nose -p no:doctest --disable-warnings -sv -vv --cov=pytorch_optimizer --cov-report=xml ./tests

requirements:
	poetry export -f requirements.txt --output requirements.txt --without-hashes
	poetry export -f requirements.txt --output requirements-dev.txt --without-hashes --with dev

visualize:
	python -m examples.visualize_optimizers

docs:
	mkdocs serve
