.PHONY: format test check requirements visualize docs

format:
	ruff check --fix pytorch_optimizer examples tests hubconf.py
	black -S -l 119 pytorch_optimizer examples tests hubconf.py

check:
	black -S -l 119 --check pytorch_optimizer examples tests hubconf.py
	ruff check pytorch_optimizer examples tests hubconf.py
	pyright pytorch_optimizer examples tests

test:
	python -m pytest -p no:pastebin -p no:nose -p no:doctest -sv -vv --cov=pytorch_optimizer --cov-report=xml ./tests

requirements:
	poetry export -f requirements.txt --output requirements.txt --without-hashes
	poetry export -f requirements.txt --output requirements-dev.txt --without-hashes --with dev

visualize:
	python -m examples.visualize_optimizers

docs:
	mkdocs serve
