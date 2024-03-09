.PHONY: init format test check requirements docs

init:
	python -m pip install -q -U poetry isort ruff pytest pytest-cov
	python -m poetry install --dev

format:
	isort --profile black -l 119 pytorch_optimizer tests hubconf.py
	ruff format pytorch_optimizer tests hubconf.py

test:
	python -m pytest -p no:pastebin -p no:nose -p no:doctest -sv -vv --cov=pytorch_optimizer --cov-report=xml ./tests

check:
	ruff check pytorch_optimizer tests hubconf.py

requirements:
	python -m poetry export -f requirements.txt --output requirements.txt --without-hashes
	python -m poetry export -f requirements.txt --output requirements-dev.txt --without-hashes --with dev

docs:
	mkdocs serve
