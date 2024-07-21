.PHONY: init format test check requirements docs

init:
	python -m pip install -q -U poetry isort black ruff pytest pytest-cov
	python -m poetry install --dev

format:
	isort --profile black -l 119 pytorch_optimizer examples tests hubconf.py
	black -S -l 119 pytorch_optimizer examples tests hubconf.py

test:
	python -m pytest -p no:pastebin -p no:nose -p no:doctest -sv -vv --cov=pytorch_optimizer --cov-report=xml ./tests

check:
	black -S -l 119 --check pytorch_optimizer examples tests hubconf.py
	ruff check pytorch_optimizer examples tests hubconf.py

requirements:
	poetry export -f requirements.txt --output requirements.txt --without-hashes
	poetry export -f requirements.txt --output requirements-dev.txt --without-hashes --with dev

docs:
	mkdocs serve
