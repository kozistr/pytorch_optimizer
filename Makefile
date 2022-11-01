.PHONY: init format test check requirements

init:
	python -m pip install -q -U poetry
	python -m poetry install

format:
	isort --profile black -l 119 pytorch_optimizer tests lint.py hubconf.py
	black -S -l 119 pytorch_optimizer tests lint.py hubconf.py

test:
	python -m pytest -sv -vv --cov=pytorch_optimizer --cov-report=xml ./tests

check:
	isort --check-only --profile black -l 119 pytorch_optimizer tests lint.py hubconf.py
	black -S -l 119 --check pytorch_optimizer tests lint.py hubconf.py
	python lint.py

requirements:
	python -m poetry export -f requirements.txt --output requirements.txt --without-hashes
	python -m poetry export -f requirements.txt --output requirements-dev.txt --without-hashes --with dev
