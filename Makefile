.PHONY: init format test check deploy requirements

init:
	python -m pip install -U poetry
	python -m poetry install

format:
	isort --profile black -l 119 pytorch_optimizer tests setup.py lint.py
	black -S -l 119 pytorch_optimizer tests setup.py lint.py

test:
	python -m pytest -sv -vv --cov=pytorch_optimizer --cov-report=xml ./tests

check:
	isort --check-only --profile black -l 119 pytorch_optimizer tests setup.py lint.py
	black -S -l 119 --check pytorch_optimizer tests setup.py lint.py
	python lint.py

deploy:
	python -m poetry publish --build

requirements:
	python -m poetry export -f requirements.txt --output requirements.txt --without-hashes
	python -m poetry export --dev -f requirements.txt --output requirements-dev.txt --without-hashes
