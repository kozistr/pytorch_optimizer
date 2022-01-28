.PHONY: init format check test build deploy requirements

init:
	python3 -m pip install -U pipenv setuptools
	python3 -m pipenv install --dev

format:
	isort --profile black -l 119 pytorch_optimizer setup.py lint.py
	black -S -l 119 pytorch_optimizer setup.py lint.py

check:
	isort --check-only --profile black -l 119 pytorch_optimizer tests setup.py lint.py
	black -S -l 119 --check pytorch_optimizer tests setup.py lint.py
	python3 lint.py
	pytest -vv --forked --durations=0 ./tests

test:
	pytest -vv --forked --durations=0 ./tests

build:
	python3 setup.py sdist bdist_wheel

deploy:
	python3 -m twine check dist/*
	python3 -m twine upload dist/*

requirements:
	python3 -m pipenv lock -r > requirements.txt
	python3 -m pipenv lock -dr > requirements-dev.txt
