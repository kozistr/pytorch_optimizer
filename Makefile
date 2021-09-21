.PHONY: init check format requirements

init:
	python3 -m pip install -U pipenv setuptools
	python3 -m pipenv install --dev

check:
	isort --check-only --profile black -l 79 pytorch_optimizer setup.py
	black -S -l 79 --check pytorch_optimizer setup.py
	pylint pytorch_optimizer

format:
	isort --profile black -l 79 pytorch_optimizer setup.py
	black -S -l 79 pytorch_optimizer setup.py

requirements:
	python3 -m pipenv lock -r > requirements.txt
	python3 -m pipenv lock -dr > requirements-dev.txt
