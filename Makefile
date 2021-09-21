.PHONY: init check format requirements build deploy

init:
	python3 -m pip install -U pipenv setuptools wheel
	python3 -m pipenv install --dev

check:
	isort --check-only --profile black -l 79 pytorch_optimizer setup.py
	black -S -l 79 --check pytorch_optimizer setup.py
	pylint pytorch_optimizer

build:
	python3 setup.py sdist bdist_wheel

deploy:
	python3 -m twine check dist/*
	python3 -m twine upload dist/*

format:
	isort --profile black -l 79 pytorch_optimizer setup.py
	black -S -l 79 pytorch_optimizer setup.py

requirements:
	python3 -m pipenv lock -r > requirements.txt
	python3 -m pipenv lock -dr > requirements-dev.txt
