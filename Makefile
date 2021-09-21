.PHONY: init check format requirements

init:
	python3 -m pip install -U pipenv setuptools
	python3 -m pipenv install --dev

check:
	isort --check-only --profile black rubik_cube -l 79
	black -S -l 79 --check rubik_cube
	pylint rubik_cube

format:
	isort --profile black rubik_cube -l 79
	black -S -l 79 rubik_cube

requirements:
	python3 -m pipenv lock -r > requirements.txt
	python3 -m pipenv lock -dr > requirements-dev.txt
