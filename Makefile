.PHONY: init check format requirements build clean

init:
	python3 -m pip install -U pipenv setuptools
	python3 -m pipenv install --dev

check:
	isort --check-only --profile black -l 79 pytorch_optimizer setup.py
	black -S -l 79 --check pytorch_optimizer setup.py
	pylint pytorch_optimizer

build:
	python3 setup.py sdist bdist_wheel
	twine check dist/*

clean:
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]' `
	rm -f `find . -type f -name '*~' `
	rm -f `find . -type f -name '.*~' `
	rm -f `find . -type f -name '@*' `
	rm -f `find . -type f -name '#*#' `
	rm -f `find . -type f -name '*.orig' `
	rm -f `find . -type f -name '*.rej' `
	rm -rf build
	rm -rf dist

format:
	isort --profile black -l 79 pytorch_optimizer setup.py
	black -S -l 79 pytorch_optimizer setup.py

requirements:
	python3 -m pipenv lock -r > requirements.txt
	python3 -m pipenv lock -dr > requirements-dev.txt
