.PHONY: init check format requirements

init:
    python3 -m pip install -U pipenv setuptools
    python3 -m pipenv install --dev

check:
    isort --check-only --profile black pytorch_optimizer -l 79
    black -S -l 79 --check pytorch_optimizer
    pylint pytorch_optimizer

format:
    isort --profile black pytorch_optimizer -l 79
    black -S -l 79 pytorch_optimizer

requirements:
    python3 -m pipenv lock -r > requirements.txt
    python3 -m pipenv lock -dr > requirements-dev.txt
