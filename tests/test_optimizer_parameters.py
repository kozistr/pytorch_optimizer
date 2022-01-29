from typing import List

import pytest

from pytorch_optimizer import load_optimizers

OPTIMIZER_NAMES: List[str] = [
    'adamp',
    'sgdp',
    'madgrad',
    'ranger',
    'ranger21',
    'radam',
    'adabound',
    'adahessian',
    'adabelief',
    'diffgrad',
    'diffrgrad',
    'lamb',
]

BETA_OPTIMIZER_NAMES: List[str] = [
    'adabelief',
    'adabound',
    'adahessian',
    'adamp',
    'diffgrad',
    'diffrgrad',
    'lamb',
    'radam',
    'ranger',
    'ranger21',
]


@pytest.mark.parametrize('optimizer_names', OPTIMIZER_NAMES)
def test_learning_rate(optimizer_names):
    with pytest.raises(ValueError):
        optimizer = load_optimizers(optimizer_names)
        optimizer(None, lr=-1e-2)


@pytest.mark.parametrize('optimizer_names', OPTIMIZER_NAMES)
def test_epsilon(optimizer_names):
    with pytest.raises(ValueError):
        optimizer = load_optimizers(optimizer_names)
        optimizer(None, eps=-1e-6)


@pytest.mark.parametrize('optimizer_names', OPTIMIZER_NAMES)
def test_weight_decay(optimizer_names):
    with pytest.raises(ValueError):
        optimizer = load_optimizers(optimizer_names)
        optimizer(None, weight_decay=-1e-3)


@pytest.mark.parametrize('optimizer_names', BETA_OPTIMIZER_NAMES)
def test_betas(optimizer_names):
    with pytest.raises(ValueError):
        optimizer = load_optimizers(optimizer_names)
        optimizer(None, betas=(-0.1, 0.1))

    with pytest.raises(ValueError):
        optimizer = load_optimizers(optimizer_names)
        optimizer(None, betas=(0.1, -0.1))
