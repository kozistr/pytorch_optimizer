from typing import List

import pytest

from pytorch_optimizer import load_optimizers

VALID_OPTIMIZER_NAMES: List[str] = [
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

INVALID_OPTIMIZER_NAMES: List[str] = [
    'asam',
    'sam',
    'pcgrad',
    'adamd',
    'lookahead',
    'chebyshev_schedule',
]


@pytest.mark.parametrize('valid_optimizer_names', VALID_OPTIMIZER_NAMES)
def test_load_optimizers_valid(valid_optimizer_names):
    load_optimizers(valid_optimizer_names)


@pytest.mark.parametrize('invalid_optimizer_names', INVALID_OPTIMIZER_NAMES)
def test_load_optimizers_invalid(invalid_optimizer_names):
    with pytest.raises(NotImplementedError):
        load_optimizers(invalid_optimizer_names)


@pytest.mark.parametrize('optimizer_names', VALID_OPTIMIZER_NAMES)
def test_learning_rate(optimizer_names):
    with pytest.raises(ValueError):
        optimizer = load_optimizers(optimizer_names)
        optimizer(None, lr=-1e-2)


@pytest.mark.parametrize('optimizer_names', VALID_OPTIMIZER_NAMES)
def test_epsilon(optimizer_names):
    with pytest.raises(ValueError):
        optimizer = load_optimizers(optimizer_names)
        optimizer(None, eps=-1e-6)


@pytest.mark.parametrize('optimizer_names', VALID_OPTIMIZER_NAMES)
def test_weight_decay(optimizer_names):
    with pytest.raises(ValueError):
        optimizer = load_optimizers(optimizer_names)
        optimizer(None, weight_decay=-1e-3)
