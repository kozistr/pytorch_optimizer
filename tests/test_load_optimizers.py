from typing import List

import pytest

from pytorch_optimizer import load_optimizers


def ids(v) -> str:
    return f'{v[0].__name__}_{v[1:]}'


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

INVALID_OPTIMIZER_NAMES: List[str] = [
    'invalid',
    'asam',
    'sam',
    'pcgrad',
    'adamd',
    'lookahead',
    'chebyshev_schedule',
]


@pytest.mark.parametrize('valid_optimizer_names', OPTIMIZER_NAMES)
def test_load_optimizers_valid(valid_optimizer_names):
    load_optimizers(valid_optimizer_names)


@pytest.mark.parametrize('invalid_optimizer_names', INVALID_OPTIMIZER_NAMES)
def test_load_optimizers_invalid(invalid_optimizer_names):
    try:
        load_optimizers(invalid_optimizer_names)
    except NotImplementedError:
        return True
    return False
