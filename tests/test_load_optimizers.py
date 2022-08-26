from typing import List

import pytest

from pytorch_optimizer import get_supported_optimizers, load_optimizer

VALID_OPTIMIZER_NAMES: List[str] = [
    'adamp',
    'adan',
    'sgdp',
    'madgrad',
    'ranger',
    'ranger21',
    'radam',
    'adabound',
    'adabelief',
    'diffgrad',
    'diffrgrad',
    'lamb',
    'ralamb',
    'lars',
    'shampoo',
    'pnm',
    'adapnm',
    'nero',
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
    load_optimizer(valid_optimizer_names)


@pytest.mark.parametrize('invalid_optimizer_names', INVALID_OPTIMIZER_NAMES)
def test_load_optimizers_invalid(invalid_optimizer_names):
    with pytest.raises(NotImplementedError):
        load_optimizer(invalid_optimizer_names)


def test_get_supported_optimizers():
    assert len(get_supported_optimizers()) == 17
