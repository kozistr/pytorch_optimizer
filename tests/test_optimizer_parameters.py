from typing import List

import pytest
from torch import nn

from pytorch_optimizer import SAM, Lookahead, PCGrad, load_optimizers
from tests.utils import Example

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
    'ralamb',
    'lars',
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
    'ralamb',
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


@pytest.mark.parametrize('optimizer_names', ['adamp', 'sgdp'])
def test_wd_ratio(optimizer_names):
    with pytest.raises(ValueError):
        optimizer = load_optimizers(optimizer_names)
        optimizer(None, wd_ratio=-1e-3)


@pytest.mark.parametrize('optimizer_names', ['lars'])
def test_trust_coefficient(optimizer_names):
    with pytest.raises(ValueError):
        optimizer = load_optimizers(optimizer_names)
        optimizer(None, trust_coefficient=-1e-3)


@pytest.mark.parametrize('optimizer_names', ['madgrad', 'lars'])
def test_momentum(optimizer_names):
    with pytest.raises(ValueError):
        optimizer = load_optimizers(optimizer_names)
        optimizer(None, momentum=-1e-3)


@pytest.mark.parametrize('optimizer_names', ['ranger'])
def test_lookahead_k(optimizer_names):
    with pytest.raises(ValueError):
        optimizer = load_optimizers(optimizer_names)
        optimizer(None, k=-1)


@pytest.mark.parametrize('optimizer_names', ['adahessian'])
def test_hessian_power(optimizer_names):
    with pytest.raises(ValueError):
        optimizer = load_optimizers(optimizer_names)
        optimizer(None, hessian_power=-1e-3)


@pytest.mark.parametrize('optimizer_names', ['ranger21'])
def test_beta0(optimizer_names):
    optimizer = load_optimizers(optimizer_names)

    with pytest.raises(ValueError):
        optimizer(None, num_iterations=200, beta0=-0.1)


@pytest.mark.parametrize('optimizer_names', BETA_OPTIMIZER_NAMES)
def test_betas(optimizer_names):
    optimizer = load_optimizers(optimizer_names)

    with pytest.raises(ValueError):
        optimizer(None, betas=(-0.1, 0.1))

    with pytest.raises(ValueError):
        optimizer(None, betas=(0.1, -0.1))


@pytest.mark.parametrize('optimizer_names', ['pcgrad'])
def test_reduction(optimizer_names):
    model: nn.Module = Example()
    parameters = model.parameters()
    optimizer = load_optimizers('adamp')(parameters)

    with pytest.raises(ValueError):
        PCGrad(optimizer, reduction='wrong')


def test_sam_parameters():
    with pytest.raises(ValueError):
        SAM(None, load_optimizers('adamp'), rho=-0.1)


def test_lookahead_parameters():
    model: nn.Module = Example()
    parameters = model.parameters()
    optimizer = load_optimizers('adamp')(parameters)

    pullback_momentum_list: List[str] = ['none', 'reset', 'pullback']
    for pullback_momentum in pullback_momentum_list:
        opt = Lookahead(optimizer, pullback_momentum=pullback_momentum)
        opt.load_state_dict(opt.state_dict())

    with pytest.raises(ValueError):
        Lookahead(optimizer, k=0)

    with pytest.raises(ValueError):
        Lookahead(optimizer, alpha=-0.1)

    with pytest.raises(ValueError):
        Lookahead(optimizer, pullback_momentum='invalid')
