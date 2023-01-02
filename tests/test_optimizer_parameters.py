from typing import List

import pytest
import torch
from torch import nn

from pytorch_optimizer import SAM, Lookahead, PCGrad, Ranger21, SafeFP16Optimizer, load_optimizer
from pytorch_optimizer.base.exception import ZeroParameterSizeError
from tests.utils import Example

OPTIMIZER_NAMES: List[str] = [
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
    'pnm',
    'adapnm',
    'adai',
]
BETA_OPTIMIZER_NAMES: List[str] = [
    'adabelief',
    'adabound',
    'adamp',
    'diffgrad',
    'diffrgrad',
    'lamb',
    'radam',
    'ranger',
    'ranger21',
    'ralamb',
    'pnm',
    'adapnm',
    'adan',
    'adai',
]


@pytest.mark.parametrize('optimizer_name', OPTIMIZER_NAMES + ['nero'])
def test_learning_rate(optimizer_name):
    optimizer = load_optimizer(optimizer_name)

    with pytest.raises(ValueError):
        if optimizer_name == 'ranger21':
            optimizer(None, num_iterations=100, lr=-1e-2)
        else:
            optimizer(None, lr=-1e-2)


@pytest.mark.parametrize('optimizer_name', OPTIMIZER_NAMES)
def test_epsilon(optimizer_name):
    optimizer = load_optimizer(optimizer_name)

    with pytest.raises(ValueError):
        if optimizer_name == 'ranger21':
            optimizer(None, num_iterations=100, eps=-1e-6)
        else:
            optimizer(None, eps=-1e-6)


@pytest.mark.parametrize('optimizer_name', OPTIMIZER_NAMES)
def test_weight_decay(optimizer_name):
    optimizer = load_optimizer(optimizer_name)

    with pytest.raises(ValueError):
        if optimizer_name == 'ranger21':
            optimizer(None, num_iterations=100, weight_decay=-1e-3)
        else:
            optimizer(None, weight_decay=-1e-3)


@pytest.mark.parametrize('optimizer_name', ['adamp', 'sgdp'])
def test_wd_ratio(optimizer_name):
    optimizer = load_optimizer(optimizer_name)
    with pytest.raises(ValueError):
        optimizer(None, wd_ratio=-1e-3)


@pytest.mark.parametrize('optimizer_name', ['lars'])
def test_trust_coefficient(optimizer_name):
    optimizer = load_optimizer(optimizer_name)
    with pytest.raises(ValueError):
        optimizer(None, trust_coefficient=-1e-3)


@pytest.mark.parametrize('optimizer_name', ['madgrad', 'lars'])
def test_momentum(optimizer_name):
    optimizer = load_optimizer(optimizer_name)
    with pytest.raises(ValueError):
        optimizer(None, momentum=-1e-3)


@pytest.mark.parametrize('optimizer_name', ['ranger'])
def test_lookahead_k(optimizer_name):
    optimizer = load_optimizer(optimizer_name)
    with pytest.raises(ValueError):
        optimizer(None, k=-1)


@pytest.mark.parametrize('optimizer_name', ['ranger21'])
def test_beta0(optimizer_name):
    optimizer = load_optimizer(optimizer_name)
    with pytest.raises(ValueError):
        optimizer(None, num_iterations=200, beta0=-0.1)


@pytest.mark.parametrize('optimizer_name', ['nero'])
def test_beta(optimizer_name):
    optimizer = load_optimizer(optimizer_name)
    with pytest.raises(ValueError):
        optimizer(None, beta=-0.1)


@pytest.mark.parametrize('optimizer_name', BETA_OPTIMIZER_NAMES)
def test_betas(optimizer_name):
    optimizer = load_optimizer(optimizer_name)

    with pytest.raises(ValueError):
        if optimizer_name == 'ranger21':
            optimizer(None, num_iterations=100, betas=(-0.1, 0.1))
        elif optimizer not in ('adapnm', 'adan'):
            optimizer(None, betas=(-0.1, 0.1))

    with pytest.raises(ValueError):
        if optimizer_name == 'ranger21':
            optimizer(None, num_iterations=100, betas=(0.1, -0.1))
        elif optimizer not in ('adapnm', 'adan'):
            optimizer(None, betas=(0.1, -0.1))

    if optimizer_name in ('adapnm', 'adan'):
        with pytest.raises(ValueError):
            optimizer(None, betas=(0.1, 0.1, -0.1))


def test_reduction():
    parameters = Example().parameters()
    optimizer = load_optimizer('adamp')(parameters)

    with pytest.raises(ValueError):
        PCGrad(optimizer, reduction='wrong')


@pytest.mark.parametrize('optimizer_name', ['shampoo'])
def test_update_frequency(optimizer_name):
    optimizer = load_optimizer(optimizer_name)
    with pytest.raises(ValueError):
        optimizer(None, update_freq=0)


def test_sam_parameters():
    with pytest.raises(ValueError):
        SAM(None, load_optimizer('adamp'), rho=-0.1)


def test_lookahead_parameters():
    model: nn.Module = Example()
    parameters = model.parameters()
    optimizer = load_optimizer('adamp')(parameters)

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


def test_sam_methods():
    model: nn.Module = Example()
    parameters = model.parameters()

    optimizer = SAM(parameters, load_optimizer('adamp'))
    optimizer.reset()
    optimizer.load_state_dict(optimizer.state_dict())


def test_safe_fp16_methods():
    model: nn.Module = Example()

    optimizer = SafeFP16Optimizer(load_optimizer('adamp')(model.parameters(), lr=5e-1))
    optimizer.load_state_dict(optimizer.state_dict())
    optimizer.scaler.decrease_loss_scale()
    optimizer.zero_grad()
    optimizer.update_main_grads()
    optimizer.clip_main_grads(100.0)
    optimizer.multiply_grads(100.0)

    with pytest.raises(AttributeError):
        optimizer.get_lr()
        optimizer.set_lr(lr=5e-1)

    assert optimizer.loss_scale == 2.0 ** (15 - 1)


def test_ranger21_warm_methods():
    assert Ranger21.build_warm_up_iterations(1000, 0.999) == 220
    assert Ranger21.build_warm_up_iterations(4500, 0.999) == 2000
    assert Ranger21.build_warm_down_iterations(1000) == 280


@pytest.mark.parametrize('optimizer', ['ranger21', 'adai'])
def test_size_of_parameter(optimizer):
    model: nn.Module = nn.Linear(1, 1, bias=False)
    model.requires_grad_(False)

    with pytest.raises(ZeroParameterSizeError):
        load_optimizer(optimizer)(model.parameters(), 100).step()


def test_ranger21_closure():
    model: nn.Module = Example()
    optimizer = load_optimizer('ranger21')(model.parameters(), num_iterations=100, betas=(0.9, 1e-9))

    loss_fn = nn.BCEWithLogitsLoss()

    def closure():
        loss = loss_fn(torch.ones((1, 1)), model(torch.ones((1, 1))))
        loss.backward()
        return loss

    optimizer.step(closure)
