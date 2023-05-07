import pytest
import torch
from torch import nn

from pytorch_optimizer import SAM, Lookahead, Ranger21, SafeFP16Optimizer, load_optimizer
from tests.constants import ADANORM_SUPPORTED_OPTIMIZERS, AMSBOUND_SUPPORTED_OPTIMIZERS, OPTIMIZERS, PULLBACK_MOMENTUM
from tests.utils import Example, ids, simple_parameter, simple_sparse_parameter, simple_zero_rank_parameter


@pytest.fixture(scope='module')
def parameter():
    return [simple_parameter()]


@pytest.fixture(scope='module')
def model_parameters():
    return Example().parameters()


def test_lookahead_methods(parameter):
    optimizer = load_optimizer('adamp')(parameter)

    for pullback_momentum in PULLBACK_MOMENTUM:
        opt = Lookahead(optimizer, pullback_momentum=pullback_momentum)
        opt.load_state_dict(opt.state_dict())

    opt = Lookahead(optimizer, pullback_momentum=pullback_momentum)
    opt.backup_and_load_cache()
    opt.clear_and_load_backup()

    _ = opt.__getstate__()


def test_sam_methods(parameter):
    optimizer = SAM(parameter, load_optimizer('adamp'))
    optimizer.reset()

    optimizer.load_state_dict(optimizer.state_dict())


def test_safe_fp16_methods(parameter):
    optimizer = SafeFP16Optimizer(load_optimizer('adamp')(parameter, lr=5e-1))
    optimizer.load_state_dict(optimizer.state_dict())
    optimizer.scaler.decrease_loss_scale()
    optimizer.zero_grad()
    optimizer.update_main_grads()
    optimizer.clip_main_grads(100.0)
    optimizer.multiply_grads(100.0)

    with pytest.raises(AttributeError):
        optimizer.get_lr()

    with pytest.raises(AttributeError):
        optimizer.set_lr(lr=5e-1)

    assert optimizer.loss_scale == 2.0 ** (15 - 1)


def test_ranger21_warm_iterations():
    assert Ranger21.build_warm_up_iterations(1000, 0.999) == 220
    assert Ranger21.build_warm_up_iterations(4500, 0.999) == 2000
    assert Ranger21.build_warm_down_iterations(1000) == 280


def test_ranger21_warm_up_and_down():
    param = simple_parameter(require_grad=False)

    lr: float = 1e-1
    opt = Ranger21([param], num_iterations=500, lr=lr, warm_down_min_lr=3e-5)

    assert opt.warm_up_dampening(lr, 100) == 0.09090909090909091
    assert opt.warm_up_dampening(lr, 200) == 0.1
    assert opt.warm_up_dampening(lr, 300) == 0.1
    assert opt.warm_down(lr, 300) == 0.1
    assert opt.warm_down(lr, 400) == 0.07093070921985817


def test_ranger21_closure():
    model: nn.Module = Example()
    optimizer = load_optimizer('ranger21')(model.parameters(), num_iterations=100, betas=(0.9, 1e-9))

    loss_fn = nn.BCEWithLogitsLoss()

    def closure():
        loss = loss_fn(torch.ones((1, 1)), model(torch.ones((1, 1))))
        loss.backward()
        return loss

    optimizer.step(closure)


def test_ada_factor_reset():
    optimizer = load_optimizer('adafactor')([simple_zero_rank_parameter()])
    optimizer.reset()


def test_ada_factor_get_lr(model_parameters):
    optimizer = load_optimizer('adafactor')(model_parameters)

    assert optimizer.get_lr(1.0, 1, 1.0, True, True, True) == 1e-6
    assert optimizer.get_lr(1.0, 1, 1.0, True, False, True) == 1e-2


@pytest.mark.parametrize(
    'optimizer_config', OPTIMIZERS + AMSBOUND_SUPPORTED_OPTIMIZERS + ADANORM_SUPPORTED_OPTIMIZERS, ids=ids
)
def test_reset(optimizer_config, model_parameters):
    optimizer_class, config, _ = optimizer_config
    if optimizer_class.__name__ == 'Ranger21':
        config.update({'num_iterations': 1})

    optimizer = optimizer_class(model_parameters, **config)
    optimizer.reset()


@pytest.mark.parametrize('require_gradient', [False, True])
@pytest.mark.parametrize('sparse_gradient', [False, True])
@pytest.mark.parametrize('optimizer_name', ['DAdaptAdaGrad', 'DAdaptAdam', 'DAdaptSGD', 'DAdaptAdan'])
def test_d_adapt_reset(require_gradient, sparse_gradient, optimizer_name):
    param = simple_sparse_parameter(require_gradient)[1] if sparse_gradient else simple_parameter(require_gradient)
    if not require_gradient:
        param.grad = None

    optimizer = load_optimizer(optimizer_name)([param])
    optimizer.reset()

    assert str(optimizer) == optimizer_name
