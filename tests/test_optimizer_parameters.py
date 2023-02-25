import pytest
import torch
from torch import nn

from pytorch_optimizer import SAM, Lookahead, PCGrad, Ranger21, SafeFP16Optimizer, load_optimizer
from pytorch_optimizer.base.exception import NegativeLRError, NegativeStepError, ZeroParameterSizeError
from tests.constants import BETA_OPTIMIZER_NAMES, PULLBACK_MOMENTUM, VALID_OPTIMIZER_NAMES
from tests.utils import Example, simple_parameter


@pytest.mark.parametrize('optimizer_name', VALID_OPTIMIZER_NAMES)
def test_learning_rate(optimizer_name):
    if optimizer_name in ('alig',):
        pytest.skip(f'skip {optimizer_name} optimizer')

    optimizer = load_optimizer(optimizer_name)

    config = {'lr': -1e-2}
    if optimizer_name == 'ranger21':
        config.update({'num_iterations': 100})

    with pytest.raises(NegativeLRError):
        optimizer(None, **config)


@pytest.mark.parametrize('optimizer_name', VALID_OPTIMIZER_NAMES)
def test_epsilon(optimizer_name):
    if optimizer_name in ('nero', 'shampoo', 'scalableshampoo', 'dadaptsgd', 'adafactor', 'lion'):
        pytest.skip(f'skip {optimizer_name} optimizer')

    optimizer = load_optimizer(optimizer_name)

    config = {'eps': -1e-6}
    if optimizer_name == 'ranger21':
        config.update({'num_iterations': 100})

    with pytest.raises(ValueError) as error_info:
        optimizer(None, **config)

    assert str(error_info.value) == '[-] epsilon -1e-06 must be non-negative'


def test_shampoo_epsilon():
    shampoo = load_optimizer('Shampoo')
    scalable_shampoo = load_optimizer('ScalableShampoo')

    with pytest.raises(ValueError):
        scalable_shampoo(None, diagonal_eps=-1e-6)

    with pytest.raises(ValueError):
        scalable_shampoo(None, matrix_eps=-1e-6)

    with pytest.raises(ValueError):
        shampoo(None, matrix_eps=-1e-6)


def test_adafactor_epsilon():
    adafactor = load_optimizer('adafactor')

    with pytest.raises(ValueError):
        adafactor(None, eps1=-1e-6)

    with pytest.raises(ValueError):
        adafactor(None, eps2=-1e-6)


@pytest.mark.parametrize('optimizer_name', VALID_OPTIMIZER_NAMES)
def test_weight_decay(optimizer_name):
    if optimizer_name in ('nero', 'alig'):
        pytest.skip(f'skip {optimizer_name} optimizer')

    optimizer = load_optimizer(optimizer_name)

    config = {'weight_decay': -1e-3}
    if optimizer_name == 'ranger21':
        config.update({'num_iterations': 100})

    with pytest.raises(ValueError) as error_info:
        optimizer(None, **config)

    assert str(error_info.value) == '[-] weight_decay -0.001 must be non-negative'


@pytest.mark.parametrize('optimizer_name', ['apollo'])
def test_weight_decay_type(optimizer_name):
    optimizer = load_optimizer(optimizer_name)

    with pytest.raises(ValueError):
        optimizer(None, weight_decay_type='dummy')


@pytest.mark.parametrize('optimizer_name', ['apollo'])
def test_rebound(optimizer_name):
    optimizer = load_optimizer(optimizer_name)

    with pytest.raises(ValueError):
        optimizer(None, rebound='dummy')


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


@pytest.mark.parametrize('optimizer_name', ['nero', 'apollo'])
def test_beta(optimizer_name):
    optimizer = load_optimizer(optimizer_name)
    with pytest.raises(ValueError):
        optimizer(None, beta=-0.1)


@pytest.mark.parametrize('optimizer_name', BETA_OPTIMIZER_NAMES)
def test_betas(optimizer_name):
    optimizer = load_optimizer(optimizer_name)

    config1 = {'betas': (-0.1, 0.1)}
    config2 = {'betas': (0.1, -0.1)}
    if optimizer_name == 'ranger21':
        config1.update({'num_iterations': 100})
        config2.update({'num_iterations': 100})

    if optimizer_name not in ('adapnm', 'adan'):
        with pytest.raises(ValueError):
            optimizer(None, **config1)

        with pytest.raises(ValueError):
            optimizer(None, **config2)
    else:
        with pytest.raises(ValueError):
            optimizer(None, betas=(0.1, 0.1, -0.1))


def test_reduction():
    parameters = Example().parameters()
    optimizer = load_optimizer('adamp')(parameters)

    with pytest.raises(ValueError):
        PCGrad(optimizer, reduction='wrong')


@pytest.mark.parametrize('optimizer_name', ['scalableshampoo', 'shampoo'])
def test_update_frequency(optimizer_name):
    optimizer = load_optimizer(optimizer_name)

    if optimizer_name == 'scalableshampoo':
        with pytest.raises(NegativeStepError):
            optimizer(None, start_preconditioning_step=-1)

        with pytest.raises(NegativeStepError):
            optimizer(None, statistics_compute_steps=-1)

    with pytest.raises(NegativeStepError):
        optimizer(None, preconditioning_compute_steps=-1)


@pytest.mark.parametrize('optimizer_name', ['adan', 'lamb'])
def test_norm(optimizer_name):
    optimizer = load_optimizer(optimizer_name)
    with pytest.raises(ValueError):
        optimizer(None, max_grad_norm=-0.1)


def test_sam_parameters():
    with pytest.raises(ValueError, match=''):
        SAM(None, load_optimizer('adamp'), rho=-0.1)


def test_lookahead_parameters():
    param = simple_parameter()
    optimizer = load_optimizer('adamp')([param])

    for pullback_momentum in PULLBACK_MOMENTUM:
        opt = Lookahead(optimizer, pullback_momentum=pullback_momentum)
        opt.load_state_dict(opt.state_dict())

        opt.update_lookahead()
        opt.add_param_group({'params': [simple_parameter()]})

    with pytest.raises(ValueError):
        Lookahead(optimizer, k=0)

    with pytest.raises(ValueError):
        Lookahead(optimizer, alpha=-0.1)

    with pytest.raises(ValueError):
        Lookahead(optimizer, pullback_momentum='invalid')


def test_sam_methods():
    param = simple_parameter()

    optimizer = SAM([param], load_optimizer('adamp'))
    optimizer.reset()
    optimizer.load_state_dict(optimizer.state_dict())


def test_safe_fp16_methods():
    param = simple_parameter()

    optimizer = SafeFP16Optimizer(load_optimizer('adamp')([param], lr=5e-1))
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


def test_ranger21_warm_methods():
    assert Ranger21.build_warm_up_iterations(1000, 0.999) == 220
    assert Ranger21.build_warm_up_iterations(4500, 0.999) == 2000
    assert Ranger21.build_warm_down_iterations(1000) == 280


@pytest.mark.parametrize('optimizer', ['ranger21', 'adai'])
def test_size_of_parameter(optimizer):
    param = simple_parameter(require_grad=False)
    param.grad = None

    with pytest.raises(ZeroParameterSizeError):
        load_optimizer(optimizer)([param], 1).step()


def test_ranger21_closure():
    model: nn.Module = Example()
    optimizer = load_optimizer('ranger21')(model.parameters(), num_iterations=100, betas=(0.9, 1e-9))

    loss_fn = nn.BCEWithLogitsLoss()

    def closure():
        loss = loss_fn(torch.ones((1, 1)), model(torch.ones((1, 1))))
        loss.backward()
        return loss

    optimizer.step(closure)


def test_adafactor_reset():
    param = torch.zeros(1).requires_grad_(True)
    param.grad = torch.zeros(1)

    optimizer = load_optimizer('adafactor')([param])
    optimizer.reset()


def test_adafactor_get_lr():
    model: nn.Module = Example()

    optimizer = load_optimizer('adafactor')(model.parameters(), warmup_init=True)
    assert optimizer.get_lr(1.0, 1, 1.0) == 1e-6

    optimizer = load_optimizer('adafactor')(model.parameters(), warmup_init=False)
    assert optimizer.get_lr(1.0, 1, 1.0) == 1e-2
