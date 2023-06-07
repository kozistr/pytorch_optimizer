import pytest

from pytorch_optimizer import PCGrad, load_optimizer
from pytorch_optimizer.base.exception import NegativeLRError, NegativeStepError, ZeroParameterSizeError
from tests.constants import BETA_OPTIMIZER_NAMES, VALID_OPTIMIZER_NAMES
from tests.utils import Example, simple_parameter


@pytest.mark.parametrize('optimizer_name', VALID_OPTIMIZER_NAMES)
def test_learning_rate(optimizer_name):
    if optimizer_name in ('alig', 'a2grad'):
        pytest.skip(f'skip {optimizer_name} optimizer')

    optimizer = load_optimizer(optimizer_name)

    config = {'lr': -1e-2}
    if optimizer_name == 'ranger21':
        config.update({'num_iterations': 100})

    with pytest.raises(NegativeLRError):
        optimizer(None, **config)


@pytest.mark.parametrize('optimizer_name', VALID_OPTIMIZER_NAMES)
def test_epsilon(optimizer_name):
    if optimizer_name in (
        'shampoo',
        'scalableshampoo',
        'dadaptsgd',
        'adafactor',
        'lion',
        'a2grad',
        'accsgd',
        'sgdw',
        'fromage',
        'msvag',
        'aggmo',
        'qhm',
        'pid',
        'lars',
        'alig',
        'gravity',
        'srmm',
        'signsgd'
    ):
        pytest.skip(f'skip {optimizer_name} optimizer')

    optimizer = load_optimizer(optimizer_name)

    config = {'eps': -1e-6}
    if optimizer_name == 'ranger21':
        config.update({'num_iterations': 100})

    with pytest.raises(ValueError):
        optimizer(None, **config)


@pytest.mark.parametrize('optimizer_name', VALID_OPTIMIZER_NAMES)
def test_weight_decay(optimizer_name):
    if optimizer_name in ('nero', 'alig', 'sm3', 'a2grad', 'fromage', 'msvag', 'gravity', 'srmm', 'adashift', 'amos'):
        pytest.skip(f'skip {optimizer_name} optimizer')

    optimizer = load_optimizer(optimizer_name)

    config = {'weight_decay': -1e-3}
    if optimizer_name == 'ranger21':
        config.update({'num_iterations': 100})

    with pytest.raises(ValueError):
        optimizer(None, **config)


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


@pytest.mark.parametrize('optimizer_name', ['madgrad', 'lars', 'sm3', 'sgdw'])
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


@pytest.mark.parametrize('optimizer_name', ['nero', 'apollo', 'sm3', 'msvag'])
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

    if optimizer_name not in ('adapnm', 'adan', 'adamod', 'aggmo'):
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


@pytest.mark.parametrize('optimizer_name', ['a2grad'])
def test_rho(optimizer_name):
    optimizer = load_optimizer(optimizer_name)
    with pytest.raises(ValueError):
        optimizer(None, rho=-0.1)


@pytest.mark.parametrize('optimizer_name', ['accsgd'])
def test_kappa(optimizer_name):
    optimizer = load_optimizer(optimizer_name)
    with pytest.raises(ValueError):
        optimizer([simple_parameter(False)], kappa=-0.1)


@pytest.mark.parametrize('optimizer_name', ['accsgd'])
def test_xi(optimizer_name):
    optimizer = load_optimizer(optimizer_name)
    with pytest.raises(ValueError):
        optimizer([simple_parameter(False)], xi=-0.1)


@pytest.mark.parametrize('optimizer_name', ['accsgd'])
def test_constant(optimizer_name):
    optimizer = load_optimizer(optimizer_name)
    with pytest.raises(ValueError):
        optimizer([simple_parameter(False)], constant=42)


@pytest.mark.parametrize('optimizer', ['ranger21', 'adai'])
def test_size_of_parameter(optimizer):
    param = simple_parameter(require_grad=False)
    param.grad = None

    with pytest.raises(ZeroParameterSizeError):
        load_optimizer(optimizer)([param], 1).step()


@pytest.mark.parametrize('optimizer_name', ['asgd'])
def test_amplifier(optimizer_name):
    optimizer = load_optimizer(optimizer_name)
    with pytest.raises(ValueError):
        optimizer([simple_parameter(False)], amplifier=-1.0)


@pytest.mark.parametrize('optimizer_name', ['qhadam', 'qhm'])
def test_nus(optimizer_name):
    optimizer = load_optimizer(optimizer_name)

    if optimizer_name == 'qhadam':
        with pytest.raises(ValueError):
            optimizer([simple_parameter(False)], nus=(-0.1, 0.1))

        with pytest.raises(ValueError):
            optimizer([simple_parameter(False)], nus=(0.1, -0.1))
    else:
        with pytest.raises(ValueError):
            optimizer([simple_parameter(False)], nu=-0.1)
