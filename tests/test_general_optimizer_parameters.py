import pytest

from pytorch_optimizer.base.exception import NegativeLRError, NegativeStepError, ZeroParameterSizeError
from pytorch_optimizer.optimizer import PCGrad, load_optimizer
from tests.constants import (
    BETA_OPTIMIZER_NAMES,
    SKIP_EPSILON,
    SKIP_LEARNING_RATE,
    SKIP_WEIGHT_DECAY,
    VALID_OPTIMIZER_NAMES,
)
from tests.utils import Example, simple_parameter


def _config_for_optimizer(optimizer_name: str, **config):
    if optimizer_name == 'ranger21':
        config['num_iterations'] = config.get('num_iterations', 100)
    elif optimizer_name == 'bsam':
        config['num_data'] = config.get('num_data', 100)
    return config


class TestBasicParameterValidation:
    @pytest.mark.parametrize('optimizer_name', VALID_OPTIMIZER_NAMES)
    def test_learning_rate(self, optimizer_name):
        if optimizer_name in SKIP_LEARNING_RATE:
            pytest.skip(f'skip {optimizer_name} optimizer')

        optimizer = load_optimizer(optimizer_name)
        config = _config_for_optimizer(optimizer_name, lr=-1e-2)

        with pytest.raises(NegativeLRError):
            optimizer(None, **config)

    @pytest.mark.parametrize('optimizer_name', VALID_OPTIMIZER_NAMES)
    def test_epsilon(self, optimizer_name):
        if optimizer_name in SKIP_EPSILON:
            pytest.skip(f'skip {optimizer_name} optimizer')

        optimizer = load_optimizer(optimizer_name)
        config = _config_for_optimizer(optimizer_name, eps=-1e-6)

        with pytest.raises(ValueError):
            optimizer(None, **config)

    @pytest.mark.parametrize('optimizer_name', VALID_OPTIMIZER_NAMES)
    def test_weight_decay(self, optimizer_name):
        if optimizer_name in SKIP_WEIGHT_DECAY:
            pytest.skip(f'skip {optimizer_name} optimizer')

        optimizer = load_optimizer(optimizer_name)
        config = _config_for_optimizer(optimizer_name, weight_decay=-1e-3)

        with pytest.raises(ValueError):
            optimizer(None, **config)

    @pytest.mark.parametrize('optimizer_name', ['adamp', 'sgdp'])
    def test_wd_ratio(self, optimizer_name):
        optimizer = load_optimizer(optimizer_name)
        with pytest.raises(ValueError):
            optimizer(None, wd_ratio=-1e-3)

    @pytest.mark.parametrize('optimizer_name', ['madgrad', 'lars', 'sm3', 'sgdw'])
    def test_momentum(self, optimizer_name):
        optimizer = load_optimizer(optimizer_name)
        with pytest.raises(ValueError):
            optimizer(None, momentum=-1e-3)


class TestBetaParameterValidation:
    @pytest.mark.parametrize('optimizer_name', ['nero', 'apollodqn', 'sm3', 'msvag', 'ranger21'])
    def test_beta(self, optimizer_name):
        optimizer = load_optimizer(optimizer_name)
        config = _config_for_optimizer(
            optimizer_name,
            beta0=-0.1 if optimizer_name == 'ranger21' else None,
            beta=-0.1,
        )

        with pytest.raises(ValueError):
            optimizer(None, **config)

    @pytest.mark.parametrize('optimizer_name', BETA_OPTIMIZER_NAMES)
    def test_betas(self, optimizer_name):
        optimizer = load_optimizer(optimizer_name)

        config1 = _config_for_optimizer(optimizer_name, betas=(-0.1, 0.1))
        config2 = _config_for_optimizer(optimizer_name, betas=(0.1, -0.1))

        if optimizer_name not in ('adapnm', 'adan', 'adamod', 'aggmo', 'came'):
            with pytest.raises(ValueError):
                optimizer(None, **config1)

            with pytest.raises(ValueError):
                optimizer(None, **config2)
        elif optimizer_name == 'prodigy':
            with pytest.raises(ValueError):
                optimizer(None, beta3=-0.1)
        else:
            with pytest.raises(ValueError):
                optimizer(None, betas=(0.1, 0.1, -0.1))


class TestSpecialParameterValidation:
    def test_reduction(self):
        optimizer = load_optimizer('adamp')(Example().parameters())
        with pytest.raises(ValueError):
            PCGrad(optimizer, reduction='wrong')

    @pytest.mark.parametrize('optimizer_name', ['scalableshampoo', 'shampoo'])
    def test_update_frequency(self, optimizer_name):
        optimizer = load_optimizer(optimizer_name)

        if optimizer_name == 'shampoo':
            with pytest.raises(NegativeStepError):
                optimizer(None, preconditioning_compute_steps=-1)
        elif optimizer_name == 'scalableshampoo':
            with pytest.raises(NegativeStepError):
                optimizer(None, start_preconditioning_step=-1)

            with pytest.raises(NegativeStepError):
                optimizer(None, statistics_compute_steps=-1)

    @pytest.mark.parametrize('optimizer_name', ['adan', 'lamb'])
    def test_norm(self, optimizer_name):
        with pytest.raises(ValueError):
            load_optimizer(optimizer_name)(None, max_grad_norm=-0.1)

    @pytest.mark.parametrize('optimizer', ['ranger21', 'adai'])
    def test_size_of_parameter(self, optimizer):
        param = simple_parameter(require_grad=False)
        param.grad = None

        with pytest.raises(ZeroParameterSizeError):
            load_optimizer(optimizer)([param], 1).step()

    @pytest.mark.parametrize('optimizer_name', ['qhadam', 'qhm'])
    def test_nus(self, optimizer_name):
        optimizer = load_optimizer(optimizer_name)

        if optimizer_name == 'qhadam':
            with pytest.raises(ValueError):
                optimizer([simple_parameter(False)], nus=(-0.1, 0.1))

            with pytest.raises(ValueError):
                optimizer([simple_parameter(False)], nus=(0.1, -0.1))
        else:
            with pytest.raises(ValueError):
                optimizer([simple_parameter(False)], nu=-0.1)
