import pytest

from pytorch_optimizer.base.exception import NegativeLRError, NegativeStepError
from pytorch_optimizer.optimizer import load_optimizer
from tests.fixtures import make_parameter
from tests.optimizer_cases import (
    BETA_OPTIMIZER_NAMES,
    SKIP_EPSILON,
    SKIP_LEARNING_RATE,
    SKIP_WEIGHT_DECAY,
    VALID_OPTIMIZER_NAMES,
)


def _config_for_optimizer(optimizer_name: str, **config):
    if optimizer_name == 'ranger21':
        config['num_iterations'] = config.get('num_iterations', 100)
    elif optimizer_name == 'bsam':
        config['num_data'] = config.get('num_data', 100)
    elif optimizer_name == 'distributedmuon' and 'eps' in config:
        config['adamw_eps'] = config.pop('eps')
    return config


class TestBasicParameterValidation:
    @pytest.mark.parametrize(
        'optimizer_name', [name for name in VALID_OPTIMIZER_NAMES if name not in SKIP_LEARNING_RATE]
    )
    def test_learning_rate(self, optimizer_name):
        optimizer = load_optimizer(optimizer_name)
        config = _config_for_optimizer(optimizer_name, lr=-1e-2)

        with pytest.raises(NegativeLRError):
            optimizer(None, **config)

    @pytest.mark.parametrize('optimizer_name', [name for name in VALID_OPTIMIZER_NAMES if name not in SKIP_EPSILON])
    def test_epsilon(self, optimizer_name):
        optimizer = load_optimizer(optimizer_name)
        config = _config_for_optimizer(optimizer_name, eps=-1e-6)

        with pytest.raises(ValueError):
            optimizer(None, **config)

    @pytest.mark.parametrize(
        'optimizer_name', [name for name in VALID_OPTIMIZER_NAMES if name not in SKIP_WEIGHT_DECAY]
    )
    def test_weight_decay(self, optimizer_name):
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
        config = _config_for_optimizer(optimizer_name, **{('beta0' if optimizer_name == 'ranger21' else 'beta'): -0.1})

        with pytest.raises(ValueError):
            optimizer(None, **config)

    @pytest.mark.parametrize('optimizer_name', sorted(BETA_OPTIMIZER_NAMES))
    def test_betas(self, optimizer_name):
        optimizer = load_optimizer(optimizer_name)

        betas = [(0.1, 0.1, -0.1)] if optimizer_name in ('adapnm', 'adan', 'adamod', 'aggmo', 'came') else [
            (-0.1, 0.1), (0.1, -0.1)
        ]
        for invalid_betas in betas:
            with pytest.raises(ValueError):
                optimizer(None, **_config_for_optimizer(optimizer_name, betas=invalid_betas))


class TestSpecialParameterValidation:
    @pytest.mark.parametrize('optimizer_name', ['scalableshampoo', 'shampoo'])
    def test_update_frequency(self, optimizer_name):
        optimizer = load_optimizer(optimizer_name)
        options = ('preconditioning_compute_steps',) if optimizer_name == 'shampoo' else (
            'start_preconditioning_step', 'statistics_compute_steps'
        )
        for option in options:
            with pytest.raises(NegativeStepError):
                optimizer(None, **{option: -1})

    @pytest.mark.parametrize('optimizer_name', ['adan', 'lamb'])
    def test_norm(self, optimizer_name):
        with pytest.raises(ValueError):
            load_optimizer(optimizer_name)(None, max_grad_norm=-0.1)

    @pytest.mark.parametrize(
        ('optimizer_name', 'options'),
        [('qhadam', {'nus': (-0.1, 0.1)}), ('qhadam', {'nus': (0.1, -0.1)}), ('qhm', {'nu': -0.1})],
    )
    def test_nus(self, optimizer_name, options):
        with pytest.raises(ValueError):
            load_optimizer(optimizer_name)([make_parameter()], **options)
