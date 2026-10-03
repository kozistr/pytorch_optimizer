import pytest

from pytorch_optimizer.optimizer import load_optimizer
from tests.fixtures import make_parameter


class TestEpsilonParameters:
    """Tests for epsilon-related parameter validation."""

    @pytest.mark.parametrize(
        ('optimizer_name', 'param_name'),
        [
            ('Shampoo', 'matrix_eps'),
            ('ScalableShampoo', 'diagonal_eps'),
            ('ScalableShampoo', 'matrix_eps'),
        ],
    )
    def test_shampoo_epsilon_parameters(self, optimizer_name, param_name):
        opt = load_optimizer(optimizer_name)
        with pytest.raises(ValueError):
            opt(None, **{param_name: -1e-6})

    @pytest.mark.parametrize('optimizer_name', ['adafactor', 'came'])
    @pytest.mark.parametrize('param_name', ['eps1', 'eps2'])
    def test_multi_epsilon_parameters(self, optimizer_name, param_name):
        opt = load_optimizer(optimizer_name)
        with pytest.raises(ValueError):
            opt(None, **{param_name: -1e-6})


class TestNegativeParameterValidation:
    """Tests for negative parameter value validation."""

    @pytest.mark.parametrize(
        ('optimizer_name', 'param_name', 'param_value'),
        [
            ('accsgd', 'xi', -0.1),
            ('accsgd', 'kappa', -0.1),
            ('asgd', 'amplifier', -1.0),
            ('lars', 'dampening', -0.1),
            ('lars', 'trust_coefficient', -1e-3),
            ('ranger', 'alpha', -0.1),
            ('ranger', 'k', -1),
        ],
    )
    def test_negative_parameter_raises_error(self, optimizer_name, param_name, param_value):
        opt = load_optimizer(optimizer_name)
        params = [make_parameter(requires_grad=False)] if optimizer_name in ('accsgd', 'asgd') else None
        with pytest.raises(ValueError):
            opt(params, **{param_name: param_value})

    def test_accsgd_constant_validation(self):
        opt = load_optimizer('accsgd')
        with pytest.raises(ValueError):
            opt([make_parameter(requires_grad=False)], constant=42)


class TestInvalidOptionParameters:
    """Tests for invalid option/enum parameter validation."""

    @pytest.mark.parametrize(
        ('optimizer_name', 'param_name', 'invalid_value'),
        [
            ('apollodqn', 'rebound', 'dummy'),
            ('apollodqn', 'weight_decay_type', 'dummy'),
        ],
    )
    def test_invalid_option_raises_error(self, optimizer_name, param_name, invalid_value):
        opt = load_optimizer(optimizer_name)
        with pytest.raises(ValueError):
            opt(None, **{param_name: invalid_value})
