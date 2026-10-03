import pytest

from pytorch_optimizer.optimizer import load_optimizer
from tests.fixtures import make_parameter


class TestA2GradParameters:
    """Tests for A2Grad optimizer parameter validation."""

    @pytest.mark.parametrize(
        ('param_name', 'param_value'),
        [
            ('lips', -1.0),
            ('rho', -0.1),
        ],
    )
    def test_negative_parameters(self, param_name, param_value):
        param = [make_parameter(requires_grad=False)]
        opt = load_optimizer('a2grad')
        params = param if param_name == 'lips' else None
        with pytest.raises(ValueError):
            opt(params, **{param_name: param_value})

    @pytest.mark.parametrize('variant', ['uni', 'inc', 'exp'])
    def test_valid_variants(self, variant):
        param = [make_parameter(requires_grad=False)]
        opt = load_optimizer('a2grad')
        opt(param, variant=variant)

    def test_invalid_variant(self):
        param = [make_parameter(requires_grad=False)]
        opt = load_optimizer('a2grad')
        with pytest.raises(ValueError):
            opt(param, variant='dummy')
