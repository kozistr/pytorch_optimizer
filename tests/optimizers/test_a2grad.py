import pytest

from pytorch_optimizer.optimizer import load_optimizer
from tests.fixtures import make_parameter


class TestA2GradParameters:
    @pytest.mark.parametrize(
        ('param_name', 'param_value'),
        [
            ('lips', -1.0),
            ('rho', -0.1),
        ],
    )
    def test_negative_parameters(self, param_name, param_value):
        params = [make_parameter(requires_grad=False)] if param_name == 'lips' else None
        with pytest.raises(ValueError):
            load_optimizer('a2grad')(params, **{param_name: param_value})

    def test_invalid_variant(self):
        with pytest.raises(ValueError):
            load_optimizer('a2grad')([make_parameter(requires_grad=False)], variant='dummy')
