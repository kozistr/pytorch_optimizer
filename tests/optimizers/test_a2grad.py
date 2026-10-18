import pytest
import torch

from pytorch_optimizer.optimizer import load_optimizer
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


class TestA2GradParameters:
    @pytest.mark.parametrize(
        ('param_name', 'param_value'),
        [
            ('lips', -1.0),
            ('lips', 0.0),
            ('rho', -0.1),
        ],
    )
    def test_invalid_parameters(self, param_name, param_value):
        params = [make_parameter(requires_grad=False)] if param_name == 'lips' else None
        with pytest.raises(ValueError):
            load_optimizer('a2grad')(params, **{param_name: param_value})

    def test_invalid_variant(self):
        with pytest.raises(ValueError):
            load_optimizer('a2grad')([make_parameter(requires_grad=False)], variant='dummy')


def test_incremental_low_precision_accumulator():
    param = make_parameter(shape=(12,), dtype=torch.float16)
    optimizer = build_optimizer('a2grad', [param], variant='inc', foreach=True)
    grad = torch.arange(12, dtype=param.dtype) - 3.0

    for step in range(9):
        param.grad = grad * ((step % 3 - 1) * 0.1)
        optimizer.step()

    torch.testing.assert_close(optimizer.state[param]['v_k'], torch.tensor([5.03125], dtype=param.dtype))
