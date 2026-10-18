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


class TestA2GradForeach:
    def test_incremental_low_precision_accumulator(self):
        params = [make_parameter(shape=(4, 3), dtype=torch.float16) for _ in range(2)]
        reference = build_optimizer('a2grad', [params[0]], variant='inc', foreach=False)
        optimizer = build_optimizer('a2grad', [params[1]], variant='inc', foreach=True)

        for step in range(9):
            grad = (torch.arange(12, dtype=torch.float16).reshape(4, 3) - 3.0).mul_((step % 3 - 1) * 0.1)
            for param in params:
                param.grad = grad.clone()
            reference.step()
            optimizer.step()

        torch.testing.assert_close(optimizer.state[params[1]]['v_k'], reference.state[params[0]]['v_k'])
        torch.testing.assert_close(params[1], params[0])
