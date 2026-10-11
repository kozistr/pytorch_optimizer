import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def test_aggmo_cpu_bfloat16_momentum_precision():
    param = make_parameter((64,), dtype=torch.bfloat16)
    expected = make_parameter((64,), dtype=torch.bfloat16)
    optimizer = build_optimizer('aggmo', [param], lr=0.01, foreach=True)
    reference = build_optimizer('aggmo', [expected], lr=0.01, foreach=False)

    for gradient in (0.6796875, 0.0):
        param.grad = torch.full_like(param, gradient)
        expected.grad = param.grad.clone()

        optimizer.step()
        reference.step()

        torch.testing.assert_close(param, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            optimizer.state[param]['momentum_buffer'], reference.state[expected]['momentum_buffer'], rtol=0.0, atol=0.0
        )
