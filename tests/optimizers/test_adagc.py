import pytest
import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_mixed_precision_warmup_clipping(dtype):
    params = [make_parameter((2,), grad=3.0), make_parameter((2,), dtype=dtype, grad=4.0)]
    optimizer = build_optimizer('adagc', params, betas=(0.0, 0.0), weight_decay=0.0, eps=1e-4)
    expected = [p.grad.mul(50.0**-0.5) for p in params]

    optimizer.step()

    for p, gradient in zip(params, expected):
        torch.testing.assert_close(optimizer.state[p]['exp_avg'], gradient)
        assert torch.isfinite(p).all()
