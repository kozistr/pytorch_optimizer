from unittest.mock import patch

import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def test_adopt_compiled_clipping():
    param = make_parameter(grad=1.0)
    optimizer = build_optimizer('adopt', [param], lr=0.1, betas=(0.5, 0.5), clip_lambda=lambda _: 0.25)
    with patch('torch.compile', side_effect=lambda step, **_: step):
        optimizer._compile_foreach()

    optimizer.step()
    torch.testing.assert_close(param, torch.zeros_like(param))

    for gradient, momentum, value in [(10.0, 0.125, -0.0125), (-10.0, -0.0625, -0.00625)]:
        param.grad.fill_(gradient)
        optimizer.step()
        torch.testing.assert_close(optimizer.state[param]['exp_avg'], torch.full_like(param, momentum))
        torch.testing.assert_close(param, torch.full_like(param, value))
