import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def test_foreach_proximal_weight_decay():
    param = make_parameter(dtype=torch.float64, grad=1.0)
    lr, weight_decay = 0.1, 0.2
    optimizer = build_optimizer(
        'adan', [param], lr=lr, weight_decay=weight_decay, eps=0.0, foreach=True
    )
    optimizer.step()

    expected = -lr / (1.0 + lr * weight_decay)
    torch.testing.assert_close(param, torch.full_like(param, expected))
