import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def test_spam_optimizer():
    param = make_parameter(grad=1.0)
    optimizer = build_optimizer('spam', [param], density=0.0, grad_accu_steps=0, update_proj_gap=1)
    optimizer.step()

    torch.testing.assert_close(param, torch.zeros_like(param))
