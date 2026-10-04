import math

import pytest
import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('lr', [0.0, 0.1])
def test_zero_epsilon_matches_dual_averaging(sparse, lr):
    param = make_parameter((2, 2), grad=None)
    optimizer = build_optimizer('madgrad', [param], lr=lr, momentum=0.0, eps=0.0)

    grad_sum = torch.zeros_like(param)
    grad_sum_sq = torch.zeros_like(param)
    gradients = (
        torch.tensor([[2.0, 0.0], [0.0, 0.0]]),
        torch.tensor([[0.0, 0.0], [0.0, -4.0]]),
        torch.tensor([[-1.0, 0.0], [0.0, 3.0]]),
    )
    for step, grad in enumerate(gradients, start=1):
        param.grad = grad.to_sparse() if sparse else grad.clone()
        optimizer.step()

        weight = lr * math.sqrt(step)
        grad_sum += weight * grad
        grad_sum_sq += weight * grad.square()
        denominator = grad_sum_sq.pow(1.0 / 3.0).clamp_min(torch.finfo(param.dtype).tiny)

        torch.testing.assert_close(param, -grad_sum / denominator)
