import pytest
import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer, sphere_loss


@pytest.mark.parametrize('optimizer_name', ['sophiah', 'adahessian'])
def test_hessian_optimizer(optimizer_name):
    param = make_parameter()

    optimizer = build_optimizer(optimizer_name, [param], hessian_distribution='gaussian', num_samples=2)

    (param.grad,) = torch.autograd.grad(sphere_loss(param), param, create_graph=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    sphere_loss(param).backward()
    optimizer.step(hessian=torch.zeros_like(param).unsqueeze(0))

    torch.testing.assert_close(optimizer.state[param]['hessian'], torch.zeros_like(param))
