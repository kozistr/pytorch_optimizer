import pytest
import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer, sphere_loss


@pytest.mark.parametrize('optimizer_name', ['sophiah', 'adahessian'])
def test_hessian_optimizer(optimizer_name):
    param = make_parameter()

    parameters = {'hessian_distribution': 'gaussian', 'num_samples': 2}

    optimizer = build_optimizer(optimizer_name, [param], **parameters)
    optimizer.zero_grad(set_to_none=True)

    (param.grad,) = torch.autograd.grad(sphere_loss(param), param, create_graph=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    sphere_loss(param).backward()
    optimizer.step(hessian=torch.zeros_like(param).unsqueeze(0))
