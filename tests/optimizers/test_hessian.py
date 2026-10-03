import pytest
import torch

from pytorch_optimizer.optimizer import load_optimizer
from tests.fixtures import make_parameter
from tests.utils import sphere_loss


@pytest.mark.parametrize('optimizer_name', ['sophiah', 'adahessian'])
def test_hessian_optimizer(optimizer_name):
    param = make_parameter()

    parameters = {'hessian_distribution': 'gaussian', 'num_samples': 2}

    optimizer = load_optimizer(optimizer_name)([param], **parameters)
    optimizer.zero_grad(set_to_none=True)

    sphere_loss(param).backward(create_graph=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    sphere_loss(param).backward()
    optimizer.step(hessian=torch.zeros_like(param).unsqueeze(0))
