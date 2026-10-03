import pytest
import torch

from pytorch_optimizer.optimizer import load_optimizer
from tests.fixtures import make_parameter


@pytest.mark.parametrize('optimizer_name', ['racs', 'alice'])
def test_non_linear_parameters(optimizer_name):
    params = [make_parameter(shape, grad=1.0) for shape in ((4, 8, 1), (4, 8, 2, 2))]
    optimizer = load_optimizer(optimizer_name)(params, rank=4, leading_basis=2)
    optimizer.step()
    assert all(torch.isfinite(param).all() for param in params)
