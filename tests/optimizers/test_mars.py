import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def test_mars_c_t_norm():
    param = make_parameter(grad=100.0)

    optimizer = build_optimizer('mars', [param], optimize_1d=True)
    optimizer.step()

    torch.testing.assert_close(optimizer.state[param]['exp_avg'], torch.full_like(param, 0.05))
