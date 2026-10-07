import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def test_splus_methods():
    param = make_parameter(grad=1.0)
    optimizer = build_optimizer('splus', [param])
    optimizer.step()
    training_param = param.detach().clone()

    optimizer.eval()
    assert not optimizer.param_groups[0]['train_mode']
    optimizer.train()
    torch.testing.assert_close(param, training_param)
