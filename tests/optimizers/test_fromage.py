import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def test_fromage_zero_norm():
    param = make_parameter()
    optimizer = build_optimizer('fromage', [param])
    optimizer.step()

    torch.testing.assert_close(param, torch.zeros_like(param))
