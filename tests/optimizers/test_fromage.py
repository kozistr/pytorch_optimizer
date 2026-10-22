import pytest
import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


@pytest.mark.parametrize('foreach', [False, True])
def test_fromage_zero_norm(foreach):
    param = make_parameter()
    optimizer = build_optimizer('fromage', [param], foreach=foreach)
    optimizer.step()

    torch.testing.assert_close(param, torch.zeros_like(param))
