import pytest
import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


@pytest.mark.parametrize('foreach', [False, True])
def test_swats_sgd_phase(foreach, device):
    active = make_parameter((1, 2), device=device)
    other = make_parameter((1, 2), device=device)
    optimizer = build_optimizer('swats', [active, other], lr=1e-1, nesterov=True, eps=1.0, foreach=foreach)
    optimizer.step()
    active.grad = torch.ones_like(active)
    other.grad = torch.ones_like(other)
    optimizer.step()
    torch.testing.assert_close(active, other)
    assert optimizer.param_groups[0]['phase'] == 'sgd'
    expected = active.detach() - optimizer.param_groups[0]['lr']
    optimizer.step()

    torch.testing.assert_close(active, expected)
    torch.testing.assert_close(other, expected)
