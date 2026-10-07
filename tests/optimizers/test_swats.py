import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def test_swats_sgd_phase():
    active = make_parameter((1, 2))
    optimizer = build_optimizer('swats', [active], lr=1e-1, nesterov=True, eps=1.0)
    optimizer.step()
    active.grad = torch.ones_like(active)
    optimizer.step()
    assert optimizer.param_groups[0]['phase'] == 'sgd'
    expected = active.detach() - optimizer.param_groups[0]['lr']
    optimizer.step()

    torch.testing.assert_close(active, expected)
