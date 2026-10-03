import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def test_swats_sgd_phase():
    inactive = make_parameter(grad=None)
    active = make_parameter((1, 2))
    optimizer = build_optimizer('swats', [inactive, active], lr=1e-1, nesterov=True, eps=1.0)
    optimizer.step()
    active.grad = torch.ones_like(active)
    optimizer.step()
    optimizer.param_groups[0]['phase'] = 'sgd'
    optimizer.step()
