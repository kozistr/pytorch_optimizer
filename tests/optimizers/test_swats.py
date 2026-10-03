import torch

from pytorch_optimizer.optimizer import load_optimizer
from tests.fixtures import make_parameter


def test_swats_sgd_phase():
    inactive = make_parameter(grad=None)
    active = make_parameter((1, 2))
    optimizer = load_optimizer('swats')([inactive, active], lr=1e-1, nesterov=True, eps=1.0)
    optimizer.step()
    active.grad = torch.ones_like(active)
    optimizer.step()
    optimizer.param_groups[0]['phase'] = 'sgd'
    optimizer.step()
