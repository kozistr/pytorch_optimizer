import pytest
import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


@pytest.mark.parametrize('optimizer_name', ['ScheduleFreeAdamW', 'ScheduleFreeSGD', 'ScheduleFreeRAdam'])
def test_schedule_free_methods(optimizer_name):
    param = make_parameter(grad=1.0)
    optimizer = build_optimizer(optimizer_name, [param])
    for _ in range(2):
        optimizer.step()
    training_param = param.detach().clone()

    optimizer.eval()
    assert not optimizer.param_groups[0]['train_mode']
    optimizer.train()
    torch.testing.assert_close(param, training_param)


@pytest.mark.parametrize('foreach', [False, True])
def test_schedule_free_adamw_ams_bound(foreach):
    param = make_parameter((2,), grad=2.0)
    optimizer = build_optimizer(
        'schedulefreeadamw', [param], lr=0.1, betas=(0.5, 0.5), ams_bound=True, eps=0.0, foreach=foreach
    )
    optimizer.step()
    param.grad.fill_(1.0)
    optimizer.step()

    torch.testing.assert_close(param, torch.full_like(param, -0.1375))
    torch.testing.assert_close(optimizer.state[param]['max_exp_avg_sq'], torch.full_like(param, 4.0))
