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
