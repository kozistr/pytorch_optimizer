import pytest

from pytorch_optimizer.optimizer import load_optimizer
from tests.fixtures import make_parameter


@pytest.mark.parametrize('optimizer_name', ['ScheduleFreeAdamW', 'ScheduleFreeSGD', 'ScheduleFreeRAdam'])
def test_schedule_free_methods(optimizer_name):
    optimizer = load_optimizer(optimizer_name)([make_parameter()])
    optimizer.step()

    optimizer.eval()
    optimizer.train()
