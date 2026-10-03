import pytest

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


@pytest.mark.parametrize('optimizer_name', ['ScheduleFreeAdamW', 'ScheduleFreeSGD', 'ScheduleFreeRAdam'])
def test_schedule_free_methods(optimizer_name):
    optimizer = build_optimizer(optimizer_name, [make_parameter()])
    optimizer.step()

    optimizer.eval()
    optimizer.train()
