import math

import pytest
import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


@pytest.mark.parametrize('optimizer_name', ['emonavi', 'emolynx', 'emofact'])
def test_shadow_correction(optimizer_name):
    param = make_parameter()
    optimizer = build_optimizer(optimizer_name, [param], use_shadow=True)

    optimizer.init_group(optimizer.param_groups[0])
    optimizer.state[param]['shadow'].fill_(1.0)
    optimizer.state['ema'] = {'short': 1.0, 'medium': 5.0, 'long': 40.0}

    optimizer.step()

    expected = 1.0 - math.tanh((39.6 - 0.7) / 39.6)
    torch.testing.assert_close(param, torch.full_like(param, expected))
