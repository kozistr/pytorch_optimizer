import math

import pytest
import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


@pytest.mark.parametrize(
    ('optimizer_name', 'foreach'),
    [('emonavi', False), ('emonavi', True), ('emolynx', False), ('emofact', False)],
)
def test_shadow_correction(optimizer_name, foreach):
    param = make_parameter()
    optimizer = build_optimizer(optimizer_name, [param], use_shadow=True, foreach=foreach)

    optimizer.init_group(optimizer.param_groups[0])
    optimizer.state[param]['shadow'].fill_(1.0)
    optimizer.state['ema'] = {'short': 1.0, 'medium': 5.0, 'long': 40.0}

    optimizer.step()

    expected = 1.0 - math.tanh((39.6 - 0.7) / 39.6)
    torch.testing.assert_close(param, torch.full_like(param, expected))

    expected_shadow = 0.95 + 0.05 * expected if optimizer_name == 'emonavi' else 1.0
    torch.testing.assert_close(optimizer.state[param]['shadow'], torch.full_like(param, expected_shadow))


def test_compiled_emolynx_momentum_sign(device):
    if device.type != 'cuda':
        pytest.skip('Requires CUDA Inductor')

    param = make_parameter(device=device, grad=-0.25)
    optimizer = build_optimizer('emolynx', [param], lr=0.001, foreach=True)
    optimizer.init_group(optimizer.param_groups[0])
    optimizer.state[param]['exp_avg'].fill_(0.03)
    optimizer._compile_foreach({'fullgraph': True})

    optimizer.step()

    torch.testing.assert_close(param, torch.full_like(param, -0.001))
    torch.testing.assert_close(optimizer.state[param]['exp_avg'], torch.full_like(param, 0.0272))
