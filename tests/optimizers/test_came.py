import pytest
import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def test_compiled_came_records_rms_before_update(device):
    if device.type != 'cuda':
        pytest.skip('Requires CUDA Inductor')

    param = make_parameter((), device=device, grad=0.2)

    with torch.no_grad():
        param.fill_(0.5)

    optimizer = build_optimizer(
        'came', [param], lr=0.001, weight_decay=0.01, weight_decouple=False,
        maximize=True, ams_bound=True, foreach=True,
    )

    optimizer._compile_foreach({'fullgraph': True})
    optimizer.step()

    torch.testing.assert_close(optimizer.state[param]['RMS'], param.new_tensor(0.5))


def test_compiled_came_preserves_fp16_moment_rounding(device):
    if device.type != 'cuda':
        pytest.skip('Requires CUDA Inductor')

    moments = []

    for foreach in (False, True):
        param = make_parameter((2, 2), device=device, dtype=torch.float16)

        with torch.no_grad():
            param.fill_(0.5)

        param.grad = param.new_tensor([[0.001, 0.002], [0.003, 0.004]])

        optimizer = build_optimizer(
            'came', [param], lr=0.001, eps1=1e-3, eps2=1e-3, ams_bound=True,
            weight_decay=0.01, weight_decouple=False, foreach=foreach,
        )

        if foreach:
            optimizer._compile_foreach({'fullgraph': True})

        optimizer.step()
        moments.append(optimizer.state[param]['exp_avg'])

    torch.testing.assert_close(moments[0], moments[1])
