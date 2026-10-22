import pytest
import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def test_compiled_adagc_preserves_fp16_accumulator_rounding(device):
    if device.type != 'cuda':
        pytest.skip('Requires CUDA Inductor')

    params = [make_parameter((1,), device=device, dtype=torch.float16, grad=0.005) for _ in range(2)]

    for param, foreach in zip(params, (False, True)):
        with torch.no_grad():
            param.fill_(0.5)

        optimizer = build_optimizer(
            'adagc', [param], lr=1e-6, eps=1e-6, weight_decay=0.0, warmup_steps=2, foreach=foreach,
        )

        if foreach:
            optimizer._compile_foreach({'fullgraph': True})

        for _ in range(2):
            optimizer.step()

    torch.testing.assert_close(params[0], params[1])
