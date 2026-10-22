import pytest
import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def test_compiled_adamwsn_coupled_decay(device):
    if device.type != 'cuda' or not torch.cuda.is_bf16_supported():
        pytest.skip('Requires CUDA Inductor with bfloat16 support')

    param = make_parameter((2, 2), dtype=torch.bfloat16, device=device, grad=0.2)
    with torch.no_grad():
        param.fill_(0.5)
    optimizer = build_optimizer(
        'adamwsn', [param], lr=0.001, weight_decay=0.01, weight_decouple=False, foreach=True
    )
    optimizer._compile_foreach({'fullgraph': True})
    optimizer.step()

    torch.testing.assert_close(optimizer.state[param]['exp_avg'], torch.full_like(param, 0.02), rtol=0.0, atol=0.0)
    torch.testing.assert_close(param.grad, torch.full_like(param, 0.205), rtol=0.0, atol=0.0)
