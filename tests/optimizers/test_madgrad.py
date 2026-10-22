import pytest
import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def test_compiled_madgrad_without_momentum(device):
    if device.type != 'cuda':
        pytest.skip('Requires CUDA Inductor')

    param = make_parameter((2,), device=device, grad=0.2)
    with torch.no_grad():
        param.fill_(0.5)
    param.grad[1] = 0.0
    optimizer = build_optimizer('madgrad', [param], lr=0.001, momentum=0.0, eps=0.0, foreach=True)
    optimizer._compile_foreach({'fullgraph': True})

    for lr in (0.001, 0.0005):
        optimizer.param_groups[0]['lr'] = torch.tensor(lr, device=device)
        optimizer.step()

    torch.testing.assert_close(param, param.new_tensor([0.4916468466021341, 0.5]))
