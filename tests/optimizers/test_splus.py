import pytest
import torch

from pytorch_optimizer import create_optimizer
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


@pytest.mark.parametrize('compiled', [False, True])
def test_splus_methods(compiled, device):
    if compiled and device.type != 'cuda':
        pytest.skip('Requires CUDA Inductor')

    param = make_parameter(grad=1.0, device=device)
    if compiled:
        model = torch.nn.ParameterList([param])
        param = model[0]
        param.grad = torch.ones_like(param)
        optimizer = create_optimizer(model, 'splus', lr=0.1, weight_decay=0.01, compile=True)
    else:
        optimizer = build_optimizer('splus', [param])
    optimizer.step()
    training_param = param.detach().clone()

    for side in optimizer.state[param]['sides']:
        torch.testing.assert_close(side, torch.full_like(side, 0.001))

    optimizer.eval()
    assert not optimizer.param_groups[0]['train_mode']
    optimizer.train()
    torch.testing.assert_close(param, training_param)
