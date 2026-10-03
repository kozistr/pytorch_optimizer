import torch
from torch import nn

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def test_stableadamw_optimizer():
    params = [
        make_parameter((2, 2), dtype=torch.float16, grad=None),
        make_parameter((2,), grad=None),
        make_parameter((1, 2), dtype=torch.float16, grad=None),
        make_parameter((1,), grad=None),
    ]
    optimizer = build_optimizer('stableadamw', [{'params': params[:1]}, {'params': params[1:]}])
    optimizer.step()
    params[0].grad = torch.full_like(params[0], 400.0)
    params[1].grad = torch.full_like(params[1], 2.0)
    optimizer.step()

    restored_params = [nn.Parameter(param.detach().clone()) for param in params]
    restored = build_optimizer('stableadamw', [{'params': restored_params[:1]}, {'params': restored_params[1:]}])
    restored.load_state_dict(optimizer.state_dict())

    second_moment = restored.state[restored_params[0]]['exp_avg_sq']
    assert second_moment.dtype == torch.float32
    assert second_moment.device == restored_params[0].device
    torch.testing.assert_close(second_moment, optimizer.state[params[0]]['exp_avg_sq'])
    torch.testing.assert_close(
        restored.state[restored_params[1]]['exp_avg_sq'], optimizer.state[params[1]]['exp_avg_sq']
    )
    assert restored_params[2] not in restored.state
