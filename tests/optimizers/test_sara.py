import pytest
import torch

from tests.utils import build_optimizer


@pytest.mark.parametrize('foreach', [False, True])
def test_sara_optimizer(device, foreach):
    param = torch.full((65, 65), 0.01, dtype=torch.bfloat16, device=device).t().requires_grad_()
    param.grad = torch.ones_like(param)
    optimizer = build_optimizer('sara', [param], lr=1e-3, threshold=0.1, lambda_rank=5e-4, foreach=foreach)

    optimizer.step()
    optimizer.load_state_dict(optimizer.state_dict())
    assert optimizer.state[param]['mask'].dtype == torch.bool

    optimizer.step()
    assert torch.isfinite(param).all()
