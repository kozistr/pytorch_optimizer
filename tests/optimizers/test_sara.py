import torch

from tests.utils import build_optimizer


def test_sara_optimizer(device):
    param = torch.full((65, 65), 0.01, dtype=torch.bfloat16, device=device, requires_grad=True)
    param.grad = torch.ones_like(param)
    optimizer = build_optimizer('sara', [param], lr=1e-3, threshold=0.1, lambda_rank=5e-4)

    optimizer.step()
    optimizer.load_state_dict(optimizer.state_dict())
    assert optimizer.state[param]['mask'].dtype == torch.bool

    optimizer.step()
    assert torch.isfinite(param).all()
