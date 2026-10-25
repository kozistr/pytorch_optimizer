import pytest
import torch
from torch import nn

from pytorch_optimizer.optimizer.rotograd import HAS_GEOTORCH, RotoGrad, RotoGradNorm


@pytest.mark.skipif(not HAS_GEOTORCH, reason='requires geotorch')
@pytest.mark.parametrize('wrapper', [RotoGrad, RotoGradNorm])
def test_rotograd_refreshes_loss_normalization(wrapper, device):
    backbone = nn.Linear(2, 2, bias=False, device=device)
    heads = [nn.Linear(2, 1, bias=False, device=device) for _ in range(2)]
    with torch.no_grad():
        backbone.weight.copy_(torch.eye(2, device=device))
        for head in heads:
            head.weight.fill_(1.0)

    options = {'alpha': 1.0} if wrapper is RotoGradNorm else {}
    model = wrapper(backbone, heads, 2, burn_in_period=1, normalize_losses=True, **options).to(device)
    for index in range(2):
        setattr(model, f'rotation_{index}', torch.eye(2, device=device))
    for target in (0.0, -2.0):
        model.zero_grad()
        backbone.zero_grad()
        for head in heads:
            head.zero_grad()
        outputs = model(torch.tensor([[1.0, 0.0]], device=device))
        model.backward([(output - target).square().mean() for output in outputs])

    for head in heads:
        torch.testing.assert_close(head[1].weight.grad, torch.tensor([[2.0 / 3.0, 0.0]], device=device))
