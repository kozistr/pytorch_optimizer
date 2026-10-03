import pytest
import torch
from torch import nn

from tests.utils import build_optimizer


class TestSignSgd:
    @pytest.mark.parametrize('foreach', [False, True])
    def test_sign_sgd_preserves_momentum_buffer(self, foreach):
        param = nn.Parameter(torch.tensor([0.0]))
        optimizer = build_optimizer('signsgd', [param], lr=1.0, momentum=0.9, foreach=foreach)

        param.grad = torch.tensor([1.0])
        optimizer.step()

        param.grad = torch.tensor([-0.1])
        optimizer.step()

        assert torch.allclose(optimizer.state[param]['momentum_buffer'], torch.tensor([0.08]))

    @pytest.mark.parametrize('foreach', [False, True])
    @pytest.mark.parametrize(
        ('weight_decay', 'weight_decouple', 'expected'), [(0.2, True, 1.96), (0.2, False, 1.9), (0.0, True, 2.0)]
    )
    def test_sign_sgd_weight_decay(self, foreach, weight_decay, weight_decouple, expected):
        param = nn.Parameter(torch.tensor([2.0]))
        optimizer = build_optimizer(
            'signsgd',
            [param],
            lr=0.1,
            momentum=0.9,
            weight_decay=weight_decay,
            weight_decouple=weight_decouple,
            foreach=foreach,
        )

        param.grad = torch.tensor([0.0])
        optimizer.step()

        assert torch.allclose(param, torch.tensor([expected]))

    @pytest.mark.parametrize(('optimizer_name', 'kwargs'), [('signsgd', {'momentum': 0.1}), ('tiger', {'beta': 0.1})])
    def test_sign_based_foreach_parity(self, optimizer_name, kwargs):
        def run(foreach):
            param = nn.Parameter(torch.tensor([2.0]))
            optimizer = build_optimizer(optimizer_name, [param], lr=0.1, foreach=foreach, **kwargs)
            for grad in (1.0, -0.1):
                param.grad = torch.tensor([grad])
                optimizer.step()
            return param.item()

        assert run(False) == run(True)
