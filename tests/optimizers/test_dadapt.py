import pytest
import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


class TestDadapt:
    @pytest.mark.parametrize(
        ('optimizer_name', 'expected'),
        [('dadaptadam', 0.8 - 0.005 / (0.00025 ** 0.5 + 1e-8)), ('dadaptsgd', 0.81)],
    )
    def test_decoupled_weight_decay(self, optimizer_name, expected):
        parameter = make_parameter(grad=0.5)

        with torch.no_grad():
            parameter.fill_(1.0)

        optimizer = build_optimizer(
            optimizer_name, [parameter], lr=1.0, d0=0.1,
            weight_decay=0.2, weight_decouple=True, fixed_decay=True, foreach=True,
        )
        optimizer.step()

        torch.testing.assert_close(parameter, torch.full_like(parameter, expected))

    @pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
    def test_adan_low_precision_decay(self, dtype):
        params = [make_parameter((2,), dtype=dtype) for _ in range(2)]
        with torch.no_grad():
            for param in params:
                param.copy_(torch.tensor([0.3, 0.6], dtype=dtype))

        optimizers = [
            build_optimizer(
                'dadaptadan', [param], lr=0.3, d0=0.1, eps=1e-6,
                weight_decay=0.1, weight_decouple=True, foreach=foreach,
            )
            for param, foreach in zip(params, (False, True))
        ]
        for _ in range(3):
            for param, optimizer in zip(params, optimizers):
                param.grad = torch.full_like(param, 0.5)
                optimizer.step()

        torch.testing.assert_close(params[0], params[1], rtol=0.0, atol=0.0)

    @pytest.mark.parametrize(
        'optimizer_name', ['DAdaptAdaGrad', 'DAdaptAdam', 'DAdaptSGD', 'DAdaptAdan', 'DAdaptLion', 'Prodigy']
    )
    def test_2nd_stage_gradient(self, optimizer_name):
        params = [make_parameter(grad=grad) for grad in (None, 1.0, 1.0)]
        optimizer = build_optimizer(optimizer_name, [{'params': [param]} for param in params])
        optimizer.step()

        torch.testing.assert_close(params[1], params[2])
        assert params[1].item() < 0.0
