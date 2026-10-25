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

    def test_adan_low_precision_decay(self):
        parameter = make_parameter(dtype=torch.bfloat16, grad=0.5)
        with torch.no_grad():
            parameter.fill_(0.3)

        optimizer = build_optimizer(
            'dadaptadan', [parameter], lr=0.3, d0=0.1, eps=1e-6,
            weight_decay=0.1, weight_decouple=True, foreach=True,
        )
        optimizer.step()

        torch.testing.assert_close(parameter, torch.full_like(parameter, 0.2890625), rtol=0.0, atol=0.0)

    @pytest.mark.parametrize(
        'optimizer_name', ['DAdaptAdaGrad', 'DAdaptAdam', 'DAdaptSGD', 'DAdaptAdan', 'DAdaptLion', 'Prodigy']
    )
    def test_2nd_stage_gradient(self, optimizer_name):
        params = [make_parameter(grad=grad) for grad in (None, 1.0, 1.0)]
        optimizer = build_optimizer(optimizer_name, [{'params': [param]} for param in params])
        optimizer.step()

        torch.testing.assert_close(params[1], params[2])
        assert params[1].item() < 0.0
