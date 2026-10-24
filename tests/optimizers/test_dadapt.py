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

    @pytest.mark.parametrize('foreach', [False, True])
    def test_adam_coupled_weight_decay_matches_l2_penalty(self, foreach):
        optimizer_name = 'dadaptadam'
        # coupled weight decay is the gradient of an L2 penalty, so it must equal adding weight_decay * p
        # to the gradient yourself
        weight_decay = 0.2
        grads = [torch.tensor([0.5, -1.0, 2.0]), torch.tensor([1.5, 0.25, -0.75])]

        decayed = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))
        penalized = torch.nn.Parameter(decayed.detach().clone())
        decayed_optimizer = build_optimizer(
            optimizer_name, [decayed], lr=1.0, d0=0.1, weight_decay=weight_decay, weight_decouple=False,
            foreach=foreach,
        )
        penalized_optimizer = build_optimizer(optimizer_name, [penalized], lr=1.0, d0=0.1, foreach=foreach)

        for grad in grads:
            decayed.grad = grad.clone()
            penalized.grad = grad + weight_decay * penalized.detach()
            decayed_optimizer.step()
            penalized_optimizer.step()

        torch.testing.assert_close(decayed, penalized)
        assert not torch.equal(decayed, torch.tensor([1.0, -2.0, 0.5]))

    @pytest.mark.parametrize('foreach', [False, True])
    def test_sgd_coupled_weight_decay(self, foreach):
        # first step: d stays d0 and p = p0 - (1 - momentum) * d0 * lr / ||g|| * (g + weight_decay * p0),
        # with ||g|| the norm of the raw gradient (as in the reference implementation)
        weight_decay, momentum, d0 = 0.2, 0.9, 0.1
        grad = torch.tensor([0.5, -1.0, 2.0])
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))
        p0 = parameter.detach().clone()
        expected = p0 - (1.0 - momentum) * d0 / grad.norm() * (grad + weight_decay * p0)

        optimizer = build_optimizer(
            'dadaptsgd', [parameter], lr=1.0, d0=d0, momentum=momentum, weight_decay=weight_decay,
            weight_decouple=False, foreach=foreach,
        )
        parameter.grad = grad.clone()
        optimizer.step()

        torch.testing.assert_close(parameter.detach(), expected)

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
