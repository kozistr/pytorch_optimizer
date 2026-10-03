import pytest
import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


class TestDadapt:
    @pytest.mark.parametrize('optimizer_name', ['DAdaptAdaGrad', 'DAdaptAdam', 'DAdaptSGD', 'DAdaptAdan', 'Prodigy'])
    def test_no_progression(self, optimizer_name):
        param = make_parameter()
        param.grad = None

        optimizer = build_optimizer(optimizer_name, [param])
        optimizer.zero_grad()
        optimizer.step()

    @pytest.mark.parametrize(
        'optimizer_name', ['DAdaptAdaGrad', 'DAdaptAdam', 'DAdaptSGD', 'DAdaptAdan', 'DAdaptLion', 'Prodigy']
    )
    def test_2nd_stage_gradient(self, optimizer_name):
        p1 = make_parameter(requires_grad=False)
        p2 = make_parameter(requires_grad=True)
        p3 = make_parameter(requires_grad=True)
        params = [{'params': [p1]}, {'params': [p2]}, {'params': [p3]}]

        optimizer = build_optimizer(optimizer_name, params)
        optimizer.zero_grad()

        p1.grad = None
        p2.grad = torch.randn(1, 1)
        p3.grad = torch.randn(1, 1)

        optimizer.step()
