import pytest
import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


class TestDadapt:
    @pytest.mark.parametrize(
        'optimizer_name', ['DAdaptAdaGrad', 'DAdaptAdam', 'DAdaptSGD', 'DAdaptAdan', 'DAdaptLion', 'Prodigy']
    )
    def test_2nd_stage_gradient(self, optimizer_name):
        params = [make_parameter(grad=grad) for grad in (None, 1.0, 1.0)]
        optimizer = build_optimizer(optimizer_name, [{'params': [param]} for param in params])
        optimizer.step()

        torch.testing.assert_close(params[1], params[2])
        assert params[1].item() < 0.0
