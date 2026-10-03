import pytest
import torch

from pytorch_optimizer.optimizer.scion import build_lmo_norm
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


class TestScion:
    @pytest.mark.parametrize('lmo_type', list(range(9)))
    def test_build_lmo_types(self, lmo_type):
        build_lmo_norm(lmo_type)

    def test_scion_lmo_types(self):
        params = [make_parameter(), make_parameter((1,))]

        build_optimizer('scion', params).init()
        build_optimizer('scionlight', params).init()

        grad_1d = torch.ones(1)
        grad_2d = torch.ones(1, 1)
        grad_4d = torch.ones(1, 1, 1, 1)
        grad_5d = torch.ones(1, 1, 1, 1, 1)

        norm = build_lmo_norm(norm_type=0)
        norm.init(grad_2d)
        norm.lmo(grad_2d)

        norm = build_lmo_norm(norm_type=1, max_scale=True)
        for grad in (grad_1d, grad_2d, grad_4d):
            norm.init(grad)
            norm.lmo(grad)

        with pytest.raises(NotImplementedError):
            norm.init(grad_5d)

        with pytest.raises(NotImplementedError):
            norm.lmo(grad_5d)

        norm = build_lmo_norm(norm_type=2, max_scale=True)
        norm.init(grad_2d)
        norm.lmo(grad_2d)

        norm = build_lmo_norm(norm_type=4, zero_init=True)
        norm.init(grad_2d)
        norm.lmo(grad_2d)

        norm = build_lmo_norm(norm_type=4, zero_init=False)
        norm.init(grad_2d)
        norm.lmo(grad_2d)

        norm = build_lmo_norm(norm_type=6, normalized=True, transpose=True)
        norm.init(grad_2d)
        norm.lmo(grad_2d)

        norm = build_lmo_norm(norm_type=7, normalized=True, transpose=True)
        norm.init(grad_2d)
        norm.lmo(grad_2d)
