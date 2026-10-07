import math

import pytest
import torch

from pytorch_optimizer.optimizer import load_optimizer
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


class TestMuon:
    @pytest.mark.parametrize('foreach', [False, True])
    @pytest.mark.parametrize('use_adjusted_lr', [False, True])
    def test_adamuon_update_scale(self, foreach, use_adjusted_lr):
        params = [make_parameter(shape, grad=1.0) for shape in ((16, 16), (4, 8), (8, 4))]
        lr = 0.1
        optimizer = build_optimizer(
            'adamuon', [{'params': params, 'use_muon': True}], lr=lr,
            foreach=foreach, use_adjusted_lr=use_adjusted_lr,
        )

        optimizer.step()

        for param in params:
            expected_rms = lr / math.sqrt(param.size(1)) if use_adjusted_lr else 0.2 * lr
            torch.testing.assert_close(
                param.square().mean().sqrt(), torch.tensor(expected_rms), rtol=5e-3, atol=0.0
            )

    @pytest.mark.parametrize('optimizer_name', ['Muon', 'AdaMuon', 'AdaGO', 'NorMuon'])
    def test_muon_high_dimensions(self, optimizer_name):
        matrices = [make_parameter(shape, grad=1.0) for shape in ((1, 1, 1), (1, 1, 2, 2), (4, 1))]
        params = [
            {'params': [*matrices, make_parameter(grad=None)], 'use_muon': True},
            {'params': [make_parameter((1,), grad=None)], 'use_muon': False},
        ]
        optimizer = build_optimizer(optimizer_name, params, cautious=optimizer_name in ('Muon', 'AdaGO'))
        optimizer.step()
        assert all(torch.isfinite(param).all() for param in matrices)

    @pytest.mark.parametrize('optimizer_name', ['Muon', 'AdaMuon', 'AdaGO', 'NorMuon'])
    def test_muon_use_muon_param(self, optimizer_name):
        with pytest.raises(ValueError):
            load_optimizer(optimizer_name)([{'params': [make_parameter()]}])

    @pytest.mark.parametrize('optimizer_name', ['Muon', 'AdaMuon', 'AdaGO', 'NorMuon'])
    def test_muon_invalid_ns_coeffs(self, optimizer_name):
        with pytest.raises(ValueError):
            load_optimizer(optimizer_name)(
                [{'params': [make_parameter((2, 2))], 'use_muon': True}], ns_coeffs='invalid'
            )
