import pytest
import torch

from pytorch_optimizer.optimizer import load_optimizer
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


class TestMuon:
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
