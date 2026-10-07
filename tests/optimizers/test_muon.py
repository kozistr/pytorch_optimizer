from unittest.mock import patch

import pytest
import torch

from pytorch_optimizer.optimizer import load_optimizer
from pytorch_optimizer.optimizer.utils.matrix import zero_power_via_newton_schulz_5
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


class TestMuon:
    @pytest.mark.parametrize('optimizer_name', ['Muon', 'AdaMuon', 'AdaGO', 'NorMuon'])
    @pytest.mark.parametrize('use_adjusted_lr', [False, True])
    @pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
    def test_muon_matrix_batching(self, optimizer_name, use_adjusted_lr, dtype, device):
        shapes = [(2, 8), (8, 2), (2, 2, 2, 2), (8, 1, 2), (2, 2)]
        params = [make_parameter(shape, dtype=dtype, device=device) for shape in shapes]
        reference_params = [make_parameter(shape, dtype=dtype, device=device) for shape in shapes]
        with torch.no_grad():
            params[2].transpose_(1, 2)
        config = {'lr': 0.1, 'weight_decay': 0.1, 'use_adjusted_lr': use_adjusted_lr, 'ns_steps': 3}
        optimizer = build_optimizer(optimizer_name, [{'params': params, 'use_muon': True}], foreach=True, **config)
        reference = build_optimizer(
            optimizer_name, [{'params': reference_params, 'use_muon': True}], foreach=False, **config
        )

        for iteration in range(3):
            for index, (actual, expected) in enumerate(zip(params, reference_params)):
                if iteration == 1 and index == 1:
                    actual.grad = expected.grad = None
                    continue

                gradient = torch.eye(actual.size(0), actual.numel() // actual.size(0), dtype=dtype, device=device)
                gradient[1].mul_(0.5)
                actual.grad = gradient.reshape(actual.shape).mul(1.0 + 0.25 * iteration)
                expected.grad = actual.grad.clone()

            with patch(
                'pytorch_optimizer.optimizer.muon.zero_power_via_newton_schulz_5',
                wraps=zero_power_via_newton_schulz_5,
            ) as orthogonalize:
                optimizer.step()

            expected_shapes = [(2, 2, 8), (2, 2, 8), (2, 2)] if use_adjusted_lr else [(4, 2, 8), (2, 2)]
            if iteration == 1:
                expected_shapes = [(2, 2, 8), (8, 2), (2, 2)] if use_adjusted_lr else [(3, 2, 8), (2, 2)]
            assert [call.args[0].shape for call in orthogonalize.call_args_list] == expected_shapes

            reference.step()
            for actual, expected in zip(params, reference_params):
                torch.testing.assert_close(actual, expected)
                for key in reference.state[expected]:
                    torch.testing.assert_close(optimizer.state[actual][key], reference.state[expected][key])

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
