import pytest
import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


class TestSoap:
    @pytest.mark.parametrize(
        'params',
        [
            {'merge_dims': True, 'precondition_1d': True, 'max_precondition_dim': 4, 'precondition_frequency': 1},
            {
                'merge_dims': True,
                'precondition_1d': False,
                'max_precondition_dim': 1,
                'precondition_frequency': 1,
                'normalize_gradient': True,
            },
        ],
    )
    def test_soap_parameters(self, params):
        parameters = [make_parameter(shape) for shape in ((8, 2), (8,), (1, 8))]
        parameters.append(make_parameter((1,), grad=None))
        optimizer = build_optimizer('soap', parameters, **params)
        for _ in range(2):
            optimizer.step()

        torch.testing.assert_close(parameters, [torch.zeros_like(param) for param in parameters])

    @pytest.mark.parametrize('max_precondition_dim', [1, 8])
    def test_soap_merge_dims_channel_last(self, max_precondition_dim):
        param = make_parameter((2, 3, 4, 5), dtype=torch.float64)
        reference = param.detach().permute(0, 3, 1, 2).contiguous().requires_grad_()
        options = {
            'merge_dims': True,
            'precondition_1d': True,
            'max_precondition_dim': max_precondition_dim,
            'precondition_frequency': 1,
        }
        optimizer = build_optimizer(
            'soap', [param], data_format='channels_last', **options
        )
        reference_optimizer = build_optimizer('soap', [reference], **options)

        for step in range(3):
            param.grad = (torch.arange(param.numel(), dtype=param.dtype).reshape_as(param) + step).sin()
            reference.grad = param.grad.permute(0, 3, 1, 2).contiguous()
            optimizer.step()
            reference_optimizer.step()

            torch.testing.assert_close(param.permute(0, 3, 1, 2), reference)
