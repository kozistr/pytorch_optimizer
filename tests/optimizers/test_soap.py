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

    def test_soap_merge_dims_channel_last(self):
        param = make_parameter((1, 1, 2, 2), grad=1.0)
        optimizer = build_optimizer(
            'soap',
            [param],
            merge_dims=True,
            precondition_1d=True,
            max_precondition_dim=2,
            precondition_frequency=1,
            data_format='channels_last',
        )

        for _ in range(2):
            optimizer.step()

        assert torch.isfinite(param).all()
