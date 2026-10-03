import pytest
import torch
from torch import nn

from pytorch_optimizer.optimizer import load_optimizer
from tests.fixtures import make_parameter


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
        optimizer = load_optimizer('soap')(parameters, **params)
        for _ in range(2):
            optimizer.step()

    def test_soap_merge_dims_channel_last(self, environment):
        x_data, y_data = environment

        x_data = x_data.reshape(-1, 1, 2, 1).repeat_interleave(2, dim=-1).to(memory_format=torch.channels_last)

        model = nn.Sequential(
            nn.Conv2d(1, 1, 2, 1),
        ).to(x_data.device)

        optimizer = load_optimizer('soap')(
            model.parameters(),
            merge_dims=True,
            precondition_1d=True,
            max_precondition_dim=2,
            precondition_frequency=1,
            data_format='channels_last',
        )

        for _ in range(2):
            optimizer.zero_grad()
            nn.BCEWithLogitsLoss()(model(x_data).squeeze(), y_data.squeeze()).backward()
            optimizer.step()
