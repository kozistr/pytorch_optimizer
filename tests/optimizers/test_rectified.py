import pytest
import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


@pytest.mark.parametrize('optimizer_name', ['adabelief', 'radam', 'lamb', 'diffgrad', 'ranger'])
def test_rectified_optimizer(optimizer_name):
    param = make_parameter(grad=1.0)

    parameters = {'n_sma_threshold': 1000, 'degenerated_to_sgd': False}
    if optimizer_name not in ('radam', 'ranger'):
        parameters.update({'rectify': True})

    optimizer = build_optimizer(optimizer_name, [param], **parameters)
    optimizer.step()

    torch.testing.assert_close(param, torch.zeros_like(param))
