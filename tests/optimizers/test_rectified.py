import pytest
import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


@pytest.mark.parametrize('optimizer_name', ['adabelief', 'radam', 'lamb', 'diffgrad', 'ranger'])
def test_rectified_optimizer(optimizer_name):
    param = make_parameter()

    parameters = {'n_sma_threshold': 1000, 'degenerated_to_sgd': False}
    if optimizer_name not in ('adabelief', 'radam', 'ranger'):
        parameters.update({'rectify': True})

    optimizer = build_optimizer(optimizer_name, [param], **parameters)
    optimizer.zero_grad()

    param.grad = torch.zeros(1, 1)
    optimizer.step()
