import pytest
import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


@pytest.mark.parametrize('foreach', [False, True])
def test_sgdw_delayed_momentum(foreach):
    parameters = [make_parameter((2,)), make_parameter((2,), grad=None)]
    optimizer = build_optimizer('sgdw', parameters, lr=0.1, momentum=0.9, dampening=0.2, foreach=foreach)

    steps = [
        (0.0, (0.0, 0.0), (0.0, None)),
        (1.0, (-0.08, -0.1), (0.8, 1.0)),
        (-2.0, (0.008, -0.03), (-0.88, -0.7)),
    ]

    for iteration, (gradient, expected_params, expected_momentum) in enumerate(steps):
        for index, param in enumerate(parameters):
            param.grad = None if index == 1 and iteration == 0 else torch.full_like(param, gradient)

        optimizer.step()

        for param, expected, momentum in zip(parameters, expected_params, expected_momentum):
            torch.testing.assert_close(param, torch.full_like(param, expected))
            if momentum is None:
                assert 'momentum_buffer' not in optimizer.state[param]
            else:
                torch.testing.assert_close(
                    optimizer.state[param]['momentum_buffer'], torch.full_like(param, momentum)
                )
