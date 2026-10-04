import pytest
import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


@pytest.mark.parametrize('foreach', [False, True])
def test_sgdw_delayed_momentum(foreach):
    parameters = [make_parameter((2,)), make_parameter((2,), grad=None)]
    references = [make_parameter((2,)), make_parameter((2,), grad=None)]
    options = {'lr': 0.1, 'momentum': 0.9, 'dampening': 0.2, 'foreach': foreach}
    optimizer = build_optimizer('sgdw', parameters, **options)
    reference = torch.optim.SGD(references, **options)

    for iteration, gradient in enumerate((0.0, 1.0, -2.0)):
        for index, (param, expected) in enumerate(zip(parameters, references)):
            param.grad = None if index == 1 and iteration == 0 else torch.full_like(param, gradient)
            expected.grad = None if param.grad is None else param.grad.clone()
        optimizer.step()
        reference.step()
        for param, expected in zip(parameters, references):
            torch.testing.assert_close(param, expected)
            if param.grad is not None:
                torch.testing.assert_close(
                    optimizer.state[param]['momentum_buffer'], reference.state[expected]['momentum_buffer']
                )
