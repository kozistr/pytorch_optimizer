import numpy as np
import pytest
import torch

from pytorch_optimizer.optimizer.nero import neuron_mean, neuron_norm
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def test_nero_zero_scale():
    param = make_parameter()

    optimizer = build_optimizer('nero', [param], constraints=False)
    optimizer.zero_grad()

    param.grad = torch.zeros(1, 1)
    optimizer.step()


class TestNeroUtils:
    def test_neuron_mean_norm(self):
        x = torch.arange(-5, 5, dtype=torch.float32)

        with pytest.raises(ValueError) as error_info:
            neuron_mean(x)

        assert str(error_info.value) == '[-] neuron_mean not defined on 1D tensors.'

        np.testing.assert_array_equal(
            neuron_mean(x.view(-1, 1)).numpy(),
            np.asarray([[-5.0], [-4.0], [-3.0], [-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0], [4.0]]),
        )
        np.testing.assert_array_equal(
            neuron_norm(x).numpy(), np.asarray([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
        )
        np.testing.assert_array_equal(
            neuron_norm(x.view(-1, 1)).numpy(),
            np.asarray([[5.0], [4.0], [3.0], [2.0], [1.0], [0.0], [1.0], [2.0], [3.0], [4.0]]),
        )
