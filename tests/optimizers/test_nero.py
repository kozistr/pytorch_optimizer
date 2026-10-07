import pytest
import torch

from pytorch_optimizer.optimizer.nero import neuron_mean, neuron_norm
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def test_nero_zero_scale():
    param = make_parameter()

    optimizer = build_optimizer('nero', [param], constraints=False)
    optimizer.step()

    torch.testing.assert_close(param, torch.zeros_like(param))


class TestNeroUtils:
    def test_neuron_mean_norm(self):
        x = torch.arange(-5, 5, dtype=torch.float32)

        with pytest.raises(ValueError, match='neuron_mean not defined on 1D tensors'):
            neuron_mean(x)

        torch.testing.assert_close(neuron_mean(x.view(-1, 1)), x.view(-1, 1))
        torch.testing.assert_close(neuron_norm(x), x.abs())
        torch.testing.assert_close(neuron_norm(x.view(-1, 1)), x.abs().view(-1, 1))
