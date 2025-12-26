from typing import Tuple

import numpy as np
import pytest
import torch

from pytorch_optimizer.optimizer import AdamW
from tests.utils import Example, simple_parameter


@pytest.fixture(scope='session')
def environment(num_samples: int = 100, dims: int = 2, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    rng = np.random.RandomState(seed)

    x = rng.randn(num_samples, dims) * 2

    mid: int = num_samples // 2
    x[:mid, :] = x[:mid, :] - 2 * np.ones((mid, dims))
    x[mid:, :] = x[mid:, :] + 2 * np.ones((mid, dims))

    y = np.array([0] * mid + [1] * mid).reshape(100, 1)

    return torch.Tensor(x), torch.Tensor(y)


@pytest.fixture
def optimizer_factory():
    return AdamW(Example().parameters())


@pytest.fixture
def param_groups():
    return [{'params': simple_parameter()}]


@pytest.fixture
def binary_predictions():
    return torch.arange(0.0, 1.0, 0.1), torch.FloatTensor([0.0] * 5 + [1.0] * 5)
