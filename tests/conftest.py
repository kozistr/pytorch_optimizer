from typing import Tuple

import numpy as np
import pytest
import torch


@pytest.fixture(scope='session')
def environment(num_samples: int = 100, dims: int = 2, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    rng = np.random.RandomState(seed)

    x = rng.randn(num_samples, dims) * 2

    # center the first N/2 points at (-2, -2)
    mid: int = num_samples // 2
    x[:mid, :] = x[:mid, :] - 2 * np.ones((mid, dims))

    # center the last N/2 points at (2, 2)
    x[mid:, :] = x[mid:, :] + 2 * np.ones((mid, dims))

    # labels: first N/2 are 0, last N/2 are 1
    y = np.array([0] * mid + [1] * mid).reshape(100, 1)

    return torch.Tensor(x), torch.Tensor(y)
