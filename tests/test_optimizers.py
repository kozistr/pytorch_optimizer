from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from pytorch_optimizer import (
    MADGRAD,
    SGDP,
    AdaBelief,
    AdaBound,
    AdaHessian,
    AdamP,
    DiffGrad,
    DiffRGrad,
    Lamb,
    Lookahead,
    RAdam,
    Ranger,
)
from pytorch_optimizer.types import BETAS

__REFERENCE__ = 'https://github.com/jettify/pytorch-optimizer/blob/master/tests/test_optimizer_with_nn.py'


class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def make_dataset(num_samples: int = 100, dims: int = 2, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(seed)

    x = rng.randn(num_samples, dims) * 2

    # center the first N/2 points at (-2, -2)
    mid: int = num_samples // 2
    x[:mid, :] = x[:mid, :] - 2 * np.ones((mid, dims))

    # center the last N/2 points at (2, 2)
    x[mid:, :] = x[mid:, :] + 2 * np.ones((mid, dims))

    # labels: first N/2 are 0, last N/2 are 1
    y = np.array([0] * mid + [1] * mid).reshape(100, 1)

    x = torch.Tensor(x)
    y = torch.Tensor(y)

    return x, y


def ids(v) -> str:
    return f'{v[0].__name__}_{v[1:]}'


def build_lookahead(*parameters, **kwargs):
    return Lookahead(AdamP(*parameters, **kwargs))


OPTIMIZERS: List[Tuple[Any, Dict[str, Union[float, bool, int, BETAS]], int]] = [
    (build_lookahead, {'lr': 1e-2, 'weight_decay': 1e-3}, 200),
    (AdaBelief, {'lr': 1e-2, 'weight_decay': 1e-3}, 200),
    (AdaBound, {'lr': 1e-2, 'gamma': 0.1, 'weight_decay': 1e-3}, 200),
    (AdamP, {'lr': 1e-3, 'weight_decay': 1e-3}, 800),
    (DiffGrad, {'lr': 1e-2, 'weight_decay': 1e-3}, 200),
    (DiffRGrad, {'lr': 1e-1, 'weight_decay': 1e-3}, 200),
    (Lamb, {'lr': 1e-2, 'weight_decay': 1e-3}, 1000),
    (MADGRAD, {'lr': 1e-2, 'weight_decay': 1e-3}, 200),
    (RAdam, {'lr': 1e-1, 'weight_decay': 1e-3}, 200),
    (SGDP, {'lr': 1e-1, 'weight_decay': 1e-3}, 200),
    (Ranger, {'lr': 1e-1, 'weight_decay': 1e-3}, 200),
    # (AdaHessian, {'lr': 1e-2, 'weight_decay': 1e-3}, 200),
]


@pytest.mark.parametrize('optimizer_config', OPTIMIZERS, ids=ids)
def test_optimizers(optimizer_config):
    torch.manual_seed(42)

    x_data, y_data = make_dataset()

    model: nn.Module = LogisticRegression()
    loss_fn: nn.Module = nn.BCEWithLogitsLoss()

    optimizer_class, config, iterations = optimizer_config
    optimizer = optimizer_class(model.parameters(), **config)

    loss: float = np.inf
    init_loss: float = np.inf
    for _ in range(iterations):
        optimizer.zero_grad()

        y_pred = model(x_data)
        loss = loss_fn(y_pred, y_data)

        if init_loss == np.inf:
            init_loss = loss

        loss.backward()

        optimizer.step()

    assert init_loss > 2.0 * loss
