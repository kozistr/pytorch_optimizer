from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pytorch_optimizer import AdamP, Lookahead
from pytorch_optimizer.types import LOSS


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


class MultiHeadLogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.head1 = nn.Linear(2, 1)
        self.head2 = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.fc1(x)
        x = F.relu(x)
        return self.head1(x), self.head2(x)


class Example(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)
        self.norm1 = nn.LayerNorm(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm1(self.fc1(x))


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


def dummy_closure() -> LOSS:
    return 1.0


def build_lookahead(*parameters, **kwargs):
    return Lookahead(AdamP(*parameters, **kwargs))


def ids(v) -> str:
    return f'{v[0].__name__}_{v[1:]}'
