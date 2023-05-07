from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as f

from pytorch_optimizer import AdamP, Lookahead
from pytorch_optimizer.base.types import LOSS


class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = f.relu(x)
        return self.fc2(x)


class MultiHeadLogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.head1 = nn.Linear(2, 1)
        self.head2 = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.fc1(x)
        x = f.relu(x)
        return self.head1(x), self.head2(x)


class Example(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)
        self.norm1 = nn.LayerNorm(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm1(self.fc1(x))


def simple_zero_rank_parameter(require_grad: bool = True) -> torch.Tensor:
    param = torch.tensor(0.0).requires_grad_(require_grad)
    param.grad = torch.tensor(0.0)
    return param


def simple_parameter(require_grad: bool = True) -> torch.Tensor:
    param = torch.zeros(1, 1).requires_grad_(require_grad)
    param.grad = torch.zeros(1, 1)
    return param


def simple_sparse_parameter(require_grad: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    weight = torch.randn(5, 1).requires_grad_(require_grad)
    weight_sparse = weight.detach().requires_grad_(require_grad)

    if require_grad:
        weight.grad = torch.rand_like(weight)
        weight.grad[0] = 0.0
        weight_sparse.grad = weight.grad.to_sparse()

    return weight, weight_sparse


def make_dataset(num_samples: int = 100, dims: int = 2, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
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

    x = torch.Tensor(x)
    y = torch.Tensor(y)

    return x, y


def dummy_closure() -> LOSS:
    return 1.0


def build_lookahead(*parameters, **kwargs):
    return Lookahead(AdamP(*parameters, **kwargs))


def ids(v) -> str:
    return f'{v[0].__name__}_{v[1:]}'


def names(v) -> str:
    return v.__name__


def build_environment(use_gpu: bool = False) -> Tuple[Tuple[torch.Tensor, torch.Tensor], nn.Module, nn.Module]:
    torch.manual_seed(42)

    x_data, y_data = make_dataset()
    model: nn.Module = LogisticRegression()
    loss_fn: nn.Module = nn.BCEWithLogitsLoss()

    if use_gpu and torch.cuda.is_available():
        x_data, y_data = x_data.cuda(), y_data.cuda()
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    return (x_data, y_data), model, loss_fn
