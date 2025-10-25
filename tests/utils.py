from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.functional import relu
from torch.optim import AdamW

from pytorch_optimizer.base.type import Loss
from pytorch_optimizer.optimizer import Lookahead, OrthoGrad, ScheduleFreeWrapper
from pytorch_optimizer.optimizer.alig import l2_projection


class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = relu(x)
        return self.fc2(x)


class ComplexLogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2, dtype=torch.complex64)
        self.fc2 = nn.Linear(2, 1, dtype=torch.complex64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = relu(x.real) + 1.0j * relu(x.imag)
        return self.fc2(x).real


class MultiHeadLogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.head1 = nn.Linear(2, 1)
        self.head2 = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.fc1(x)
        x = relu(x)
        return self.head1(x), self.head2(x)


class Example(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)
        self.norm1 = nn.LayerNorm(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm1(self.fc1(x))


class MultiClassExample(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def simple_zero_rank_parameter(require_grad: bool = True) -> torch.Tensor:
    param = torch.tensor(0.0).requires_grad_(require_grad)
    param.grad = torch.tensor(0.0)
    return param


def simple_parameter(require_grad: bool = True) -> torch.Tensor:
    param = torch.zeros(1, 1).requires_grad_(require_grad)
    param.grad = torch.zeros(1, 1)
    return param


def simple_complex_parameter(require_grad: bool = True) -> torch.Tensor:
    param = torch.zeros(1, 1, dtype=torch.complex64).requires_grad_(require_grad)
    param.grad = torch.randn(1, 1, dtype=torch.complex64)
    return param


def simple_sparse_parameter(require_grad: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    weight = torch.randn(5, 1).requires_grad_(require_grad)
    weight_sparse = weight.detach().requires_grad_(require_grad)

    if require_grad:
        weight.grad = torch.rand_like(weight)
        weight.grad[0] = 0.0
        weight_sparse.grad = weight.grad.to_sparse()

    return weight, weight_sparse


def dummy_closure() -> Loss:
    return 1.0


def build_lookahead(*parameters, **kwargs):
    return Lookahead(AdamW(*parameters, **kwargs))


def build_orthograd(*parameters, **kwargs):
    return OrthoGrad(AdamW(*parameters, **kwargs))


def build_schedulefree(*parameters, **kwargs):
    return ScheduleFreeWrapper(AdamW(*parameters, **kwargs))


def ids(v) -> str:
    return f'{v[0].__name__}_{v[1:]}'


def names(v) -> str:
    return v.__name__


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def sphere_loss(x: torch.Tensor) -> torch.Tensor:
    return (x ** 2).sum()  # fmt: skip


def build_model(use_complex: bool = False):
    torch.manual_seed(42)
    return ComplexLogisticRegression() if use_complex else LogisticRegression(), nn.BCEWithLogitsLoss()


def build_optimizer_parameter(parameters, optimizer_name, config):
    if optimizer_name == 'AliG':
        config.update({'projection_fn': lambda: l2_projection(parameters, max_norm=1)})
    elif optimizer_name in ('Muon', 'AdaMuon', 'AdaGO'):
        hidden_weights = [p for p in parameters if p.ndim >= 2]
        hidden_gains_biases = [p for p in parameters if p.ndim < 2]

        parameters = [
            {'params': hidden_weights, 'use_muon': True},
            {'params': hidden_gains_biases, 'use_muon': False},
        ]
    elif optimizer_name == 'AdamWSN':
        sn_params = [p for p in parameters if p.ndim == 2]
        regular_params = [p for p in parameters if p.ndim != 2]
        parameters = [{'params': sn_params, 'sn': True}, {'params': regular_params, 'sn': False}]
    elif optimizer_name == 'AdamC':
        norm_params = [p for i, p in enumerate(parameters) if i == 1]
        regular_params = [p for i, p in enumerate(parameters) if i != 1]
        parameters = [{'params': norm_params, 'normalized': True}, {'params': regular_params}]

    return parameters, config
