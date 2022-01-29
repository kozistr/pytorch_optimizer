from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from pytorch_optimizer import (
    MADGRAD,
    SAM,
    SGDP,
    AdaBelief,
    AdaBound,
    AdamP,
    DiffGrad,
    DiffRGrad,
    Lamb,
    Lookahead,
    PCGrad,
    RAdam,
    RaLamb,
    Ranger,
    Ranger21,
    SafeFP16Optimizer,
)

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


FP32_OPTIMIZERS: List[Tuple[Any, Dict[str, Union[float, bool, int]], int]] = [
    (build_lookahead, {'lr': 1e-2, 'weight_decay': 1e-3}, 200),
    (AdaBelief, {'lr': 1e-2, 'weight_decay': 1e-3}, 200),
    (AdaBound, {'lr': 1e-2, 'gamma': 0.1, 'weight_decay': 1e-3}, 200),
    (AdaBound, {'lr': 1e-2, 'gamma': 0.1, 'weight_decay': 1e-3, 'amsbound': True}, 200),
    (AdamP, {'lr': 1e-3, 'weight_decay': 1e-3}, 800),
    (DiffGrad, {'lr': 1e-2, 'weight_decay': 1e-3}, 200),
    (DiffRGrad, {'lr': 1e-1, 'weight_decay': 1e-3}, 200),
    (Lamb, {'lr': 1e-1, 'weight_decay': 1e-3}, 500),
    (RaLamb, {'lr': 1e-3, 'weight_decay': 1e-3}, 500),
    (MADGRAD, {'lr': 1e-2, 'weight_decay': 1e-3}, 200),
    (RAdam, {'lr': 1e-1, 'weight_decay': 1e-3}, 200),
    (SGDP, {'lr': 1e-1, 'weight_decay': 1e-3}, 200),
    (Ranger, {'lr': 1e-1, 'weight_decay': 1e-3}, 200),
    (Ranger21, {'lr': 5e-1, 'weight_decay': 1e-3, 'num_iterations': 500}, 500),
]

FP16_OPTIMIZERS: List[Tuple[Any, Dict[str, Union[float, bool, int]], int]] = [
    (build_lookahead, {'lr': 5e-1, 'weight_decay': 1e-3}, 500),
    (AdaBelief, {'lr': 5e-1, 'weight_decay': 1e-3}, 200),
    (AdaBound, {'lr': 5e-1, 'gamma': 0.1, 'weight_decay': 1e-3}, 200),
    (AdaBound, {'lr': 1e-1, 'gamma': 0.1, 'weight_decay': 1e-3, 'amsbound': True}, 200),
    (AdamP, {'lr': 5e-1, 'weight_decay': 1e-3}, 500),
    (DiffGrad, {'lr': 5e-1, 'weight_decay': 1e-3}, 500),
    (DiffRGrad, {'lr': 1e-1, 'weight_decay': 1e-3}, 200),
    (Lamb, {'lr': 1e-1, 'weight_decay': 1e-3}, 200),
    (RaLamb, {'lr': 1e-1, 'weight_decay': 1e-3}, 500),
    (RAdam, {'lr': 1e-1, 'weight_decay': 1e-3}, 200),
    (SGDP, {'lr': 5e-1, 'weight_decay': 1e-3}, 500),
    (Ranger, {'lr': 5e-1, 'weight_decay': 1e-3}, 200),
    (Ranger21, {'lr': 5e-1, 'weight_decay': 1e-3, 'num_iterations': 500}, 500),
]

ADAMD_SUPPORTED_OPTIMIZERS: List[Tuple[Any, Dict[str, Union[float, bool, int]], int]] = [
    (build_lookahead, {'lr': 5e-1, 'weight_decay': 1e-3, 'adamd_debias_term': True}, 500),
    (AdaBelief, {'lr': 5e-1, 'weight_decay': 1e-3, 'adamd_debias_term': True}, 200),
    (AdaBound, {'lr': 5e-1, 'gamma': 0.1, 'weight_decay': 1e-3, 'adamd_debias_term': True}, 200),
    (AdaBound, {'lr': 1e-2, 'gamma': 0.1, 'weight_decay': 1e-3, 'amsbound': True, 'adamd_debias_term': True}, 200),
    (AdamP, {'lr': 5e-1, 'weight_decay': 1e-3, 'adamd_debias_term': True}, 500),
    (DiffGrad, {'lr': 15 - 1, 'weight_decay': 1e-3, 'adamd_debias_term': True}, 500),
    (DiffRGrad, {'lr': 1e-1, 'weight_decay': 1e-3, 'adamd_debias_term': True}, 200),
    (Lamb, {'lr': 1e-1, 'weight_decay': 1e-3, 'adamd_debias_term': True}, 300),
    (RaLamb, {'lr': 1e-1, 'weight_decay': 1e-3, 'adamd_debias_term': True}, 500),
    (RAdam, {'lr': 1e-1, 'weight_decay': 1e-3, 'adamd_debias_term': True}, 200),
    (Ranger, {'lr': 5e-1, 'weight_decay': 1e-3, 'adamd_debias_term': True}, 200),
]


@pytest.mark.parametrize('optimizer_fp32_config', FP32_OPTIMIZERS, ids=ids)
def test_f32_optimizers(optimizer_fp32_config):
    torch.manual_seed(42)

    x_data, y_data = make_dataset()

    model: nn.Module = LogisticRegression()
    loss_fn: nn.Module = nn.BCEWithLogitsLoss()

    optimizer_class, config, iterations = optimizer_fp32_config
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


@pytest.mark.parametrize('optimizer_fp16_config', FP16_OPTIMIZERS, ids=ids)
def test_f16_optimizers(optimizer_fp16_config):
    torch.manual_seed(42)

    x_data, y_data = make_dataset()

    model: nn.Module = LogisticRegression()
    loss_fn: nn.Module = nn.BCEWithLogitsLoss()

    optimizer_class, config, iterations = optimizer_fp16_config
    optimizer = SafeFP16Optimizer(optimizer_class(model.parameters(), **config))

    loss: float = np.inf
    init_loss: float = np.inf
    for _ in range(1000):
        optimizer.zero_grad()

        y_pred = model(x_data)
        loss = loss_fn(y_pred, y_data)

        if init_loss == np.inf:
            init_loss = loss

        loss.backward()

        optimizer.step()

    assert init_loss - 0.01 > loss


@pytest.mark.parametrize('optimizer_sam_config', FP32_OPTIMIZERS, ids=ids)
def test_sam_optimizers(optimizer_sam_config):
    torch.manual_seed(42)

    x_data, y_data = make_dataset()

    model: nn.Module = LogisticRegression()
    loss_fn: nn.Module = nn.BCEWithLogitsLoss()

    optimizer_class, config, iterations = optimizer_sam_config
    optimizer = SAM(model.parameters(), optimizer_class, **config)

    loss: float = np.inf
    init_loss: float = np.inf
    for _ in range(iterations):
        loss = loss_fn(y_data, model(x_data))
        loss.backward()
        optimizer.first_step(zero_grad=True)

        loss_fn(y_data, model(x_data)).backward()
        optimizer.second_step(zero_grad=True)

        if init_loss == np.inf:
            init_loss = loss

    assert init_loss > 2.0 * loss


@pytest.mark.parametrize('optimizer_pc_grad_config', FP32_OPTIMIZERS, ids=ids)
def test_pc_grad_optimizers(optimizer_pc_grad_config):
    torch.manual_seed(42)

    x_data, y_data = make_dataset()

    model: nn.Module = MultiHeadLogisticRegression()
    loss_fn_1: nn.Module = nn.BCEWithLogitsLoss()
    loss_fn_2: nn.Module = nn.L1Loss()

    optimizer_class, config, iterations = optimizer_pc_grad_config
    optimizer = PCGrad(optimizer_class(model.parameters(), **config))

    loss: float = np.inf
    init_loss: float = np.inf
    for _ in range(iterations):
        optimizer.zero_grad()
        y_pred_1, y_pred_2 = model(x_data)
        loss1, loss2 = loss_fn_1(y_pred_1, y_data), loss_fn_2(y_pred_2, y_data)

        loss = (loss1 + loss2) / 2.0
        if init_loss == np.inf:
            init_loss = loss

        optimizer.pc_backward([loss1, loss2])
        optimizer.step()

    assert init_loss > 2.0 * loss


@pytest.mark.parametrize('optimizer_adamd_config', ADAMD_SUPPORTED_OPTIMIZERS, ids=ids)
def test_adamd_optimizers(optimizer_adamd_config):
    torch.manual_seed(42)

    x_data, y_data = make_dataset()

    model: nn.Module = LogisticRegression()
    loss_fn: nn.Module = nn.BCEWithLogitsLoss()

    optimizer_class, config, iterations = optimizer_adamd_config
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
