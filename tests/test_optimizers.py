from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from pytorch_optimizer.types import LOSS

from pytorch_optimizer import (
    LARS,
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


def dummy_closure() -> LOSS:
    return 1.0


def build_lookahead(*parameters, **kwargs):
    return Lookahead(AdamP(*parameters, **kwargs))


OPTIMIZERS: List[Tuple[Any, Dict[str, Union[float, bool, int]], int]] = [
    (build_lookahead, {'lr': 5e-1, 'weight_decay': 1e-3}, 200),
    (AdaBelief, {'lr': 5e-1, 'weight_decay': 1e-3}, 200),
    (AdaBelief, {'lr': 5e-1, 'weight_decay': 1e-3, 'amsgrad': True}, 200),
    (AdaBelief, {'lr': 5e-1, 'weight_decay': 1e-3, 'weight_decouple': False}, 200),
    (AdaBelief, {'lr': 5e-1, 'weight_decay': 1e-3, 'fixed_decay': True}, 200),
    (AdaBelief, {'lr': 5e-1, 'weight_decay': 1e-3, 'rectify': False}, 200),
    (AdaBound, {'lr': 5e-1, 'gamma': 0.1, 'weight_decay': 1e-3}, 200),
    (AdaBound, {'lr': 5e-1, 'gamma': 0.1, 'weight_decay': 1e-3, 'amsbound': True}, 200),
    (AdamP, {'lr': 5e-1, 'weight_decay': 1e-3}, 200),
    (AdamP, {'lr': 5e-1, 'weight_decay': 1e-3, 'use_gc': True}, 200),
    (AdamP, {'lr': 5e-1, 'weight_decay': 1e-3, 'nesterov': True}, 200),
    (DiffGrad, {'lr': 5e-1, 'weight_decay': 1e-3}, 200),
    (DiffRGrad, {'lr': 5e-1, 'weight_decay': 1e-3}, 200),
    (Lamb, {'lr': 1e-1, 'weight_decay': 1e-3}, 500),
    (Lamb, {'lr': 1e-1, 'weight_decay': 1e-3, 'adam': True, 'eps': 1e-8}, 500),
    (Lamb, {'lr': 1e-1, 'weight_decay': 1e-3, 'pre_norm': True, 'eps': 1e-8}, 500),
    (LARS, {'lr': 1e-1, 'weight_decay': 1e-3}, 500),
    (RaLamb, {'lr': 1e-1, 'weight_decay': 1e-3}, 200),
    (MADGRAD, {'lr': 1e-2, 'weight_decay': 1e-3}, 500),
    (RAdam, {'lr': 1e-1, 'weight_decay': 1e-3}, 200),
    (SGDP, {'lr': 2e-1, 'weight_decay': 1e-3}, 500),
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


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


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


@pytest.mark.parametrize('optimizer_config', OPTIMIZERS, ids=ids)
def test_closure(optimizer_config):
    if optimizer_config[0] == Ranger21:
        return True

    _, model, _ = build_environment()

    optimizer_class, config, _ = optimizer_config
    optimizer = optimizer_class(model.parameters(), **config)

    optimizer.zero_grad()
    optimizer.step(closure=dummy_closure)


@pytest.mark.parametrize('optimizer_fp32_config', OPTIMIZERS, ids=ids)
def test_f32_optimizers(optimizer_fp32_config):
    (x_data, y_data), model, loss_fn = build_environment()

    optimizer_class, config, iterations = optimizer_fp32_config
    optimizer = optimizer_class(model.parameters(), **config)

    init_loss, loss = np.inf, np.inf
    for _ in range(iterations):
        optimizer.zero_grad()

        y_pred = model(x_data)
        loss = loss_fn(y_pred, y_data)

        if init_loss == np.inf:
            init_loss = loss

        loss.backward()

        optimizer.step()

    assert tensor_to_numpy(init_loss) > 2.0 * tensor_to_numpy(loss)


@pytest.mark.parametrize('optimizer_fp16_config', OPTIMIZERS, ids=ids)
def test_f16_optimizers(optimizer_fp16_config):
    (x_data, y_data), model, loss_fn = build_environment()

    optimizer_class, config, iterations = optimizer_fp16_config
    if optimizer_class.__name__ == 'MADGRAD':
        return True

    optimizer = SafeFP16Optimizer(optimizer_class(model.parameters(), **config))

    init_loss, loss = np.inf, np.inf
    for _ in range(1000):
        optimizer.zero_grad()

        y_pred = model(x_data)
        loss = loss_fn(y_pred, y_data)

        if init_loss == np.inf:
            init_loss = loss

        loss.backward()

        optimizer.step()

    assert tensor_to_numpy(init_loss) - 0.01 > tensor_to_numpy(loss)


@pytest.mark.parametrize('adaptive', (False, True))
@pytest.mark.parametrize('optimizer_sam_config', OPTIMIZERS, ids=ids)
def test_sam_optimizers(adaptive, optimizer_sam_config):
    (x_data, y_data), model, loss_fn = build_environment()

    optimizer_class, config, iterations = optimizer_sam_config
    optimizer = SAM(model.parameters(), optimizer_class, **config, adaptive=adaptive)

    init_loss, loss = np.inf, np.inf
    for _ in range(iterations):
        loss = loss_fn(y_data, model(x_data))
        loss.backward()
        optimizer.first_step(zero_grad=True)

        loss_fn(y_data, model(x_data)).backward()
        optimizer.second_step(zero_grad=True)

        if init_loss == np.inf:
            init_loss = loss

    assert tensor_to_numpy(init_loss) > 2.0 * tensor_to_numpy(loss)


@pytest.mark.parametrize('optimizer_adamd_config', ADAMD_SUPPORTED_OPTIMIZERS, ids=ids)
def test_adamd_optimizers(optimizer_adamd_config):
    (x_data, y_data), model, loss_fn = build_environment()

    optimizer_class, config, iterations = optimizer_adamd_config
    optimizer = optimizer_class(model.parameters(), **config)

    init_loss, loss = np.inf, np.inf
    for _ in range(iterations):
        optimizer.zero_grad()

        y_pred = model(x_data)
        loss = loss_fn(y_pred, y_data)

        if init_loss == np.inf:
            init_loss = loss

        loss.backward()

        optimizer.step()

    assert tensor_to_numpy(init_loss) > 2.0 * tensor_to_numpy(loss)


@pytest.mark.parametrize('optimizer_pc_grad_config', OPTIMIZERS, ids=ids)
def test_pc_grad_optimizers(optimizer_pc_grad_config):
    torch.manual_seed(42)

    x_data, y_data = make_dataset()

    model: nn.Module = MultiHeadLogisticRegression()
    loss_fn_1: nn.Module = nn.BCEWithLogitsLoss()
    loss_fn_2: nn.Module = nn.L1Loss()

    optimizer_class, config, iterations = optimizer_pc_grad_config
    optimizer = PCGrad(optimizer_class(model.parameters(), **config))

    init_loss, loss = np.inf, np.inf
    for _ in range(iterations):
        optimizer.zero_grad()
        y_pred_1, y_pred_2 = model(x_data)
        loss1, loss2 = loss_fn_1(y_pred_1, y_data), loss_fn_2(y_pred_2, y_data)

        loss = (loss1 + loss2) / 2.0
        if init_loss == np.inf:
            init_loss = loss

        optimizer.pc_backward([loss1, loss2])
        optimizer.step()

    assert tensor_to_numpy(init_loss) > 1.5 * tensor_to_numpy(loss)


@pytest.mark.parametrize('optimizer_config', OPTIMIZERS, ids=ids)
def test_no_gradients(optimizer_config):
    (x_data, y_data), model, loss_fn = build_environment()

    model.fc1.weight.requires_grad = False
    model.fc1.bias.requires_grad = False

    optimizer_class, config, iterations = optimizer_config
    optimizer = optimizer_class(model.parameters(), **config)

    init_loss, loss = np.inf, np.inf
    for _ in range(iterations):
        optimizer.zero_grad()

        y_pred = model(x_data)
        loss = loss_fn(y_pred, y_data)

        if init_loss == np.inf:
            init_loss = loss

        loss.backward()

        optimizer.step()

    assert tensor_to_numpy(init_loss) >= tensor_to_numpy(loss)
