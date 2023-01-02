from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pytest
import torch
from torch import nn

from pytorch_optimizer import (
    LARS,
    MADGRAD,
    PNM,
    SAM,
    SGDP,
    AdaBelief,
    AdaBound,
    Adai,
    AdamP,
    Adan,
    AdaPNM,
    DiffGrad,
    DiffRGrad,
    Lamb,
    Lookahead,
    Nero,
    PCGrad,
    RAdam,
    RaLamb,
    Ranger,
    Ranger21,
    SafeFP16Optimizer,
    Shampoo,
)
from pytorch_optimizer.base.exception import NoClosureError, ZeroParameterSizeError
from tests.utils import (
    MultiHeadLogisticRegression,
    build_environment,
    build_lookahead,
    dummy_closure,
    ids,
    make_dataset,
    names,
    tensor_to_numpy,
)

OPTIMIZERS: List[Tuple[Any, Dict[str, Union[float, bool, int]], int]] = [
    (build_lookahead, {'lr': 5e-1, 'weight_decay': 1e-3}, 100),
    (AdaBelief, {'lr': 5e-1, 'weight_decay': 1e-3}, 100),
    (AdaBelief, {'lr': 5e-1, 'weight_decay': 1e-3, 'amsgrad': True}, 100),
    (AdaBelief, {'lr': 5e-1, 'weight_decay': 1e-3, 'weight_decouple': False}, 100),
    (AdaBelief, {'lr': 5e-1, 'weight_decay': 1e-3, 'fixed_decay': True}, 100),
    (AdaBelief, {'lr': 5e-1, 'weight_decay': 1e-3, 'rectify': False}, 100),
    (AdaBound, {'lr': 5e-1, 'gamma': 0.1, 'weight_decay': 1e-3}, 100),
    (AdaBound, {'lr': 5e-1, 'gamma': 0.1, 'weight_decay': 1e-3, 'fixed_decay': True}, 100),
    (AdaBound, {'lr': 5e-1, 'gamma': 0.1, 'weight_decay': 1e-3, 'weight_decouple': False}, 100),
    (AdaBound, {'lr': 5e-1, 'gamma': 0.1, 'weight_decay': 1e-3, 'amsbound': True}, 100),
    (Adai, {'lr': 1e-1, 'weight_decay': 0.0}, 200),
    (Adai, {'lr': 1e-1, 'weight_decay': 0.0, 'use_gc': True}, 200),
    (Adai, {'lr': 1e-1, 'weight_decay': 0.0, 'dampening': 0.9}, 200),
    (Adai, {'lr': 1e-1, 'weight_decay': 1e-4, 'weight_decouple': False}, 200),
    (Adai, {'lr': 1e-1, 'weight_decay': 1e-4, 'weight_decouple': True}, 200),
    (AdamP, {'lr': 5e-1, 'weight_decay': 1e-3}, 100),
    (AdamP, {'lr': 5e-1, 'weight_decay': 1e-3, 'use_gc': True}, 100),
    (AdamP, {'lr': 5e-1, 'weight_decay': 1e-3, 'nesterov': True}, 100),
    (DiffGrad, {'lr': 5e-1, 'weight_decay': 1e-3}, 100),
    (DiffRGrad, {'lr': 5e-1, 'weight_decay': 1e-3}, 100),
    (Lamb, {'lr': 1e-1, 'weight_decay': 1e-3}, 100),
    (Lamb, {'lr': 1e-1, 'weight_decay': 1e-3, 'adam': True, 'eps': 1e-8}, 100),
    (Lamb, {'lr': 1e-1, 'weight_decay': 1e-3, 'pre_norm': True, 'eps': 1e-8}, 500),
    (LARS, {'lr': 1e-1, 'weight_decay': 1e-3}, 300),
    (RaLamb, {'lr': 1e-1, 'weight_decay': 1e-4}, 100),
    (RaLamb, {'lr': 1e-2, 'weight_decay': 1e-4, 'pre_norm': True}, 100),
    (RaLamb, {'lr': 1e-2, 'weight_decay': 1e-4, 'degenerated_to_sgd': True}, 100),
    (MADGRAD, {'lr': 1e-2, 'weight_decay': 1e-3}, 100),
    (MADGRAD, {'lr': 1e-2, 'weight_decay': 1e-3, 'eps': 0.0}, 100),
    (MADGRAD, {'lr': 1e-2, 'weight_decay': 1e-3, 'momentum': 0.0}, 100),
    (MADGRAD, {'lr': 1e-2, 'weight_decay': 1e-3, 'decouple_decay': True}, 100),
    (RAdam, {'lr': 1e-1, 'weight_decay': 1e-3}, 200),
    (RAdam, {'lr': 1e-1, 'weight_decay': 1e-3, 'degenerated_to_sgd': True}, 200),
    (SGDP, {'lr': 5e-2, 'weight_decay': 1e-4}, 100),
    (SGDP, {'lr': 5e-2, 'weight_decay': 1e-4, 'nesterov': True}, 100),
    (Ranger, {'lr': 5e-1, 'weight_decay': 1e-3}, 200),
    (Ranger21, {'lr': 5e-1, 'weight_decay': 1e-3, 'num_iterations': 500}, 200),
    (Shampoo, {'lr': 5e-2, 'weight_decay': 1e-3, 'momentum': 0.01}, 300),
    (PNM, {'lr': 2e-1}, 100),
    (PNM, {'lr': 2e-1, 'weight_decouple': False}, 100),
    (AdaPNM, {'lr': 3e-1, 'weight_decay': 1e-3}, 100),
    (AdaPNM, {'lr': 3e-1, 'weight_decay': 1e-3, 'weight_decouple': False}, 100),
    (AdaPNM, {'lr': 3e-1, 'weight_decay': 1e-3, 'amsgrad': False}, 100),
    (Nero, {'lr': 5e-1}, 100),
    (Nero, {'lr': 5e-1, 'constraints': False}, 100),
    (Adan, {'lr': 5e-1}, 100),
    (Adan, {'lr': 1e-0, 'weight_decay': 1e-3, 'use_gc': True}, 100),
    (Adan, {'lr': 1e-0, 'weight_decay': 1e-3, 'use_gc': True, 'weight_decouple': True}, 100),
]
ADAMD_SUPPORTED_OPTIMIZERS: List[Tuple[Any, Dict[str, Union[float, bool, int]], int]] = [
    (build_lookahead, {'lr': 5e-1, 'weight_decay': 1e-3, 'adamd_debias_term': True}, 100),
    (AdaBelief, {'lr': 5e-1, 'weight_decay': 1e-3, 'adamd_debias_term': True}, 100),
    (AdaBound, {'lr': 5e-1, 'gamma': 0.1, 'weight_decay': 1e-3, 'adamd_debias_term': True}, 100),
    (AdaBound, {'lr': 1e-2, 'gamma': 0.1, 'weight_decay': 1e-3, 'amsbound': True, 'adamd_debias_term': True}, 100),
    (AdamP, {'lr': 5e-1, 'weight_decay': 1e-3, 'adamd_debias_term': True}, 100),
    (DiffGrad, {'lr': 15 - 1, 'weight_decay': 1e-3, 'adamd_debias_term': True}, 100),
    (DiffRGrad, {'lr': 1e-1, 'weight_decay': 1e-3, 'adamd_debias_term': True}, 100),
    (Lamb, {'lr': 1e-1, 'weight_decay': 1e-3, 'adamd_debias_term': True}, 100),
    (RaLamb, {'lr': 1e-1, 'weight_decay': 1e-3, 'adamd_debias_term': True}, 100),
    (RAdam, {'lr': 1e-1, 'weight_decay': 1e-3, 'adamd_debias_term': True}, 100),
    (Ranger, {'lr': 5e-1, 'weight_decay': 1e-3, 'adamd_debias_term': True}, 100),
    (Ranger21, {'lr': 5e-1, 'weight_decay': 1e-3, 'adamd_debias_term': True}, 200),
    (AdaPNM, {'lr': 3e-1, 'weight_decay': 1e-3, 'adamd_debias_term': True}, 100),
]


@pytest.mark.parametrize('optimizer_fp32_config', OPTIMIZERS, ids=ids)
def test_f32_optimizers(optimizer_fp32_config):
    (x_data, y_data), model, loss_fn = build_environment()

    optimizer_class, config, iterations = optimizer_fp32_config

    optimizer_name: str = optimizer_class.__name__
    if optimizer_name == 'Nero' and 'constraints' not in config:
        pytest.skip(f'skip {optimizer_name} w/o constraints')

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

    assert tensor_to_numpy(init_loss) > 1.5 * tensor_to_numpy(loss)


@pytest.mark.parametrize('pullback_momentum', ['none', 'reset', 'pullback'])
def test_lookahead(pullback_momentum):
    (x_data, y_data), model, loss_fn = build_environment()

    optimizer = Lookahead(AdamP(model.parameters(), lr=5e-1), pullback_momentum=pullback_momentum)

    init_loss, loss = np.inf, np.inf
    for _ in range(100):
        optimizer.zero_grad()

        y_pred = model(x_data)
        loss = loss_fn(y_pred, y_data)

        if init_loss == np.inf:
            init_loss = loss

        loss.backward()

        optimizer.step()

    assert tensor_to_numpy(init_loss) > 2.0 * tensor_to_numpy(loss)


@pytest.mark.parametrize('optimizer_fp16_config', OPTIMIZERS, ids=ids)
def test_safe_f16_optimizers(optimizer_fp16_config):
    (x_data, y_data), model, loss_fn = build_environment()

    optimizer_class, config, iterations = optimizer_fp16_config

    optimizer_name: str = optimizer_class.__name__
    if (
        (optimizer_name == 'MADGRAD')
        or (optimizer_name == 'RaLamb' and 'pre_norm' in config)
        or (optimizer_name == 'PNM')
        or (optimizer_name == 'Nero')
        or (optimizer_name == 'Adan' and 'weight_decay' not in config)
        or (optimizer_name == 'RAdam')
        or (optimizer_name == 'Adai')
    ):
        pytest.skip(f'skip {optimizer_name}')

    optimizer = SafeFP16Optimizer(optimizer_class(model.parameters(), **config))

    init_loss, loss = np.inf, np.inf
    for _ in range(iterations):
        optimizer.zero_grad()

        y_pred = model(x_data)
        loss = loss_fn(y_pred, y_data)

        if init_loss == np.inf:
            init_loss = loss

        loss.backward()

        optimizer.step()

    assert tensor_to_numpy(init_loss) > tensor_to_numpy(loss)


@pytest.mark.parametrize('adaptive', (False, True))
@pytest.mark.parametrize('optimizer_sam_config', OPTIMIZERS, ids=ids)
def test_sam_optimizers(adaptive, optimizer_sam_config):
    (x_data, y_data), model, loss_fn = build_environment()

    optimizer_class, config, iterations = optimizer_sam_config

    optimizer_name: str = optimizer_class.__name__
    if (optimizer_name == 'Shampoo') or (optimizer_name == 'Adai'):
        pytest.skip(f'skip {optimizer_name}')

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


@pytest.mark.parametrize('adaptive', (False, True))
@pytest.mark.parametrize('optimizer_sam_config', OPTIMIZERS, ids=ids)
def test_sam_optimizers_with_closure(adaptive, optimizer_sam_config):
    (x_data, y_data), model, loss_fn = build_environment()

    optimizer_class, config, iterations = optimizer_sam_config

    optimizer_name: str = optimizer_class.__name__
    if (optimizer_name == 'Shampoo') or (optimizer_name == 'Adai'):
        pytest.skip(f'skip {optimizer_name}')

    optimizer = SAM(model.parameters(), optimizer_class, **config, adaptive=adaptive)

    def closure():
        first_loss = loss_fn(y_data, model(x_data))
        first_loss.backward()
        return first_loss

    init_loss, loss = np.inf, np.inf
    for _ in range(iterations):
        loss = loss_fn(y_data, model(x_data))
        loss.backward()

        optimizer.step(closure)
        optimizer.zero_grad()

        if init_loss == np.inf:
            init_loss = loss

    assert tensor_to_numpy(init_loss) > 2.0 * tensor_to_numpy(loss)


@pytest.mark.parametrize('optimizer_adamd_config', ADAMD_SUPPORTED_OPTIMIZERS, ids=ids)
def test_adamd_optimizers(optimizer_adamd_config):
    (x_data, y_data), model, loss_fn = build_environment()

    optimizer_class, config, num_iterations = optimizer_adamd_config
    if optimizer_class.__name__ == 'Ranger21':
        config.update({'num_iterations': num_iterations})

    optimizer = optimizer_class(model.parameters(), **config)

    init_loss, loss = np.inf, np.inf
    for _ in range(num_iterations):
        optimizer.zero_grad()

        y_pred = model(x_data)
        loss = loss_fn(y_pred, y_data)

        if init_loss == np.inf:
            init_loss = loss

        loss.backward()

        optimizer.step()

    assert tensor_to_numpy(init_loss) > 2.0 * tensor_to_numpy(loss)


@pytest.mark.parametrize('reduction', ('mean', 'sum'))
@pytest.mark.parametrize('optimizer_pc_grad_config', OPTIMIZERS, ids=ids)
def test_pc_grad_optimizers(reduction, optimizer_pc_grad_config):
    torch.manual_seed(42)

    x_data, y_data = make_dataset()

    model: nn.Module = MultiHeadLogisticRegression()
    loss_fn_1: nn.Module = nn.BCEWithLogitsLoss()
    loss_fn_2: nn.Module = nn.L1Loss()

    optimizer_class, config, iterations = optimizer_pc_grad_config
    optimizer = PCGrad(optimizer_class(model.parameters(), **config), reduction=reduction)

    if optimizer_class.__name__ == 'RaLamb' and 'pre_norm' in config:
        pytest.skip(f'skip {optimizer_class.__name__} w/ pre_norm')

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

    assert tensor_to_numpy(init_loss) > 1.25 * tensor_to_numpy(loss)


@pytest.mark.parametrize('optimizer_config', OPTIMIZERS, ids=ids)
def test_no_gradients(optimizer_config):
    (x_data, y_data), model, loss_fn = build_environment()

    model.fc1.weight.requires_grad = False
    model.fc1.bias.requires_grad = False

    optimizer_class, config, iterations = optimizer_config
    optimizer = optimizer_class(model.parameters(), **config)

    if optimizer_class.__name__ == 'Nero':
        pytest.skip(f'skip {optimizer_class.__name__}')

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


@pytest.mark.parametrize('optimizer', set(config[0] for config in OPTIMIZERS), ids=names)
def test_closure(optimizer):
    _, model, _ = build_environment()

    if optimizer.__name__ == 'Ranger21':
        optimizer = optimizer(model.parameters(), num_iterations=1)
    else:
        optimizer = optimizer(model.parameters())

    optimizer.zero_grad()

    try:
        optimizer.step(closure=dummy_closure)
    except ZeroParameterSizeError:  # in case of Ranger21, Adai optimizers
        pass


def test_no_closure():
    _, model, _ = build_environment()

    optimizer = SAM(model.parameters(), AdamP)
    optimizer.zero_grad()

    with pytest.raises(NoClosureError):
        optimizer.step()


def test_sam_no_gradient():
    (x_data, y_data), model, loss_fn = build_environment()
    model.fc1.require_grads = False

    optimizer = SAM(model.parameters(), AdamP)
    optimizer.zero_grad()

    loss = loss_fn(y_data, model(x_data))
    loss.backward()
    optimizer.first_step(zero_grad=True)

    loss_fn(y_data, model(x_data)).backward()
    optimizer.second_step(zero_grad=True)


@pytest.mark.parametrize('optimizer_config', OPTIMIZERS, ids=ids)
def test_reset(optimizer_config):
    _, model, _ = build_environment()

    optimizer_class, config, _ = optimizer_config
    if optimizer_class.__name__ == 'Ranger21':
        config.update({'num_iterations': 1})

    optimizer = optimizer_class(model.parameters(), **config)
    optimizer.zero_grad()
    optimizer.reset()
