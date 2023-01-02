import numpy as np
import pytest
import torch
from torch import nn

from pytorch_optimizer import SAM, Lookahead, PCGrad, SafeFP16Optimizer, load_optimizer
from pytorch_optimizer.base.exception import NoClosureError, ZeroParameterSizeError
from tests.constants import ADAMD_SUPPORTED_OPTIMIZERS, ADAPTIVE_FLAGS, OPTIMIZERS, PULLBACK_MOMENTUM
from tests.utils import (
    MultiHeadLogisticRegression,
    build_environment,
    dummy_closure,
    ids,
    make_dataset,
    names,
    simple_parameter,
    tensor_to_numpy,
)


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


@pytest.mark.parametrize('pullback_momentum', PULLBACK_MOMENTUM)
def test_lookahead(pullback_momentum):
    (x_data, y_data), model, loss_fn = build_environment()

    optimizer = Lookahead(load_optimizer('adamp')(model.parameters(), lr=5e-1), pullback_momentum=pullback_momentum)

    init_loss, loss = np.inf, np.inf
    for _ in range(10):
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


@pytest.mark.parametrize('adaptive', ADAPTIVE_FLAGS)
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


@pytest.mark.parametrize('adaptive', ADAPTIVE_FLAGS)
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


@pytest.mark.parametrize('optimizer', set(config[0] for config in OPTIMIZERS), ids=names)
def test_closure(optimizer):
    param = simple_parameter()

    if optimizer.__name__ == 'Ranger21':
        optimizer = optimizer([param], num_iterations=1)
    else:
        optimizer = optimizer([param])

    optimizer.zero_grad()

    try:
        optimizer.step(closure=dummy_closure)
    except ZeroParameterSizeError:  # in case of Ranger21, Adai optimizers
        pass


def test_no_closure():
    param = simple_parameter()

    optimizer = SAM([param], load_optimizer('adamp'))
    optimizer.zero_grad()

    with pytest.raises(NoClosureError):
        optimizer.step()


def test_nero_zero_scale():
    param = simple_parameter()

    optimizer = load_optimizer('nero')([param], constraints=False)
    optimizer.zero_grad()
    param.grad = torch.zeros(1, 1)
    optimizer.step()


@pytest.mark.parametrize('optimizer_name', ['diffrgrad', 'adabelief', 'radam', 'ralamb'])
def test_rectified_optimizer(optimizer_name):
    param = simple_parameter()

    optimizer = load_optimizer(optimizer_name)([param], n_sma_threshold=1000, degenerated_to_sgd=False)
    optimizer.zero_grad()
    param.grad = torch.zeros(1, 1)
    optimizer.step()


@pytest.mark.parametrize('optimizer_config', OPTIMIZERS, ids=ids)
def test_reset(optimizer_config):
    param = simple_parameter()

    optimizer_class, config, _ = optimizer_config
    if optimizer_class.__name__ == 'Ranger21':
        config.update({'num_iterations': 1})

    optimizer = optimizer_class([param], **config)
    optimizer.zero_grad()
    optimizer.reset()
