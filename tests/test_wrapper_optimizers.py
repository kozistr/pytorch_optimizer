import numpy as np
import pytest
import torch
from torch import nn

from pytorch_optimizer import (
    BSAM,
    GSAM,
    SAM,
    TRAC,
    WSAM,
    CosineScheduler,
    Lookahead,
    LookSAM,
    OrthoGrad,
    PCGrad,
    ProportionScheduler,
    ScheduleFreeWrapper,
    load_optimizer,
)
from tests.constants import ADAPTIVE_FLAGS, DECOUPLE_FLAGS, PULLBACK_MOMENTUM
from tests.utils import Example, MultiHeadLogisticRegression, build_model, simple_parameter, tensor_to_numpy


@pytest.mark.parametrize('pullback_momentum', PULLBACK_MOMENTUM)
def test_lookahead(pullback_momentum, environment):
    x_data, y_data = environment
    model, loss_fn = build_model()

    optimizer = Lookahead(load_optimizer('adamw')(model.parameters(), lr=5e-1), pullback_momentum=pullback_momentum)
    optimizer.init_group({})

    init_loss, loss = np.inf, np.inf
    for _ in range(5):
        optimizer.zero_grad()

        y_pred = model(x_data)
        loss = loss_fn(y_pred, y_data)

        if init_loss == np.inf:
            init_loss = loss

        loss.backward()

        optimizer.step()

    assert tensor_to_numpy(init_loss) > 2.0 * tensor_to_numpy(loss)


@pytest.mark.parametrize('adaptive', ADAPTIVE_FLAGS)
def test_sam_optimizer(adaptive, environment):
    x_data, y_data = environment
    model, loss_fn = build_model()

    optimizer = SAM(model.parameters(), load_optimizer('asgd'), lr=5e-1, adaptive=adaptive, use_gc=True)

    init_loss, loss = np.inf, np.inf
    for _ in range(5):
        loss = loss_fn(y_data, model(x_data))
        loss.backward()
        optimizer.first_step(zero_grad=True)

        loss_fn(y_data, model(x_data)).backward()
        optimizer.second_step(zero_grad=True)

        if init_loss == np.inf:
            init_loss = loss

    assert tensor_to_numpy(init_loss) > 2.0 * tensor_to_numpy(loss)


@pytest.mark.parametrize('adaptive', ADAPTIVE_FLAGS)
def test_sam_optimizer_with_closure(adaptive, environment):
    x_data, y_data = environment
    model, loss_fn = build_model()

    optimizer = SAM(model.parameters(), load_optimizer('adamw'), lr=5e-1, adaptive=adaptive)

    def closure():
        first_loss = loss_fn(y_data, model(x_data))
        first_loss.backward()
        return first_loss

    init_loss, loss = np.inf, np.inf
    for _ in range(5):
        loss = loss_fn(y_data, model(x_data))
        loss.backward()

        optimizer.step(closure)
        optimizer.zero_grad()

        if init_loss == np.inf:
            init_loss = loss

    assert tensor_to_numpy(init_loss) > 2.0 * tensor_to_numpy(loss)


def test_looksam_optimizer(environment):
    x_data, y_data = environment
    model, loss_fn = build_model()

    optimizer = LookSAM(model.parameters(), load_optimizer('adamw'), k=2, lr=5e-1, use_gc=True)

    init_loss, loss = np.inf, np.inf
    for _ in range(5):
        loss = loss_fn(y_data, model(x_data))
        loss.backward()
        optimizer.first_step(zero_grad=True)

        loss_fn(y_data, model(x_data)).backward()
        optimizer.second_step(zero_grad=True)

        if init_loss == np.inf:
            init_loss = loss

    assert tensor_to_numpy(init_loss) > 2.0 * tensor_to_numpy(loss)


def test_looksam_optimizer_with_closure(environment):
    x_data, y_data = environment
    model, loss_fn = build_model()

    optimizer = LookSAM(model.parameters(), load_optimizer('adamw'), lr=5e-1)

    def closure():
        first_loss = loss_fn(y_data, model(x_data))
        first_loss.backward()
        return first_loss

    init_loss, loss = np.inf, np.inf
    for _ in range(5):
        loss = loss_fn(y_data, model(x_data))
        loss.backward()

        optimizer.step(closure)
        optimizer.zero_grad()

        if init_loss == np.inf:
            init_loss = loss

    assert tensor_to_numpy(init_loss) > 2.0 * tensor_to_numpy(loss)


@pytest.mark.parametrize('adaptive', ADAPTIVE_FLAGS)
@pytest.mark.parametrize('decouple', DECOUPLE_FLAGS)
def test_wsam_optimizer(adaptive, decouple, environment):
    x_data, y_data = environment
    model, loss_fn = build_model()

    optimizer = WSAM(
        model,
        model.parameters(),
        load_optimizer('adamp'),
        lr=5e-2,
        adaptive=adaptive,
        decouple=decouple,
        max_norm=100.0,
    )

    init_loss, loss = np.inf, np.inf
    for _ in range(10):
        loss = loss_fn(y_data, model(x_data))
        loss.backward()
        optimizer.first_step(zero_grad=True)

        loss_fn(y_data, model(x_data)).backward()
        optimizer.second_step(zero_grad=True)

        if init_loss == np.inf:
            init_loss = loss

    assert tensor_to_numpy(init_loss) > 1.5 * tensor_to_numpy(loss)


@pytest.mark.parametrize('adaptive', ADAPTIVE_FLAGS)
def test_wsam_optimizer_with_closure(adaptive, environment):
    x_data, y_data = environment
    model, loss_fn = build_model()

    optimizer = WSAM(model, model.parameters(), load_optimizer('adamp'), lr=5e-2, adaptive=adaptive, max_norm=100.0)

    def closure():
        output = model(x_data)
        loss = loss_fn(output, y_data)
        loss.backward()
        return loss

    init_loss, loss = np.inf, np.inf
    for _ in range(10):
        loss = optimizer.step(closure)
        optimizer.zero_grad()

        if init_loss == np.inf:
            init_loss = loss

    assert tensor_to_numpy(init_loss) > 1.5 * tensor_to_numpy(loss)


@pytest.mark.parametrize('adaptive', ADAPTIVE_FLAGS)
def test_gsam_optimizer(adaptive, environment):
    pytest.skip('skip GSAM optimizer')

    x_data, y_data = environment
    model, loss_fn = build_model()

    lr: float = 5e-1
    num_iterations: int = 25

    base_optimizer = load_optimizer('adamp')(model.parameters(), lr=lr)
    lr_scheduler = CosineScheduler(base_optimizer, t_max=num_iterations, max_lr=lr, min_lr=lr, init_lr=lr)
    rho_scheduler = ProportionScheduler(lr_scheduler, max_lr=lr, min_lr=lr)
    optimizer = GSAM(
        model.parameters(), base_optimizer=base_optimizer, model=model, rho_scheduler=rho_scheduler, adaptive=adaptive
    )

    init_loss, loss = np.inf, np.inf
    for _ in range(num_iterations):
        optimizer.set_closure(loss_fn, x_data, y_data)
        _, loss = optimizer.step()

        if init_loss == np.inf:
            init_loss = loss

        lr_scheduler.step()
        optimizer.update_rho_t()

    assert tensor_to_numpy(init_loss) > 1.2 * tensor_to_numpy(loss)


@pytest.mark.parametrize('adaptive', ADAPTIVE_FLAGS)
def test_bsam_optimizer(adaptive, environment):
    x_data, y_data = environment
    model, loss_fn = build_model()

    optimizer = BSAM(model.parameters(), lr=2e-3, num_data=len(x_data), rho=1e-5, adaptive=adaptive)

    def closure():
        first_loss = loss_fn(y_data, model(x_data))
        first_loss.backward()
        return first_loss

    init_loss, loss = np.inf, np.inf
    for _ in range(20):
        loss = loss_fn(y_data, model(x_data))
        loss.backward()

        optimizer.step(closure)
        optimizer.zero_grad()

        if init_loss == np.inf:
            init_loss = loss

    assert tensor_to_numpy(init_loss) > tensor_to_numpy(loss)


def test_schedulefree_wrapper():
    model = Example()

    optimizer = ScheduleFreeWrapper(load_optimizer('adamw')(model.parameters(), lr=1e-3, weight_decay=1e-3))
    optimizer.zero_grad()

    model.fc1.weight.grad = torch.randn((1, 1))
    model.norm1.weight.grad = torch.randn((1,))

    with pytest.raises(ValueError):
        optimizer.step()

    optimizer.eval()
    optimizer.train()

    _ = optimizer.__str__
    _ = optimizer.__getstate__()
    _ = optimizer.param_groups

    optimizer.step()

    backup_state = optimizer.state_dict()

    optimizer = ScheduleFreeWrapper(load_optimizer('adamw')(model.parameters(), lr=1e-3, weight_decay=1e-3))
    optimizer.zero_grad()
    optimizer.train()

    optimizer.load_state_dict(backup_state)

    optimizer.step()

    optimizer.eval()
    optimizer.train()
    optimizer.train()

    optimizer.add_param_group({'params': []})


@pytest.mark.parametrize('reduction', ['mean', 'sum'])
def test_pc_grad_optimizers(reduction, environment):
    torch.manual_seed(42)

    x_data, y_data = environment

    model: nn.Module = MultiHeadLogisticRegression()
    loss_fn_1: nn.Module = nn.BCEWithLogitsLoss()
    loss_fn_2: nn.Module = nn.L1Loss()

    optimizer = PCGrad(load_optimizer('adamp')(model.parameters(), lr=1e-1), reduction=reduction)
    optimizer.init_group()

    init_loss, loss = np.inf, np.inf
    for _ in range(5):
        optimizer.zero_grad()

        y_pred_1, y_pred_2 = model(x_data)
        loss1, loss2 = loss_fn_1(y_pred_1, y_data), loss_fn_2(y_pred_2, y_data)

        loss = (loss1 + loss2) / 2.0
        if init_loss == np.inf:
            init_loss = loss

        optimizer.pc_backward([loss1, loss2])
        optimizer.step()

    assert tensor_to_numpy(init_loss) > 1.25 * tensor_to_numpy(loss)


def test_trac_optimizer(environment):
    x_data, y_data = environment
    model, loss_fn = build_model()

    optimizer = TRAC(load_optimizer('adamw')(model.parameters(), lr=1e0))

    init_loss, loss = np.inf, np.inf
    for _ in range(3):
        loss = loss_fn(model(x_data), y_data)

        if init_loss == np.inf:
            init_loss = loss

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    assert tensor_to_numpy(init_loss) > 2.0 * tensor_to_numpy(loss)


def test_trac_optimizer_erf_imag():
    model = Example()

    optimizer = TRAC(load_optimizer('adamw')(model.parameters()))

    optimizer.init_group()
    optimizer.zero_grad()

    complex_tensor = torch.complex(torch.tensor(0.0), torch.tensor(1.0))
    optimizer.erf_imag(complex_tensor)

    assert str(optimizer).lower() == 'trac'


@pytest.mark.parametrize('wrapper_optimizer_instance', [Lookahead, OrthoGrad, TRAC])
def test_load_wrapper_optimizer(wrapper_optimizer_instance):
    params = [simple_parameter()]

    _ = wrapper_optimizer_instance(torch.optim.AdamW(params))

    optimizer = wrapper_optimizer_instance(torch.optim.AdamW, params=params)
    optimizer.init_group({})
    optimizer.zero_grad()

    with pytest.raises(ValueError):
        wrapper_optimizer_instance(torch.optim.AdamW)

    _ = optimizer.param_groups
    _ = optimizer.state

    state = optimizer.state_dict()
    optimizer.load_state_dict(state)
