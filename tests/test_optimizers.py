import numpy as np
import pytest
import torch
from torch import nn

from pytorch_optimizer import GSAM, SAM, CosineScheduler, Lookahead, PCGrad, ProportionScheduler, load_optimizer
from pytorch_optimizer.base.exception import NoClosureError, ZeroParameterSizeError
from pytorch_optimizer.optimizer.utils import l2_projection
from tests.constants import (
    ADAMD_SUPPORTED_OPTIMIZERS,
    ADANORM_SUPPORTED_OPTIMIZERS,
    ADAPTIVE_FLAGS,
    AMSBOUND_SUPPORTED_OPTIMIZERS,
    OPTIMIZERS,
    PULLBACK_MOMENTUM,
    RECTIFY_SUPPORTED_OPTIMIZERS,
)
from tests.utils import (
    MultiHeadLogisticRegression,
    build_environment,
    dummy_closure,
    ids,
    make_dataset,
    names,
    simple_parameter,
    simple_sparse_parameter,
    simple_zero_rank_parameter,
    tensor_to_numpy,
)


@pytest.fixture(scope='function')
def build_trainer():
    return build_environment()


@pytest.mark.parametrize('optimizer_fp32_config', OPTIMIZERS, ids=ids)
def test_f32_optimizers(optimizer_fp32_config, build_trainer):
    def closure(x):
        def _closure() -> float:
            return x

        return _closure

    (x_data, y_data), model, loss_fn = build_trainer

    optimizer_class, config, iterations = optimizer_fp32_config

    optimizer_name: str = optimizer_class.__name__
    if optimizer_name == 'Nero' and 'constraints' not in config:
        pytest.skip(f'skip {optimizer_name} w/o {config}')

    parameters = list(model.parameters())

    if optimizer_name == 'AliG':
        config.update({'projection_fn': lambda: l2_projection(parameters, max_norm=1)})

    optimizer = optimizer_class(parameters, **config)

    init_loss, loss = np.inf, np.inf
    for _ in range(iterations):
        optimizer.zero_grad()

        y_pred = model(x_data)
        loss = loss_fn(y_pred, y_data)

        if init_loss == np.inf:
            init_loss = loss

        loss.backward()

        if optimizer_name == 'AliG':
            optimizer.step(closure(loss))
        else:
            optimizer.step()

    assert tensor_to_numpy(init_loss) > 1.5 * tensor_to_numpy(loss)


@pytest.mark.parametrize(
    'optimizer_config',
    ADANORM_SUPPORTED_OPTIMIZERS
    + ADAMD_SUPPORTED_OPTIMIZERS
    + AMSBOUND_SUPPORTED_OPTIMIZERS
    + RECTIFY_SUPPORTED_OPTIMIZERS,
    ids=ids,
)
def test_optimizer_variants(optimizer_config, build_trainer):
    (x_data, y_data), model, loss_fn = build_trainer

    optimizer_class, config, num_iterations = optimizer_config
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

    assert tensor_to_numpy(init_loss) > 1.5 * tensor_to_numpy(loss)


@pytest.mark.parametrize('pullback_momentum', PULLBACK_MOMENTUM)
def test_lookahead(pullback_momentum, build_trainer):
    (x_data, y_data), model, loss_fn = build_trainer

    optimizer = Lookahead(load_optimizer('adamp')(model.parameters(), lr=5e-1), pullback_momentum=pullback_momentum)

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
def test_sam_optimizers(adaptive, build_trainer):
    (x_data, y_data), model, loss_fn = build_trainer

    optimizer = SAM(model.parameters(), load_optimizer('asgd'), lr=5e-1, adaptive=adaptive)

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
def test_sam_optimizers_with_closure(adaptive, build_trainer):
    (x_data, y_data), model, loss_fn = build_trainer

    optimizer = SAM(model.parameters(), load_optimizer('adamp'), lr=5e-1, adaptive=adaptive)

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
def test_gsam_optimizers(adaptive, build_trainer):
    pytest.skip('skip GSAM optimizer')

    (x_data, y_data), model, loss_fn = build_trainer

    x_data = x_data.cuda()
    y_data = y_data.cuda()
    model.cuda()

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


@pytest.mark.parametrize('optimizer_config', ADANORM_SUPPORTED_OPTIMIZERS, ids=ids)
def test_adanorm_condition(optimizer_config):
    param = simple_parameter(True)
    param.grad = torch.ones(1, 1)

    optimizer_class, config = optimizer_config[:2]

    optimizer = optimizer_class([param], adanorm=True)
    optimizer.step()

    param.grad = torch.zeros(1, 1)
    optimizer.step()


@pytest.mark.parametrize('reduction', ['mean', 'sum'])
def test_pc_grad_optimizers(reduction):
    x_data, y_data = make_dataset()

    model: nn.Module = MultiHeadLogisticRegression()
    loss_fn_1: nn.Module = nn.BCEWithLogitsLoss()
    loss_fn_2: nn.Module = nn.L1Loss()

    optimizer = PCGrad(load_optimizer('adamp')(model.parameters(), lr=1e-1), reduction=reduction)
    optimizer.reset()

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


@pytest.mark.parametrize('optimizer', {config[0] for config in OPTIMIZERS}, ids=names)
def test_closure(optimizer):
    param = simple_parameter()
    param.grad = None

    optimizer_name: str = optimizer.__name__

    optimizer = optimizer([param], num_iterations=1) if optimizer_name == 'Ranger21' else optimizer([param])
    optimizer.zero_grad()

    if optimizer_name in ('Ranger21', 'Adai', 'AdamS'):
        with pytest.raises(ZeroParameterSizeError):
            optimizer.step(closure=dummy_closure)
    elif optimizer_name in ('AliG',):
        with pytest.raises(NoClosureError):
            optimizer.step()
    else:
        optimizer.step(closure=dummy_closure)


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


@pytest.mark.parametrize('optimizer_name', ['adabelief', 'radam', 'lamb', 'diffgrad', 'ranger'])
def test_rectified_optimizer(optimizer_name):
    param = simple_parameter()

    parameters = {'n_sma_threshold': 1000, 'degenerated_to_sgd': False}
    if optimizer_name not in ('adabelief', 'radam', 'ranger'):
        parameters.update({'rectify': True})

    optimizer = load_optimizer(optimizer_name)([param], **parameters)
    optimizer.zero_grad()

    param.grad = torch.zeros(1, 1)
    optimizer.step()


@pytest.mark.parametrize(
    'optimizer_config', OPTIMIZERS + AMSBOUND_SUPPORTED_OPTIMIZERS + ADANORM_SUPPORTED_OPTIMIZERS, ids=ids
)
def test_reset(optimizer_config):
    optimizer_class, config, _ = optimizer_config
    if optimizer_class.__name__ == 'Ranger21':
        config.update({'num_iterations': 1})

    optimizer = optimizer_class([simple_parameter()], **config)
    optimizer.reset()


@pytest.mark.parametrize('require_gradient', [False, True])
@pytest.mark.parametrize('sparse_gradient', [False, True])
@pytest.mark.parametrize('optimizer_name', ['DAdaptAdaGrad', 'DAdaptAdam', 'DAdaptSGD', 'DAdaptAdan'])
def test_d_adapt_reset(require_gradient, sparse_gradient, optimizer_name):
    param = simple_sparse_parameter(require_gradient)[1] if sparse_gradient else simple_parameter(require_gradient)
    if not require_gradient:
        param.grad = None

    optimizer = load_optimizer(optimizer_name)([param])
    assert str(optimizer) == optimizer_name
    optimizer.reset()


@pytest.mark.parametrize('pre_conditioner_type', [0, 1, 2])
def test_scalable_shampoo_pre_conditioner_with_svd(pre_conditioner_type):
    (x_data, y_data), _, loss_fn = build_environment()

    model = nn.Sequential(
        nn.Linear(2, 4096),
        nn.Linear(4096, 512),
        nn.Linear(512, 1),
    )

    optimizer = load_optimizer('scalableshampoo')(
        model.parameters(),
        start_preconditioning_step=1,
        preconditioning_compute_steps=1,
        pre_conditioner_type=pre_conditioner_type,
        use_svd=True,
    )
    optimizer.zero_grad()

    loss_fn(model(x_data), y_data).backward()

    optimizer.step()


def test_sm3_make_sparse():
    _, weight_sparse = simple_sparse_parameter(True)

    optimizer = load_optimizer('sm3')([weight_sparse])

    values = torch.tensor(1.0)
    optimizer.make_sparse(weight_sparse.grad, values)


def test_sm3_rank0():
    optimizer = load_optimizer('sm3')([simple_zero_rank_parameter(True)])
    assert str(optimizer) == 'SM3'
    optimizer.step()
