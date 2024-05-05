import numpy as np
import pytest
import torch
from torch import nn

from pytorch_optimizer import (
    BSAM,
    GSAM,
    SAM,
    WSAM,
    CosineScheduler,
    DynamicLossScaler,
    Lookahead,
    PCGrad,
    ProportionScheduler,
    load_optimizer,
)
from pytorch_optimizer.base.exception import NoClosureError, ZeroParameterSizeError
from pytorch_optimizer.optimizer.utils import l2_projection
from tests.constants import (
    ADAMD_SUPPORTED_OPTIMIZERS,
    ADANORM_SUPPORTED_OPTIMIZERS,
    ADAPTIVE_FLAGS,
    DECOUPLE_FLAGS,
    OPTIMIZERS,
    PULLBACK_MOMENTUM,
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
    sphere_loss,
    tensor_to_numpy,
)


@pytest.fixture(scope='function')
def environment():
    return build_environment()


@pytest.mark.parametrize('optimizer_fp32_config', OPTIMIZERS, ids=ids)
def test_f32_optimizers(optimizer_fp32_config, environment):
    def closure(x):
        def _closure() -> float:
            return x

        return _closure

    (x_data, y_data), model, loss_fn = environment

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

        loss.backward(create_graph=optimizer_name in ('AdaHessian', 'SophiaH'))

        optimizer.step(closure(loss) if optimizer_name == 'AliG' else None)

    assert tensor_to_numpy(init_loss) > 1.5 * tensor_to_numpy(loss)


@pytest.mark.parametrize('pullback_momentum', PULLBACK_MOMENTUM)
def test_lookahead(pullback_momentum, environment):
    (x_data, y_data), model, loss_fn = environment

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
def test_sam_optimizer(adaptive, environment):
    (x_data, y_data), model, loss_fn = environment

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
def test_sam_optimizer_with_closure(adaptive, environment):
    (x_data, y_data), model, loss_fn = environment

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
@pytest.mark.parametrize('decouple', DECOUPLE_FLAGS)
def test_wsam_optimizer(adaptive, decouple, environment):
    (x_data, y_data), model, loss_fn = environment

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
    (x_data, y_data), model, loss_fn = environment

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

    (x_data, y_data), model, loss_fn = environment

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
    (x_data, y_data), model, loss_fn = environment

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


@pytest.mark.parametrize('optimizer_config', ADANORM_SUPPORTED_OPTIMIZERS, ids=ids)
def test_adanorm_optimizer(optimizer_config, environment):
    (x_data, y_data), model, loss_fn = environment

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

    assert tensor_to_numpy(init_loss) > 1.75 * tensor_to_numpy(loss)


@pytest.mark.parametrize('optimizer_config', ADANORM_SUPPORTED_OPTIMIZERS, ids=ids)
def test_adanorm_condition(optimizer_config):
    param = simple_parameter(True)
    param.grad = torch.ones(1, 1)

    optimizer_class, config = optimizer_config[:2]

    optimizer = optimizer_class([param], adanorm=True)
    optimizer.step()

    param.grad = torch.zeros(1, 1)
    optimizer.step()


@pytest.mark.parametrize('optimizer_config', ADAMD_SUPPORTED_OPTIMIZERS, ids=ids)
def test_adamd_optimizers(optimizer_config, environment):
    (x_data, y_data), model, loss_fn = environment

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

        loss.backward(create_graph=optimizer_class.__name__ in ('AdaHessian',))

        optimizer.step()

    assert tensor_to_numpy(init_loss) > 2.0 * tensor_to_numpy(loss)


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

    optimizer = WSAM(None, [param], load_optimizer('adamp'))
    optimizer.zero_grad()

    with pytest.raises(NoClosureError):
        optimizer.step()

    optimizer = BSAM([param], 1)
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


@pytest.mark.parametrize('optimizer_name', ['sophiah', 'adahessian'])
def test_hessian_optimizer(optimizer_name):
    param = simple_parameter()

    parameters = {'hessian_distribution': 'gaussian', 'num_samples': 2}

    optimizer = load_optimizer(optimizer_name)([param], **parameters)
    optimizer.zero_grad(set_to_none=True)

    # Hutchinson (internal) estimator
    sphere_loss(param).backward(create_graph=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # External estimator
    sphere_loss(param).backward()
    optimizer.step(hessian=torch.zeros_like(param).unsqueeze(0))


def test_swats_sgd_phase(environment):
    (x_data, y_data), model, loss_fn = environment

    opt = load_optimizer('swats')(model.parameters(), lr=1e-1, nesterov=True, eps=1.0)

    opt.param_groups[0]['step'] = 1  # to bypass to adam -> sgd phase

    for _ in range(1):
        loss_fn(model(x_data), y_data).backward()
        opt.step()

    opt.param_groups[0]['phase'] = 'sgd'

    for _ in range(1):
        loss_fn(model(x_data), y_data).backward()
        opt.step()


@pytest.mark.parametrize('optimizer_config', OPTIMIZERS + ADANORM_SUPPORTED_OPTIMIZERS, ids=ids)
def test_reset(optimizer_config):
    optimizer_class, config, _ = optimizer_config
    if optimizer_class.__name__ == 'Ranger21':
        config.update({'num_iterations': 1})
    elif optimizer_class.__name__ == 'BSAM':
        config.update({'num_data': 1})

    optimizer = optimizer_class([simple_parameter()], **config)
    optimizer.reset()


@pytest.mark.parametrize('require_gradient', [False, True])
@pytest.mark.parametrize('sparse_gradient', [False, True])
@pytest.mark.parametrize('optimizer_name', ['DAdaptAdaGrad', 'DAdaptAdam', 'DAdaptSGD', 'DAdaptAdan', 'DAdaptLion'])
def test_d_adapt_reset(require_gradient, sparse_gradient, optimizer_name):
    param = simple_sparse_parameter(require_gradient)[1] if sparse_gradient else simple_parameter(require_gradient)
    if not require_gradient:
        param.grad = None

    optimizer = load_optimizer(optimizer_name)([param])
    optimizer.reset()

    assert str(optimizer) == optimizer_name


def test_prodigy_reset():
    param = simple_parameter(True)
    param.grad = None

    optimizer = load_optimizer('prodigy')([param])
    optimizer.reset()

    assert str(optimizer) == 'Prodigy'


def test_adalite_reset():
    optimizer = load_optimizer('adalite')([simple_zero_rank_parameter(True)])
    optimizer.reset()


@pytest.mark.parametrize('pre_conditioner_type', [0, 1, 2])
def test_scalable_shampoo_pre_conditioner_with_svd(pre_conditioner_type, environment):
    (x_data, y_data), _, loss_fn = environment

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
    optimizer.step()

    assert str(optimizer) == 'SM3'


def test_lomo_deepspeed_zero3(environment):
    _, model, _ = environment

    model.fc1.weight.__setattr__('ds_tensor', 0)

    optimizer = load_optimizer('lomo')(model)
    optimizer.reset()

    assert str(optimizer) == 'LOMO'


def test_lomo_clip_grad_norm_with_fp16(environment):
    _, model, _ = environment

    # clip grad norm with fp16
    model.fc1.weight.data = torch.randn(2, 2, dtype=torch.float16)

    with pytest.raises(ValueError):
        load_optimizer('lomo')(model, clip_grad_norm=None)


def test_lomo_fused_backward(environment):
    _, model, _ = environment

    optimizer = load_optimizer('lomo')(model, clip_grad_norm=1.0)
    with pytest.raises(ValueError):
        optimizer.fused_backward(loss=0.1, lr=0.1)


@pytest.mark.parametrize('precision', [16, 32])
def test_lomo_optimizer(precision, environment):
    _, model, _ = environment

    if precision == 16:
        model.fc1.weight.data = torch.randn(2, 2, dtype=torch.float16)
        model.fc1.weight.grad = torch.zeros(2, 2, dtype=torch.float16)

    optimizer = load_optimizer('lomo')(model, clip_grad_norm=1.0, clip_grad_value=1.0)

    if precision == 16:
        optimizer.clip_coef = 0.9

    loss = sphere_loss(next(iter(model.parameters())))
    optimizer.grad_norm(loss)
    optimizer.fused_backward(loss, lr=0.1)


def test_dynamic_scaler():
    scaler = DynamicLossScaler(init_scale=2.0**15, scale_window=1, threshold=1e-2)
    scaler.decrease_loss_scale()
    scaler.update_scale(overflow=False)
