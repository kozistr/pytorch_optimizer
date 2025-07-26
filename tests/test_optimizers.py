import os

import numpy as np
import pytest
import torch
from torch import nn

from pytorch_optimizer.base.exception import NoClosureError, ZeroParameterSizeError
from pytorch_optimizer.optimizer import DynamicLossScaler, load_optimizer
from pytorch_optimizer.optimizer.alig import l2_projection
from pytorch_optimizer.optimizer.grokfast import gradfilter_ema, gradfilter_ma
from pytorch_optimizer.optimizer.scion import build_lmo_norm
from tests.constants import COMPLEX_OPTIMIZERS, OPTIMIZERS
from tests.utils import (
    Example,
    LogisticRegression,
    build_model,
    dummy_closure,
    ids,
    names,
    simple_parameter,
    simple_sparse_parameter,
    simple_zero_rank_parameter,
    sphere_loss,
    tensor_to_numpy,
)


def build_optimizer_parameter(parameters, optimizer_name, config):
    if optimizer_name == 'AliG':
        config.update({'projection_fn': lambda: l2_projection(parameters, max_norm=1)})
    elif optimizer_name in ('Muon', 'AdaMuon'):
        adamw_params = [p for i, p in enumerate(parameters) if i >= 2]
        parameters = [p for i, p in enumerate(parameters) if i < 2]
        config.update({'adamw_params': adamw_params})
    elif optimizer_name == 'AdamWSN':
        sn_params = [p for p in parameters if p.ndim == 2]
        regular_params = [p for p in parameters if p.ndim != 2]
        parameters = [{'params': sn_params, 'sn': True}, {'params': regular_params, 'sn': False}]
    elif optimizer_name == 'AdamC':
        norm_params = [p for i, p in enumerate(parameters) if i == 1]
        regular_params = [p for i, p in enumerate(parameters) if i != 1]
        parameters = [{'params': norm_params, 'normalized': True}, {'params': regular_params}]

    return parameters, config


@pytest.mark.parametrize('optimizer_fp32_config', OPTIMIZERS, ids=ids)
def test_f32_optimizers(optimizer_fp32_config, environment):
    def closure(x):
        def _closure() -> float:
            return x

        return _closure

    optimizer_class, config, iterations = optimizer_fp32_config
    optimizer_name: str = optimizer_class.__name__
    if optimizer_name == 'Nero' and 'constraints' not in config:
        pytest.skip(f'skip {optimizer_name} w/o {config}')

    x_data, y_data = environment
    model, loss_fn = build_model()

    parameters, config = build_optimizer_parameter(list(model.parameters()), optimizer_name, config)

    optimizer = optimizer_class(parameters, **config)

    if optimizer_name.endswith('schedulefree'):
        optimizer.train()

    init_loss, loss = np.inf, np.inf
    for _ in range(iterations):
        optimizer.zero_grad()

        loss = loss_fn(model(x_data), y_data)

        if init_loss == np.inf:
            init_loss = loss

        loss.backward(create_graph=optimizer_name in ('AdaHessian', 'SophiaH'))

        optimizer.step(closure(loss) if optimizer_name == 'AliG' else None)

    assert tensor_to_numpy(init_loss) > 1.5 * tensor_to_numpy(loss)


@pytest.mark.parametrize('optimizer_bf16_config', OPTIMIZERS, ids=ids)
def test_bf16_optimizers(optimizer_bf16_config, environment):
    def closure(x):
        def _closure() -> float:
            return x

        return _closure

    optimizer_class, config, iterations = optimizer_bf16_config
    optimizer_name: str = optimizer_class.__name__
    if optimizer_name in ('Adai', 'Prodigy', 'Nero'):
        pytest.skip(f'skip {optimizer_name}')

    x_data, y_data = environment
    model, loss_fn = build_model()
    model = model.bfloat16()

    parameters, config = build_optimizer_parameter(list(model.parameters()), optimizer_name, config)

    optimizer = optimizer_class(parameters, **config)

    if optimizer_name.endswith('schedulefree'):
        optimizer.train()

    context = torch.autocast('cpu', dtype=torch.bfloat16)
    scaler = torch.GradScaler(device='cpu', enabled=False)

    init_loss, loss = np.inf, np.inf
    for _ in range(iterations):
        optimizer.zero_grad()

        with context:
            loss = loss_fn(model(x_data), y_data)

        if init_loss == np.inf:
            init_loss = loss

        scaler.scale(loss).backward(create_graph=optimizer_name in ('AdaHessian', 'SophiaH'))

        optimizer.step(closure(loss) if optimizer_name == 'AliG' else None)

    assert tensor_to_numpy(init_loss) > 1.5 * tensor_to_numpy(loss)


@pytest.mark.parametrize('optimizer_complex_config', OPTIMIZERS, ids=ids)
def test_complex_optimizers(optimizer_complex_config, environment):
    def closure(x):
        def _closure() -> float:
            return x

        return _closure

    optimizer_class, config, iterations = optimizer_complex_config
    optimizer_name: str = optimizer_class.__name__.lower()

    if optimizer_name not in COMPLEX_OPTIMIZERS:
        pytest.skip(f'{optimizer_name} does not support')

    x_data, y_data = environment
    model, loss_fn = build_model(use_complex=True)

    x_data = x_data.to(torch.complex64)

    parameters, config = build_optimizer_parameter(list(model.parameters()), optimizer_name, config)

    optimizer = optimizer_class(parameters, **config)

    if optimizer_name.endswith('schedulefree'):
        optimizer.train()

    init_loss, loss = np.inf, np.inf
    for _ in range(iterations):
        optimizer.zero_grad()

        loss = loss_fn(model(x_data), y_data)

        if init_loss == np.inf:
            init_loss = loss

        loss.backward(create_graph=optimizer_name in ('adahessian', 'sophiah'))

        optimizer.step(closure(loss) if optimizer_name == 'alig' else None)

    assert tensor_to_numpy(init_loss) > 1.5 * tensor_to_numpy(loss)


@pytest.mark.parametrize('optimizer_config', OPTIMIZERS, ids=ids)
def test_init_group(optimizer_config):
    optimizer_class, *_ = optimizer_config

    optimizer_name: str = optimizer_class.__name__.lower()
    if optimizer_name.startswith('build'):
        pytest.skip(f'skip {optimizer_name}')

    optimizer_class([simple_parameter()], num_iterations=1).init_group({'params': [], 'betas': (0.0, 0.0)})


@pytest.mark.parametrize('optimizer', {config[0] for config in OPTIMIZERS}, ids=names)
def test_closure(optimizer):
    param = simple_parameter()
    param.grad = None

    optimizer_name: str = optimizer.__name__

    optimizer = optimizer([param], num_iterations=1) if optimizer_name == 'Ranger21' else optimizer([param])
    optimizer.zero_grad()

    if optimizer_name.endswith('schedulefree'):
        optimizer.train()

    if optimizer_name in ('Ranger21', 'Adai', 'AdamS'):
        with pytest.raises(ZeroParameterSizeError):
            optimizer.step(closure=dummy_closure)
    elif optimizer_name in ('AliG',):
        with pytest.raises(NoClosureError):
            optimizer.step()
    else:
        optimizer.step(closure=dummy_closure)


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

    sphere_loss(param).backward(create_graph=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    sphere_loss(param).backward()
    optimizer.step(hessian=torch.zeros_like(param).unsqueeze(0))


def test_swats_sgd_phase():
    model, _ = build_model()

    opt = load_optimizer('swats')(model.parameters(), lr=1e-1, nesterov=True, eps=1.0)

    model.fc1.weight.grad = None
    model.fc2.weight.grad = torch.zeros(1, 2)
    opt.step()

    model.fc1.weight.grad = None
    model.fc2.weight.grad = torch.ones(1, 2)
    opt.step()

    opt.param_groups[0]['phase'] = 'sgd'
    opt.step()


@pytest.mark.parametrize('pre_conditioner_type', [0, 1, 2])
def test_scalable_shampoo_pre_conditioner_with_svd(pre_conditioner_type):
    model, _ = build_model()

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

    model[0].weight.grad = torch.zeros(4096, 2)
    model[1].weight.grad = torch.zeros(512, 4096)
    model[2].weight.grad = torch.zeros(1, 512)

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


@pytest.mark.parametrize('optimizer_name', ['lomo', 'adalomo'])
def test_lomo_deepspeed_zero3(optimizer_name):
    model = LogisticRegression()

    model.fc1.weight.__setattr__('ds_tensor', 0)

    optimizer = load_optimizer(optimizer_name)(model)
    optimizer.init_group({})

    assert str(optimizer).lower() == optimizer_name


def test_lomo_clip_grad_norm_with_fp16():
    model = LogisticRegression()

    model.fc1.weight.data = torch.randn(2, 2, dtype=torch.float16)

    with pytest.raises(ValueError):
        load_optimizer('lomo')(model, clip_grad_norm=None)


@pytest.mark.parametrize('optimizer_name', ['lomo'])
def test_lomo_fused_backward(optimizer_name):
    optimizer = load_optimizer(optimizer_name)(LogisticRegression(), clip_grad_norm=1.0)
    with pytest.raises(ValueError):
        optimizer.fused_backward(loss=0.1, lr=0.1)


@pytest.mark.parametrize('optimizer_name', ['lomo', 'adalomo'])
@pytest.mark.parametrize('precision', [16, 32])
def test_lomo_optimizer(optimizer_name, precision):
    model = Example()

    model.fc1.bias.data = torch.randn(1, dtype=torch.float32)
    model.fc1.bias.grad = torch.randn(1, dtype=torch.float32)

    if precision == 16:
        model.fc1.weight.data = torch.randn(1, 1, dtype=torch.float16)
        model.fc1.weight.grad = torch.zeros(1, 1, dtype=torch.float16)

    optimizer = load_optimizer(optimizer_name)(model, clip_grad_norm=1.0, clip_grad_value=1.0)

    if precision == 16:
        optimizer.clip_coef = 0.9

    parameters = iter(model.parameters())

    loss = sphere_loss(next(parameters))
    optimizer.grad_norm(loss)
    optimizer.fused_backward(loss, lr=0.1)

    loss = sphere_loss(next(parameters))
    optimizer.grad_norm(loss)
    optimizer.fused_backward(loss, lr=0.1)


def test_dynamic_scaler():
    scaler = DynamicLossScaler(init_scale=2.0 ** 15, scale_window=1, threshold=1e-2)  # fmt: skip
    scaler.decrease_loss_scale()
    scaler.update_scale(overflow=False)


@pytest.mark.parametrize('optimizer_name', ['ScheduleFreeAdamW', 'ScheduleFreeSGD', 'ScheduleFreeRAdam'])
def test_schedule_free_methods(optimizer_name):
    optimizer = load_optimizer(optimizer_name)([simple_parameter(True)])
    optimizer.step()

    optimizer.eval()
    optimizer.train()


@pytest.mark.parametrize('filter_type', ['mean', 'sum'])
def test_grokfast_ma(filter_type):
    model = LogisticRegression()

    model.fc1.weight.grad = torch.randn(2, 2)
    model.fc1.bias.grad = torch.randn(2)
    model.fc2.weight.grad = torch.randn(1, 2)
    model.fc2.bias.grad = torch.randn(1)

    gradfilter_ma(model, None, window_size=1, filter_type=filter_type, warmup=False)


def test_grokfast_ma_invalid():
    with pytest.raises(ValueError):
        gradfilter_ma(LogisticRegression(), None, window_size=1, filter_type='asdf', warmup=False)


def test_grokfast_ema():
    model = LogisticRegression()

    model.fc1.weight.grad = torch.randn(2, 2)
    model.fc1.bias.grad = torch.randn(2)
    model.fc2.weight.grad = torch.randn(1, 2)
    model.fc2.bias.grad = torch.randn(1)

    gradfilter_ema(model, None)


def test_stableadamw_optimizer():
    model = LogisticRegression()

    model.fc1.weight.data = torch.randn(2, 2, dtype=torch.float16)

    optimizer = load_optimizer('StableAdamW')(model.parameters())
    optimizer.step()


def test_adam_mini_optimizer():
    optimizer = load_optimizer('AdamMini')(LogisticRegression())
    optimizer.step()


@pytest.mark.parametrize(
    'params',
    [
        {'merge_dims': True, 'precondition_1d': True, 'max_precondition_dim': 4, 'precondition_frequency': 1},
        {
            'merge_dims': True,
            'precondition_1d': False,
            'max_precondition_dim': 1,
            'precondition_frequency': 1,
            'normalize_gradient': True,
        },
    ],
)
def test_soap_parameters(params):
    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.Linear(8, 1),
    )

    optimizer = load_optimizer('soap')(model.parameters(), **params)

    for _ in range(2):
        optimizer.zero_grad()

        model[0].weight.grad = torch.zeros((8, 2))
        model[0].bias.grad = torch.zeros((8,))
        model[1].weight.grad = torch.zeros((1, 8))
        model[1].bias.grad = None

        optimizer.step()


def test_soap_merge_dims_channel_last(environment):
    x_data, y_data = environment

    x_data = x_data.reshape(-1, 1, 2, 1).repeat_interleave(2, dim=-1).to(memory_format=torch.channels_last)

    model = nn.Sequential(
        nn.Conv2d(1, 1, 2, 1),
    )

    optimizer = load_optimizer('soap')(
        model.parameters(),
        merge_dims=True,
        precondition_1d=True,
        max_precondition_dim=2,
        precondition_frequency=1,
        data_format='channels_last',
    )

    for _ in range(2):
        optimizer.zero_grad()
        nn.BCEWithLogitsLoss()(model(x_data).squeeze(), y_data.squeeze()).backward()
        optimizer.step()


@pytest.mark.parametrize('optimizer_name', ['Muon', 'AdaMuon'])
@pytest.mark.parametrize('rank', ['1', '0'])
def test_muon_rank(optimizer_name, rank):
    os.environ['RANK'] = rank

    model = nn.Sequential(
        nn.Conv1d(1, 1, 1),
        nn.Conv1d(1, 1, 1),
        nn.Conv2d(1, 1, (2, 2)),
    )

    optimizer = load_optimizer(optimizer_name)(model.parameters())
    optimizer.zero_grad()

    model[0].weight.grad = torch.randn(1, 1, 1)
    model[1].weight.grad = torch.randn(1, 1, 1)
    model[2].weight.grad = torch.randn(1, 1, 2, 2)

    optimizer.step()


def test_mars_c_t_norm():
    param = simple_parameter(True)
    param.grad[0] = 100.0

    optimizer = load_optimizer('mars')([param], optimize_1d=True)
    optimizer.step()


def test_spam_optimizer():
    optimizer = load_optimizer('spam')(Example().parameters(), density=0.0)
    optimizer.step()

    optimizer = load_optimizer('spam')([simple_parameter(True)], grad_accu_steps=0, update_proj_gap=1)
    optimizer.step()


def test_kron_optimizer():
    model = Example()

    optimizer = load_optimizer('kron')(
        model.parameters(),
        weight_decay=1e-3,
        pre_conditioner_update_probability=1.0,
        balance_prob=1.0,
        mu_dtype=torch.bfloat16,
    )
    optimizer.zero_grad()

    model.fc1.weight.grad = torch.randn((1, 1))
    model.norm1.weight.grad = torch.randn((1,))

    optimizer.step()


@pytest.mark.parametrize('lmo_type', list(range(9)))
def test_build_lmo_types(lmo_type):
    build_lmo_norm(lmo_type)


def test_scion_lmo_types():
    model = Example()

    load_optimizer('scion')(model.parameters()).init()
    load_optimizer('scionlight')(model.parameters()).init()

    grad_1d = torch.ones(1)
    grad_2d = torch.ones(1, 1)
    grad_4d = torch.ones(1, 1, 1, 1)
    grad_5d = torch.ones(1, 1, 1, 1, 1)

    norm = build_lmo_norm(norm_type=0)
    norm.init(grad_2d)
    norm.lmo(grad_2d)

    norm = build_lmo_norm(norm_type=1, max_scale=True)
    for grad in (grad_1d, grad_2d, grad_4d):
        norm.init(grad)
        norm.lmo(grad)

    with pytest.raises(NotImplementedError):
        norm.init(grad_5d)

    with pytest.raises(NotImplementedError):
        norm.lmo(grad_5d)

    norm = build_lmo_norm(norm_type=2, max_scale=True)
    norm.init(grad_2d)
    norm.lmo(grad_2d)

    norm = build_lmo_norm(norm_type=4, zero_init=True)
    norm.init(grad_2d)
    norm.lmo(grad_2d)

    norm = build_lmo_norm(norm_type=4, zero_init=False)
    norm.init(grad_2d)
    norm.lmo(grad_2d)

    norm = build_lmo_norm(norm_type=6, normalized=True, transpose=True)
    norm.init(grad_2d)
    norm.lmo(grad_2d)

    norm = build_lmo_norm(norm_type=7, normalized=True, transpose=True)
    norm.init(grad_2d)
    norm.lmo(grad_2d)


@pytest.mark.parametrize('optimizer_name', ['racs', 'alice'])
def test_non_linear_parameters(optimizer_name):
    model = nn.Sequential(
        nn.Conv1d(8, 4, 1),
        nn.Conv2d(8, 4, (2, 2)),
    )

    optimizer = load_optimizer(optimizer_name)(model.parameters(), rank=4, leading_basis=2)
    optimizer.zero_grad()

    model[0].weight.grad = torch.randn(4, 8, 1)
    model[1].weight.grad = torch.randn(4, 8, 2, 2)

    optimizer.step()


def test_splus_methods():
    optimizer = load_optimizer('splus')([simple_parameter(True)])
    optimizer.step()

    optimizer.eval()
    optimizer.train()
