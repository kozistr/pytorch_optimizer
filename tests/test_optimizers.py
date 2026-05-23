import pytest
import torch
from torch import nn

from pytorch_optimizer.base.exception import NoClosureError, ZeroParameterSizeError
from pytorch_optimizer.optimizer import DynamicLossScaler, load_optimizer
from pytorch_optimizer.optimizer.flash_adamw import compute_ecc_bits, reconstruct_fp32_param
from pytorch_optimizer.optimizer.grokfast import gradfilter_ema, gradfilter_ma
from pytorch_optimizer.optimizer.lora_rite import LoRARiteHelper
from pytorch_optimizer.optimizer.scion import build_lmo_norm
from pytorch_optimizer.optimizer.sso import SpectralSphere, solve_lambda_with_bisection
from tests.constants import (
    COMPLEX_OPTIMIZERS,
    FOREACH_OPTIMIZERS,
    OPTIMIZERS,
    SKIP_BF16_OPTIMIZERS,
)
from tests.utils import (
    Example,
    LogisticRegression,
    Trainer,
    build_model,
    build_optimizer_parameter,
    dummy_closure,
    ids,
    make_closure,
    names,
    should_use_create_graph,
    simple_parameter,
    simple_sparse_parameter,
    simple_zero_rank_parameter,
    sphere_loss,
)


@pytest.mark.parametrize('optimizer_config', OPTIMIZERS, ids=ids)
@pytest.mark.parametrize('foreach', [False, True])
def test_f32_optimizers(optimizer_config, foreach, environment):
    optimizer_class, config, iterations = optimizer_config
    optimizer_name: str = optimizer_class.__name__

    if optimizer_name == 'Nero' and 'constraints' not in config:
        pytest.skip(f'skip {optimizer_name} w/o {config}')

    if foreach and optimizer_name.lower() not in FOREACH_OPTIMIZERS:
        pytest.skip(f'skip {optimizer_name} w/ foreach')

    x_data, y_data = environment
    model, loss_fn = build_model()
    parameters, config = build_optimizer_parameter(list(model.parameters()), optimizer_name, config)

    optimizer = optimizer_class(parameters, **config, foreach=foreach)
    if optimizer_name.endswith('schedulefree'):
        optimizer.train()

    def closure_fn(x):
        return make_closure(x) if optimizer_name in ('AliG',) or optimizer_name.startswith('Emo') else None

    trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
    trainer.run(
        iterations=iterations,
        create_graph=should_use_create_graph(optimizer_name),
        closure_fn=closure_fn,
        threshold=1.5 if optimizer_name not in ('SpectralSphere',) else 1.4,
    )


@pytest.mark.parametrize('optimizer_config', OPTIMIZERS, ids=ids)
@pytest.mark.parametrize('foreach', [False, True])
def test_bf16_optimizers(optimizer_config, foreach, environment):
    optimizer_class, config, iterations = optimizer_config
    optimizer_name: str = optimizer_class.__name__

    if optimizer_name.lower() in SKIP_BF16_OPTIMIZERS:
        pytest.skip(f'skip {optimizer_name}')

    x_data, y_data = environment
    model, loss_fn = build_model()
    model = model.bfloat16()
    parameters, config = build_optimizer_parameter(list(model.parameters()), optimizer_name, config)

    optimizer = optimizer_class(parameters, **config, foreach=foreach)
    if optimizer_name.endswith('schedulefree'):
        optimizer.train()

    def closure_fn(x):
        return make_closure(x) if optimizer_name in ('AliG',) or optimizer_name.startswith('Emo') else None

    trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
    trainer.run_bf16(
        iterations=iterations,
        create_graph=should_use_create_graph(optimizer_name),
        closure_fn=closure_fn,
        threshold=1.5 if optimizer_name not in ('SpectralSphere',) else 1.4,
    )


@pytest.mark.parametrize('optimizer_config', OPTIMIZERS, ids=ids)
def test_complex_optimizers(optimizer_config, environment):
    optimizer_class, config, iterations = optimizer_config
    optimizer_name: str = optimizer_class.__name__

    if optimizer_name.lower() not in COMPLEX_OPTIMIZERS:
        pytest.skip(f'{optimizer_name} does not support complex')

    x_data, y_data = environment
    model, loss_fn = build_model(use_complex=True)
    x_data = x_data.to(torch.complex64)
    parameters, config = build_optimizer_parameter(list(model.parameters()), optimizer_name, config)

    optimizer = optimizer_class(parameters, **config, foreach=False)
    optimizer_name_lower = optimizer_name.lower()
    if optimizer_name_lower.endswith('schedulefree'):
        optimizer.train()

    def closure_fn(x):
        return make_closure(x) if optimizer_name_lower in ('alig',) or optimizer_name_lower.startswith('emo') else None

    trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
    trainer.run(
        iterations=iterations,
        create_graph=should_use_create_graph(optimizer_name_lower),
        closure_fn=closure_fn,
        threshold=1.5,
    )


@pytest.mark.parametrize('optimizer_config', OPTIMIZERS, ids=ids)
def test_init_group(optimizer_config):
    optimizer_class, *_ = optimizer_config
    optimizer_name: str = optimizer_class.__name__.lower()
    if optimizer_name.startswith('build'):
        pytest.skip(f'skip {optimizer_name}')

    param = simple_parameter()
    common_config = {'num_iterations': 1}
    group = {'params': []}

    if optimizer_name in {'muon', 'adamuon', 'adago'}:
        optimizer_class([{'params': param, 'use_muon': True}], **common_config).init_group(group)
    else:
        optimizer_class([param], **common_config).init_group({**group, 'betas': (0.0, 0.0)})


@pytest.mark.parametrize('optimizer', {config[0] for config in OPTIMIZERS}, ids=names)
def test_closure(optimizer):
    param = simple_parameter()
    param.grad = None

    optimizer_name: str = optimizer.__name__

    if optimizer_name == 'Ranger21':
        optimizer = optimizer([param], num_iterations=1)
    elif optimizer_name in ('Muon', 'AdaMuon', 'AdaGO'):
        optimizer = optimizer([{'params': param, 'use_muon': False}])
    else:
        optimizer = optimizer([param])

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
    with pytest.raises(NotImplementedError):
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


@pytest.mark.parametrize('optimizer_name', ['Muon', 'AdaMuon', 'AdaGO'])
def test_muon_high_dimensions(optimizer_name):
    model = nn.Sequential(
        nn.Conv1d(1, 1, 1),
        nn.Conv2d(1, 1, (2, 2)),
        nn.LSTM(1, 1, num_layers=1, bias=True, bidirectional=True),
    )

    params = [
        {'params': [p for p in model.parameters() if p.ndim >= 2], 'use_muon': True},
        {'params': [p for p in model.parameters() if p.ndim < 2], 'use_muon': False},
    ]

    optimizer = load_optimizer(optimizer_name)(params)
    optimizer.zero_grad()

    model[0].weight.grad = torch.randn(1, 1, 1)
    model[1].weight.grad = torch.randn(1, 1, 2, 2)
    model[2].weight_ih_l0.grad = model[2].weight_hh_l0.grad = torch.randn(4, 1)

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


@pytest.mark.parametrize('optimizer_name', ['emonavi', 'emolynx', 'emofact'])
def test_emo_optimizers(optimizer_name):
    optimizer = load_optimizer(optimizer_name)([simple_parameter(True)], use_shadow=True)

    optimizer.init_group(optimizer.param_groups[0])
    optimizer.state['ema'] = {'short': 1.0, 'medium': 5.0, 'long': 40.0}

    optimizer.step()


def test_spectral_sphere_methods():
    opt = SpectralSphere([simple_zero_rank_parameter(True)])
    with pytest.raises(ValueError):
        opt.step()

    x = torch.full((2, 2), 10.0)
    theta = torch.full((2, 2), 0.01)
    _ = solve_lambda_with_bisection(x, theta)

    x = torch.tensor([[5.0, 1.0], [1.0, 5.0]])
    theta = torch.tensor([[1.5, 0.0], [0.0, -2.8]])
    _ = solve_lambda_with_bisection(x, theta, initial_guess=0.18, initial_step=0.012, msign_steps=0)


def test_rose_optimizer():
    with pytest.raises(ValueError):
        load_optimizer('rose')([simple_parameter(True)], compute_dtype=torch.bfloat16)

    opt = load_optimizer('rose')([simple_zero_rank_parameter(True)])
    opt.step()

    opt = load_optimizer('rose')([simple_parameter(True)], bf16_sr=False, compute_dtype=torch.float32)
    opt.step()


def _paired_lora_rite_parameters():
    param_left = nn.Parameter(torch.tensor([[1.0, 0.2, -0.3], [0.8, 0.5, -0.7]]))
    param_right = nn.Parameter(torch.tensor([[0.6, -0.4], [0.1, 0.9], [-0.8, 0.3], [0.7, -0.2]]))
    param_left.grad = torch.tensor([[0.2, 0.3, -0.2], [-0.1, 0.4, 0.5]])
    param_right.grad = torch.tensor([[0.1, -0.3], [0.2, 0.4], [-0.5, 0.1], [0.3, -0.2]])
    return param_left, param_right


def test_lora_rite_helper_methods():
    helper = LoRARiteHelper()

    tensor = torch.arange(6.0).reshape(2, 3)
    moved, shape = helper.move_lora_dim_to_last(tensor, 0)

    assert torch.equal(helper.restore_original_shape_and_dim(moved, 0, shape), tensor)
    assert helper.move_lora_dim_to_last(torch.tensor(1.0), 0)[0].shape == (1, 1)
    assert torch.isnan(helper.inf_to_nan(torch.tensor(float('inf'))))
    assert torch.isinf(LoRARiteHelper(maybe_inf_to_nan=False).inf_to_nan(torch.tensor(float('inf'))))
    assert helper.bias_corrected_decay(0, 0.9) == 0.0

    symmetric = helper.make_symmetric(torch.tensor([[1.0, 2.0], [0.0, 3.0]]))
    assert torch.allclose(symmetric, torch.tensor([[1.0, 1.0], [1.0, 3.0]]))

    preconditioner = helper.create_preconditioner(torch.zeros(3, 2))
    assert preconditioner.shape == (2, 2)

    inverse_root = helper.inverse_sqrt(torch.eye(2), torch.tensor(0.0), eps=1e-6, eps_root=0.0, relative_epsilon=False)
    assert torch.allclose(inverse_root, torch.eye(2) / (1.0 + 1e-6))

    relative_inverse_root = helper.inverse_sqrt(
        torch.eye(2), torch.tensor(0.0), eps=1e-6, eps_root=1e-4, relative_epsilon=True
    )
    assert torch.isfinite(relative_inverse_root).all()

    update = torch.tensor([[3.0, 4.0]])
    assert torch.allclose(helper.skip_update(update, 1.0), torch.zeros_like(update))
    assert torch.equal(helper.skip_update(update, 10.0), update)
    assert helper.reduce_rms(helper.clip_update(update, 1.0)) <= 1.0

    escape = helper.get_unmagnified_rotate_second_escape(torch.zeros(2, 2), torch.eye(2))
    assert torch.allclose(escape, torch.tensor(1.0))


def test_lora_rite_rich_options_and_existing_state():
    param_left, param_right = _paired_lora_rite_parameters()

    optimizer = load_optimizer('lorarite')(
        [param_left, param_right],
        lr=5e-3,
        betas=(0.5, 0.9),
        eps=1e-4,
        relative_epsilon=True,
        clip_unmagnified_grad=1e-3,
        update_capping=1e-2,
        update_skipping=10.0,
        apply_escape=True,
        balance_param=True,
        maximize=True,
        maybe_inf_to_nan=False,
    )

    optimizer.step()
    param_left.grad = torch.full_like(param_left, 0.3)
    param_right.grad = torch.full_like(param_right, -0.2)
    optimizer.step()

    assert torch.linalg.norm(param_left).sub(torch.linalg.norm(param_right)).abs() < 1e-4


def test_lora_rite_skips_large_updates_and_missing_pair():
    param_left, param_right = _paired_lora_rite_parameters()
    initial_left = param_left.detach().clone()
    initial_right = param_right.detach().clone()
    optimizer = load_optimizer('lorarite')([param_left, param_right], lr=1e-1, betas=(0.0, 0.0), update_skipping=1e-12)
    optimizer.step()

    assert torch.allclose(param_left, initial_left)
    assert torch.allclose(param_right, initial_right)

    orphan = simple_parameter(require_grad=True)
    orphan.grad = torch.ones_like(orphan)
    no_pair_optimizer = load_optimizer('lorarite')([orphan])
    no_pair_optimizer.step()
    assert torch.allclose(orphan, torch.zeros_like(orphan))

    paired_left, paired_right = _paired_lora_rite_parameters()
    paired_right.grad = None
    missing_grad_optimizer = load_optimizer('lorarite')([paired_left, paired_right])
    missing_grad_optimizer.step()
    assert torch.allclose(paired_left, torch.tensor([[1.0, 0.2, -0.3], [0.8, 0.5, -0.7]]))


def test_flash_adamw_quantized_state_and_compressed_state_dict():
    param = nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
    param.grad = torch.tensor([0.2, -0.3, 0.4])

    optimizer = load_optimizer('flashadamw')([param], lr=1e-2, weight_decay=0.0)
    optimizer.step()

    state = optimizer.state[param]
    assert state['exp_avg::quantized'].dtype == torch.int8
    assert state['exp_avg_sq::quantized'].dtype == torch.uint8
    assert state['exp_avg::scales'].dtype == torch.float16

    assert 'exp_avg' not in state
    assert optimizer.state_dict()['state'][0]['exp_avg::quantized'].dtype == torch.int8

    new_param = nn.Parameter(param.detach().clone())
    new_optimizer = load_optimizer('flashadamw')([new_param], lr=1e-2, weight_decay=0.0)
    new_optimizer.load_state_dict(optimizer.state_dict())
    assert new_optimizer.state[new_param]['exp_avg::quantized'].dtype == torch.int8

    new_param.grad = torch.tensor([-0.1, 0.5, -0.2])
    new_optimizer.step()

    assert torch.isfinite(new_param).all()


def test_flash_adamw_empty_quantized_state():
    param = nn.Parameter(torch.empty(0))
    param.grad = torch.empty(0)

    optimizer = load_optimizer('flashadamw')([param], lr=1e-2, weight_decay=0.0)
    optimizer.step()

    param.grad = torch.empty(0)
    optimizer.step()

    state = optimizer.state[param]
    assert state['exp_avg::quantized'].numel() == 0
    assert state['exp_avg::scales'].numel() == 0


def test_flash_adamw_loads_compressed_state_as_uncompressed_state():
    param = nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
    param.grad = torch.tensor([0.2, -0.3, 0.4])

    optimizer = load_optimizer('flashadamw')([param], lr=1e-2, weight_decay=0.0)
    optimizer.step()

    new_param = nn.Parameter(param.detach().clone())
    state_dict = optimizer.state_dict()
    state_dict['param_groups'][0]['quantize'] = False

    new_optimizer = load_optimizer('flashadamw')([new_param], lr=1e-2, weight_decay=0.0, quantize=False)
    new_optimizer.load_state_dict(state_dict)
    new_state = new_optimizer.state[new_param]
    assert 'exp_avg' in new_state
    assert 'exp_avg::quantized' not in new_state

    uncompressed_optimizer = load_optimizer('flashadamw')([nn.Parameter(param.detach().clone())], quantize=False)
    uncompressed_optimizer.load_state_dict(uncompressed_optimizer.state_dict())
    assert list(uncompressed_optimizer.state_dict()['state'].values()) == [{}]

    raw_param = nn.Parameter(param.detach().clone())
    raw_param.grad = torch.zeros_like(raw_param)
    raw_optimizer = load_optimizer('flashadamw')([raw_param], quantize=False, compress_state_dict=False)
    raw_optimizer.step()
    assert 'exp_avg' in raw_optimizer.state_dict()['state'][0]


def test_flash_adamw_uncompressed_state_dict_reloads_as_quantized_state():
    param = nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
    param.grad = torch.tensor([0.2, -0.3, 0.4])

    optimizer = load_optimizer('flashadamw')([param], lr=1e-2, weight_decay=0.0, compress_state_dict=False)
    optimizer.step()

    state_dict = optimizer.state_dict()
    saved_state = state_dict['state'][0]
    assert 'exp_avg' in saved_state
    assert 'exp_avg::quantized' not in saved_state

    new_optimizer = load_optimizer('flashadamw')(
        [nn.Parameter(param.detach().clone())],
        lr=1e-2,
        weight_decay=0.0,
    )
    new_optimizer.load_state_dict(state_dict)
    new_state = next(iter(new_optimizer.state.values()))
    assert 'exp_avg::quantized' in new_state
    assert 'exp_avg' not in new_state


@pytest.mark.parametrize(('master_weight_bits', 'error_dtype'), [(24, torch.int8), (32, torch.int16)])
def test_flash_adamw_master_weight_bits(master_weight_bits, error_dtype):
    model = nn.Linear(2, 1).bfloat16()
    for param in model.parameters():
        param.grad = torch.ones_like(param)

    optimizer = load_optimizer('flashadamw')(
        model.parameters(), lr=1e-2, weight_decay=0.0, quantize=False, master_weight_bits=master_weight_bits
    )
    optimizer.step()

    fp32_state = optimizer.get_fp32_model_state_dict(model)
    assert all(tensor.dtype == torch.float32 for tensor in fp32_state.values())
    assert all(state['error_bits'].dtype == error_dtype for state in optimizer.state.values())

    updated = {name: tensor.add(0.01) for name, tensor in fp32_state.items()}
    optimizer.set_fp32_model_state_dict(model, updated)

    restored = optimizer.get_fp32_model_state_dict(model)
    assert all(torch.allclose(restored[name], updated[name], atol=1e-2) for name in updated)


def test_flash_adamw_fresh_fp32_model_state_dict():
    model = nn.Linear(2, 1).bfloat16()

    optimizer = load_optimizer('flashadamw')(
        model.parameters(),
        lr=1e-2,
        weight_decay=0.0,
        quantize=False,
        master_weight_bits=24,
    )

    initial = optimizer.get_fp32_model_state_dict(model)
    assert all(tensor.dtype == torch.float32 for tensor in initial.values())

    optimizer.set_fp32_model_state_dict(model, {'missing.weight': torch.ones(1)})
    updated = {name: tensor.add(0.01) for name, tensor in initial.items()}

    optimizer.set_fp32_model_state_dict(model, updated)
    assert all('error_bits' in state for state in optimizer.state.values())


def test_flash_adamw_ecc_helpers():
    fp32_param = torch.tensor([1.01, -2.02], dtype=torch.float32)
    narrow_param = fp32_param.to(torch.bfloat16)
    error_bits = compute_ecc_bits(fp32_param, narrow_param, master_byte_width=3)
    reconstructed = reconstruct_fp32_param(narrow_param, error_bits)

    assert error_bits.dtype == torch.int8
    assert torch.allclose(reconstructed, fp32_param, atol=1e-2)

    with pytest.raises(ValueError):
        compute_ecc_bits(fp32_param.to(torch.float16), narrow_param, master_byte_width=3)
    with pytest.raises(ValueError):
        compute_ecc_bits(fp32_param, fp32_param, master_byte_width=3)
    with pytest.raises(ValueError):
        compute_ecc_bits(fp32_param, narrow_param, master_byte_width=2)

    with pytest.raises(ValueError):
        reconstruct_fp32_param(fp32_param, error_bits)
    with pytest.raises(ValueError):
        reconstruct_fp32_param(narrow_param, torch.ones_like(narrow_param))


def test_flash_adamw_numerics_guard_and_stats():
    param = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
    optimizer = load_optimizer('flashadamw')([param], lr=1e-3, weight_decay=0.0, quantize=False, check_numerics=True)

    optimizer.recompute_param_stats()
    optimizer.maybe_check_numerics(param, lr=0.0, master_byte_width=0)

    param.data.zero_()
    optimizer.param_absmax.pop(id(param), None)
    optimizer.maybe_check_numerics(param, lr=1e-3, master_byte_width=0)

    param.data.fill_(1.0)
    optimizer.param_absmax[id(param)] = float('nan')
    optimizer.maybe_check_numerics(param, lr=1e-3, master_byte_width=0)

    optimizer.param_absmax[id(param)] = 1.0
    with pytest.raises(ArithmeticError):
        optimizer.maybe_check_numerics(param, lr=1e-12, master_byte_width=2)

    empty_param = nn.Parameter(torch.empty(0, dtype=torch.bfloat16))
    empty_optimizer = load_optimizer('flashadamw')([empty_param], lr=1e-3, weight_decay=0.0, quantize=False)
    empty_optimizer.recompute_param_stats()
    assert empty_optimizer.param_absmax[id(empty_param)] == 0.0


def test_flash_adamw_parameters():
    with pytest.raises(ValueError):
        load_optimizer('flashadamw')(None, master_weight_bits=16)

    with pytest.raises(NotImplementedError):
        load_optimizer('flashadamw')(None, fused=True)

    with pytest.raises(ValueError):
        load_optimizer('flashadamw')([nn.Parameter(torch.ones(1))], master_weight_bits=24)
