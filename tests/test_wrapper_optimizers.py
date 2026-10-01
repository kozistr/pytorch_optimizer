from copy import deepcopy
from io import BytesIO

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
    FriendlySAM,
    Lookahead,
    LookSAM,
    Magma,
    OrthoGrad,
    PCGrad,
    ProportionScheduler,
    ScheduleFreeWrapper,
    load_optimizer,
)
from tests.constants import PULLBACK_MOMENTUM
from tests.utils import (
    Example,
    MultiHeadLogisticRegression,
    Trainer,
    build_model,
    simple_parameter,
    tensor_to_numpy,
)


def accelerate_style_move_to_device(state, device):
    if isinstance(state, torch.Tensor):
        return state.to(device)
    if isinstance(state, (list, tuple)):
        return type(state)(accelerate_style_move_to_device(v, device) for v in state)
    if isinstance(state, dict):
        return type(state)({k: accelerate_style_move_to_device(v, device) for k, v in state.items()})
    return state


@pytest.mark.parametrize('pullback_momentum', PULLBACK_MOMENTUM)
def test_lookahead(pullback_momentum, environment):
    x_data, y_data = environment
    model, loss_fn = build_model(device=x_data.device)

    optimizer = Lookahead(load_optimizer('adamw')(model.parameters(), lr=5e-1), pullback_momentum=pullback_momentum)
    optimizer.init_group({})

    trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
    trainer.run(iterations=5, threshold=2.0)


def test_lookahead_state_dict_with_accelerate_style_mapping():
    model = Example()
    optimizer = Lookahead(load_optimizer('adamw')(model.parameters(), lr=1e-3))

    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p)

    optimizer.step()

    state_dict = optimizer.state_dict()
    assert isinstance(state_dict['lookahead_state'], dict)

    moved_state = accelerate_style_move_to_device(state_dict, torch.device('cpu'))
    optimizer.load_state_dict(moved_state)


@pytest.mark.parametrize('pullback_momentum', PULLBACK_MOMENTUM)
def test_lookahead_resume_with_new_parameters(pullback_momentum):
    parameters = [nn.Parameter(torch.tensor([1.0])), nn.Parameter(torch.tensor([2.0, -1.0]))]
    optimizer = Lookahead(
        torch.optim.SGD([{'params': [p], 'lr': lr} for p, lr in zip(parameters, [0.1, 0.05])], momentum=0.9),
        k=2,
        pullback_momentum=pullback_momentum,
    )
    for p in parameters:
        p.grad = torch.ones_like(p)
    optimizer.step()

    restored_parameters = [nn.Parameter(p.detach().clone()) for p in parameters]
    restored = Lookahead(
        torch.optim.SGD([{'params': [p]} for p in restored_parameters], lr=0.1, momentum=0.9),
        k=2,
        pullback_momentum=pullback_momentum,
    )
    restored.load_state_dict(deepcopy(optimizer.state_dict()))

    for gradient in (-0.5, 0.25, 1.0):
        for p, restored_p in zip(parameters, restored_parameters):
            p.grad = torch.full_like(p, gradient)
            restored_p.grad = p.grad.clone()
        optimizer.step()
        restored.step()

        for p, restored_p in zip(parameters, restored_parameters):
            torch.testing.assert_close(restored_p, p)
            if pullback_momentum == 'pullback':
                assert (
                    optimizer.state[p]['slow_momentum'].data_ptr()
                    != optimizer.optimizer.state[p]['momentum_buffer'].data_ptr()
                )
            torch.testing.assert_close(restored.state[restored_p]['slow_params'], optimizer.state[p]['slow_params'])
            torch.testing.assert_close(
                restored.optimizer.state[restored_p]['momentum_buffer'],
                optimizer.optimizer.state[p]['momentum_buffer'],
            )


@pytest.mark.parametrize('mismatch', ['legacy', 'missing', 'extra'])
def test_lookahead_rejects_mismatched_state(mismatch):
    parameters = [nn.Parameter(torch.tensor([1.0])), nn.Parameter(torch.tensor([2.0]))]
    optimizer = Lookahead(torch.optim.SGD(parameters, lr=0.1))
    state_dict = deepcopy(optimizer.state_dict())
    if mismatch == 'legacy':
        state_dict['lookahead_state'] = {
            nn.Parameter(p.detach().clone()): dict(optimizer.state[p]) for p in parameters
        }
    elif mismatch == 'missing':
        del state_dict['lookahead_state'][(0, 1)]
    else:
        state_dict['lookahead_state'][(0, 2)] = deepcopy(state_dict['lookahead_state'][(0, 0)])

    original_state = optimizer.state
    with pytest.raises(ValueError, match='lookahead state does not match the current parameters'):
        optimizer.load_state_dict(state_dict)
    assert optimizer.state is original_state


def test_magma(environment):
    x_data, y_data = environment
    model, loss_fn = build_model(device=x_data.device)

    optimizer = Magma(load_optimizer('adamw')(model.parameters(), lr=5e-1), mask_prob=1.0)

    trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
    trainer.run(iterations=5, threshold=2.0)


def test_magma_str_and_closure():
    parameter = simple_parameter()
    optimizer = Magma(torch.optim.SGD([parameter], lr=1e-1), mask_prob=1.0)

    def closure():
        parameter.grad = torch.ones_like(parameter)
        return parameter.sum()

    assert str(optimizer) == 'Magma'
    optimizer.step(closure)


def test_magma_accepts_optimizer_class_and_adds_param_group():
    parameter = simple_parameter()
    optimizer = Magma(torch.optim.SGD, params=[parameter], lr=1e-1)

    new_parameter = simple_parameter()
    optimizer.add_param_group({'params': [new_parameter]})

    assert new_parameter in optimizer.param_groups[-1]['params']


def test_magma_loads_magma_state():
    parameter = simple_parameter()
    optimizer = Magma(torch.optim.SGD([parameter], lr=1e-1), mask_prob=1.0)

    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    new_parameter = simple_parameter()
    new_optimizer = Magma(torch.optim.SGD([new_parameter], lr=1e-1))
    new_optimizer.load_state_dict(optimizer.state_dict())

    assert {'alignment', 'momentum'} <= new_optimizer.state_dict()['magma_state'][(0, 0)].keys()


def test_magma_moment_selection():
    parameter = simple_parameter()
    optimizer = Magma(torch.optim.SGD([parameter], lr=1e-1))

    assert optimizer._get_first_moment(parameter) is None

    optimizer.moment_key = None
    assert optimizer._get_first_moment(parameter) is None

    optimizer.moment_key = 'custom'
    optimizer.optimizer.state[parameter]['custom'] = torch.ones_like(parameter)
    assert optimizer._get_first_moment(parameter) is not None

    optimizer.optimizer.state[parameter]['custom'] = None
    assert optimizer._get_first_moment(parameter) is None

    optimizer.moment_key = 'auto'
    assert optimizer._get_first_moment(parameter) is None


def test_magma_masks_parameters_and_updates_base_state():
    parameter = simple_parameter()
    optimizer = Magma(torch.optim.SGD([parameter], lr=1e-1, momentum=0.9), mask_prob=0.0)

    parameter.grad = torch.ones_like(parameter)
    initial_parameter = parameter.detach().clone()
    optimizer.step()

    assert torch.equal(parameter, initial_parameter)
    assert 'momentum_buffer' in optimizer.state[parameter]
    assert 'alignment' in optimizer.state_dict()['magma_state'][(0, 0)]


def test_magma_excludes_parameters_and_loads_state_dict():
    parameter = simple_parameter()
    optimizer = Magma(torch.optim.AdamW([parameter], lr=1e-1), mask_prob=0.0, exclude={parameter})

    parameter.grad = torch.ones_like(parameter)
    initial_parameter = parameter.detach().clone()
    optimizer.step()

    assert not torch.equal(parameter, initial_parameter)
    state_dict = optimizer.state_dict()

    new_parameter = simple_parameter()
    new_optimizer = Magma(torch.optim.AdamW([new_parameter], lr=1e-1))
    new_optimizer.load_state_dict(state_dict)
    assert new_optimizer.state[new_parameter].keys() == optimizer.state[parameter].keys()


@pytest.mark.parametrize('adaptive', [True, False])
@pytest.mark.parametrize('wrapper', [SAM, FriendlySAM, LookSAM])
def test_sam_optimizer(adaptive, wrapper, environment):
    x_data, y_data = environment
    model, loss_fn = build_model(device=x_data.device)

    optimizer = wrapper(model.parameters(), load_optimizer('asgd'), lr=5e-1, adaptive=adaptive, use_gc=True)

    trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
    trainer.run_sam_style(iterations=3, threshold=2.0)


@pytest.mark.parametrize('adaptive', [True, False])
@pytest.mark.parametrize('wrapper', [SAM, FriendlySAM, LookSAM])
def test_sam_optimizer_with_closure(adaptive, wrapper, environment):
    x_data, y_data = environment
    model, loss_fn = build_model(device=x_data.device)

    optimizer = wrapper(model.parameters(), load_optimizer('adamw'), lr=5e-1, adaptive=adaptive)

    trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
    trainer.run_with_closure(iterations=3, threshold=2.0)


@pytest.mark.parametrize('adaptive', [True, False])
@pytest.mark.parametrize('decouple', [True, False])
def test_wsam_optimizer(adaptive, decouple, environment):
    x_data, y_data = environment
    model, loss_fn = build_model(device=x_data.device)

    optimizer = WSAM(
        model,
        model.parameters(),
        load_optimizer('adamp'),
        lr=5e-2,
        adaptive=adaptive,
        decouple=decouple,
        max_norm=100.0,
    )

    trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
    trainer.run_sam_style(iterations=10, threshold=1.5)


@pytest.mark.parametrize('adaptive', [True, False])
def test_wsam_optimizer_with_closure(adaptive, environment):
    x_data, y_data = environment
    model, loss_fn = build_model(device=x_data.device)

    optimizer = WSAM(model, model.parameters(), load_optimizer('adamp'), lr=5e-2, adaptive=adaptive, max_norm=100.0)

    trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
    trainer.run_wsam_with_closure(iterations=10, threshold=1.5)


@pytest.mark.parametrize('adaptive', [True, False])
def test_gsam_optimizer(adaptive, environment):
    pytest.skip('skip GSAM optimizer')

    x_data, y_data = environment
    model, loss_fn = build_model(device=x_data.device)

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


@pytest.mark.parametrize('adaptive', [True, False])
def test_bsam_optimizer(adaptive, environment, monkeypatch):
    x_data, y_data = environment
    model, loss_fn = build_model(device=x_data.device)

    normal = torch.normal

    def sample_noise(mean, std):
        return normal(mean, std.cpu()).to(std.device)

    # Use the same noise samples when comparing CPU and CUDA convergence.
    monkeypatch.setattr(torch, 'normal', sample_noise)

    optimizer = BSAM(model.parameters(), lr=2e-3, num_data=len(x_data), rho=1e-5, adaptive=adaptive)

    trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
    trainer.run_with_closure(iterations=20, threshold=1.0)


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


def test_schedulefree_wrapper_legacy_state_dict():
    parameter = simple_parameter()
    optimizer = ScheduleFreeWrapper(torch.optim.SGD([parameter], lr=0.1))
    optimizer.train()
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    legacy_state = {'schedulefree_state': optimizer.state, 'base_optimizer': optimizer.optimizer.state_dict()}
    same_parameter = ScheduleFreeWrapper(torch.optim.SGD([parameter], lr=0.1))
    same_parameter.load_state_dict(legacy_state)
    torch.testing.assert_close(same_parameter.state[parameter]['z'], optimizer.state[parameter]['z'])

    different_parameter = ScheduleFreeWrapper(torch.optim.SGD([simple_parameter()], lr=0.1))
    with pytest.raises(ValueError, match='schedule-free state does not match'):
        different_parameter.load_state_dict(legacy_state)


@pytest.mark.parametrize('reduction', ['mean', 'sum'])
def test_pc_grad_optimizers(reduction, environment):
    torch.manual_seed(42)

    x_data, y_data = environment

    model: nn.Module = MultiHeadLogisticRegression().to(x_data.device)
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


@pytest.mark.parametrize('reduction', ['mean', 'sum'])
def test_pcgrad_preserves_unused_parameters(reduction):
    shared = nn.Parameter(torch.tensor([1.0]))
    task_specific = nn.Parameter(torch.tensor([2.0]))
    unused = nn.Parameter(torch.tensor([3.0]))
    optimizer = PCGrad(torch.optim.SGD([shared, task_specific, unused], lr=0.1, weight_decay=0.2), reduction)

    optimizer.pc_backward([(shared - 1.0).square().sum() + task_specific.sum(), (shared - 1.0).square().sum()])

    torch.testing.assert_close(shared.grad, torch.zeros_like(shared))
    torch.testing.assert_close(task_specific.grad, torch.ones_like(task_specific))
    assert unused.grad is None

    optimizer.step()
    torch.testing.assert_close(shared, torch.tensor([0.98]))
    torch.testing.assert_close(task_specific, torch.tensor([1.86]))
    torch.testing.assert_close(unused, torch.tensor([3.0]))


def test_trac_optimizer(environment):
    x_data, y_data = environment
    model, loss_fn = build_model(device=x_data.device)

    optimizer = TRAC(load_optimizer('adamw')(model.parameters(), lr=1e0))

    trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
    trainer.run_trac_style(iterations=3, threshold=2.0)


def test_trac_optimizer_erf_imag():
    model = Example()

    optimizer = TRAC(load_optimizer('adamw')(model.parameters()))
    optimizer.zero_grad()

    complex_tensor = torch.complex(torch.tensor(0.0), torch.tensor(1.0))
    optimizer.erf_imag(complex_tensor)

    assert str(optimizer).lower() == 'trac'


@pytest.mark.parametrize('checkpoint_format', ['indexed', 'legacy'])
def test_trac_checkpoint_resumes_with_new_parameters(checkpoint_format):
    parameters = [nn.Parameter(torch.tensor([1.0])), nn.Parameter(torch.tensor([-1.0, 2.0]))]
    optimizer = TRAC(
        torch.optim.SGD([{'params': [p], 'lr': lr} for p, lr in zip(parameters, [0.1, 0.05])], momentum=0.9)
    )
    for gradient in (0.5, -0.25, 1.0):
        for p in parameters:
            p.grad = torch.full_like(p, gradient)
        optimizer.step()

    state_dict = optimizer.state_dict() if checkpoint_format == 'indexed' else optimizer.optimizer.state_dict()
    if checkpoint_format == 'indexed':
        assert all(isinstance(key, (str, int)) for key in state_dict['state']['trac'])
    assert all(p in optimizer.state['trac'] for p in parameters)
    stream = BytesIO()
    torch.save(state_dict, stream)
    stream.seek(0)
    saved_state = torch.load(stream, weights_only=False)

    restored_parameters = [nn.Parameter(p.detach().clone()) for p in parameters]
    restored = TRAC(torch.optim.SGD([{'params': [p]} for p in restored_parameters], lr=0.1, momentum=0.9))
    restored.load_state_dict(saved_state)

    for gradient in (-0.5, 0.25, 1.0):
        for p, restored_p in zip(parameters, restored_parameters):
            p.grad = torch.full_like(p, gradient)
            restored_p.grad = p.grad.clone()
        optimizer.step()
        restored.step()

        for p, restored_p in zip(parameters, restored_parameters):
            torch.testing.assert_close(restored_p, p)
            torch.testing.assert_close(restored.state['trac'][restored_p], optimizer.state['trac'][p])
            torch.testing.assert_close(
                restored.optimizer.state[restored_p]['momentum_buffer'],
                optimizer.optimizer.state[p]['momentum_buffer'],
            )
        for key in ('s', 'variance', 'sigma'):
            torch.testing.assert_close(restored.state['trac'][key], optimizer.state['trac'][key])
        assert restored.state['trac']['step'] == optimizer.state['trac']['step']


@pytest.mark.parametrize('wrapper_optimizer_instance', [Lookahead, OrthoGrad, TRAC])
def test_load_wrapper_optimizer(wrapper_optimizer_instance):
    params = [simple_parameter()]

    _ = wrapper_optimizer_instance(torch.optim.AdamW(params))

    optimizer = wrapper_optimizer_instance(torch.optim.AdamW, params=params)
    optimizer.init_group({'params': []}, updates=[])
    optimizer.zero_grad()

    with pytest.raises(ValueError):
        wrapper_optimizer_instance(torch.optim.AdamW)

    _ = optimizer.param_groups
    _ = optimizer.state

    state = optimizer.state_dict()
    optimizer.load_state_dict(state)


def test_trac_rejects_missing_checkpoint_reference():
    parameter = nn.Parameter(torch.tensor([1.0]))
    optimizer = TRAC(torch.optim.SGD([parameter], lr=0.1))
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state_dict = deepcopy(optimizer.state_dict())
    del state_dict['state']['trac'][0]

    with pytest.raises(ValueError, match='TRAC state does not match'):
        optimizer.load_state_dict(state_dict)
