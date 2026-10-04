from collections import defaultdict
from copy import deepcopy
from io import BytesIO

import numpy as np
import pytest
import torch
from torch import nn

from pytorch_optimizer import (
    BSAM,
    SAM,
    TRAC,
    WSAM,
    FriendlySAM,
    Lookahead,
    LookSAM,
    Magma,
    OrthoGrad,
    PCGrad,
    ScheduleFreeWrapper,
    load_optimizer,
)
from pytorch_optimizer.base.exception import NoClosureError, NoSparseGradientError
from tests.fixtures import TrainingModel, build_model, make_parameter, make_sparse_parameters
from tests.utils import Trainer, build_optimizer, tensor_to_numpy

PULLBACK_MOMENTUM: tuple[str, ...] = ('none', 'reset', 'pullback')


def accelerate_style_move_to_device(state, device):
    if isinstance(state, torch.Tensor):
        return state.to(device)
    if isinstance(state, (list, tuple)):
        return type(state)(accelerate_style_move_to_device(v, device) for v in state)
    if isinstance(state, dict):
        return type(state)({k: accelerate_style_move_to_device(v, device) for k, v in state.items()})
    return state


@pytest.mark.parametrize('wrapper_optimizer_instance', [Lookahead, OrthoGrad, TRAC])
def test_load_wrapper_optimizer(wrapper_optimizer_instance):
    params = [make_parameter()]

    _ = wrapper_optimizer_instance(build_optimizer('adamw', params))

    optimizer = wrapper_optimizer_instance(load_optimizer('adamw'), params=params)
    optimizer.init_group({'params': []}, updates=[])
    optimizer.zero_grad()

    with pytest.raises(ValueError):
        wrapper_optimizer_instance(load_optimizer('adamw'))

    _ = optimizer.param_groups
    _ = optimizer.state

    state = optimizer.state_dict()
    optimizer.load_state_dict(state)


class TestLookahead:
    @pytest.mark.parametrize('pullback_momentum', PULLBACK_MOMENTUM)
    def test_lookahead(self, pullback_momentum, environment):
        x_data, y_data = environment
        model, loss_fn = build_model(device=x_data.device)

        optimizer = Lookahead(
            build_optimizer('adamw', model.parameters(), lr=5e-1), pullback_momentum=pullback_momentum
        )
        optimizer.init_group({})

        trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
        trainer.run(iterations=5, threshold=2.0)

    def test_lookahead_state_dict_with_accelerate_style_mapping(self):
        model = TrainingModel()
        optimizer = Lookahead(build_optimizer('adamw', model.parameters(), lr=1e-3))

        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p)

        optimizer.step()

        state_dict = optimizer.state_dict()
        assert isinstance(state_dict['lookahead_state'], dict)

        moved_state = accelerate_style_move_to_device(state_dict, torch.device('cpu'))
        optimizer.load_state_dict(moved_state)

    @pytest.mark.parametrize('pullback_momentum', PULLBACK_MOMENTUM)
    def test_lookahead_resume_with_new_parameters(self, pullback_momentum):
        parameters = [nn.Parameter(torch.tensor([1.0])), nn.Parameter(torch.tensor([2.0, -1.0]))]
        optimizer = Lookahead(
            build_optimizer(
                'sgd', [{'params': [p], 'lr': lr} for p, lr in zip(parameters, [0.1, 0.05])], momentum=0.9
            ),
            k=2,
            pullback_momentum=pullback_momentum,
        )

        for p in parameters:
            p.grad = torch.ones_like(p)

        optimizer.step()

        restored_parameters = [nn.Parameter(p.detach().clone()) for p in parameters]
        restored = Lookahead(
            build_optimizer('sgd', [{'params': [p]} for p in restored_parameters], lr=0.1, momentum=0.9),
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

                torch.testing.assert_close(
                    restored.state[restored_p]['slow_params'], optimizer.state[p]['slow_params']
                )

                torch.testing.assert_close(
                    restored.optimizer.state[restored_p]['momentum_buffer'],
                    optimizer.optimizer.state[p]['momentum_buffer'],
                )

    @pytest.mark.parametrize('mismatch', ['legacy', 'missing', 'extra'])
    def test_lookahead_rejects_mismatched_state(self, mismatch):
        parameters = [nn.Parameter(torch.tensor([1.0])), nn.Parameter(torch.tensor([2.0]))]
        optimizer = Lookahead(build_optimizer('sgd', parameters, lr=0.1))
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

    def test_lookahead_parameters(self):
        optimizer_instance = load_optimizer('adamp')
        optimizer = optimizer_instance([make_parameter()])

        for pullback_momentum in ('none', 'reset', 'pullback'):
            opt = Lookahead(optimizer, pullback_momentum=pullback_momentum)
            assert not opt.state[optimizer.param_groups[0]['params'][0]]['slow_params'].requires_grad

            opt.load_state_dict(opt.state_dict())

        opt = Lookahead(optimizer, pullback_momentum=pullback_momentum)
        opt.backup_and_load_cache()

        assert not opt.state[optimizer.param_groups[0]['params'][0]]['backup_params'].requires_grad
        opt.clear_and_load_backup()

        _ = opt.__getstate__()

        with pytest.raises(ValueError):
            Lookahead(optimizer, k=0)

        with pytest.raises(ValueError):
            Lookahead(optimizer, alpha=-0.1)

        with pytest.raises(ValueError):
            Lookahead(optimizer, pullback_momentum='invalid')

    def test_lookahead_load_legacy_defaultdict_state(self):
        parameter = make_parameter()
        optimizer = build_optimizer('adamp', [parameter], lr=1e-2)
        lookahead = Lookahead(optimizer)

        parameter.grad = torch.randn_like(parameter)
        lookahead.step()

        state_dict = lookahead.state_dict()
        legacy_lookahead_state = defaultdict(
            dict, {p: dict(param_state) for p, param_state in lookahead.state.items()}
        )
        lookahead.load_state_dict(
            {'lookahead_state': legacy_lookahead_state, 'base_optimizer': state_dict['base_optimizer']}
        )

        parameter.grad = torch.randn_like(parameter)
        lookahead.step()

        assert isinstance(lookahead.state, defaultdict)
        assert 'slow_params' in lookahead.state[parameter]


class TestMagma:
    def test_magma(self, environment):
        x_data, y_data = environment
        model, loss_fn = build_model(device=x_data.device)

        optimizer = Magma(build_optimizer('adamw', model.parameters(), lr=5e-1), mask_prob=1.0)

        trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
        trainer.run(iterations=5, threshold=2.0)

    def test_magma_str_and_closure(self):
        parameter = make_parameter()
        optimizer = Magma(build_optimizer('sgd', [parameter], lr=1e-1), mask_prob=1.0)

        def closure():
            parameter.grad = torch.ones_like(parameter)
            return parameter.sum()

        assert str(optimizer) == 'Magma'
        optimizer.step(closure)

    def test_magma_accepts_optimizer_class_and_adds_param_group(self):
        parameter = make_parameter()
        optimizer = Magma(load_optimizer('sgd'), params=[parameter], lr=1e-1)

        new_parameter = make_parameter()
        optimizer.add_param_group({'params': [new_parameter]})

        assert new_parameter in optimizer.param_groups[-1]['params']

    def test_magma_loads_magma_state(self):
        parameter = make_parameter()
        optimizer = Magma(build_optimizer('sgd', [parameter], lr=1e-1), mask_prob=1.0)

        parameter.grad = torch.ones_like(parameter)
        optimizer.step()

        new_parameter = make_parameter()
        new_optimizer = Magma(build_optimizer('sgd', [new_parameter], lr=1e-1))
        new_optimizer.load_state_dict(optimizer.state_dict())

        assert {'alignment', 'momentum'} <= new_optimizer.state_dict()['magma_state'][(0, 0)].keys()

    def test_magma_moment_selection(self):
        parameter = make_parameter()
        optimizer = Magma(build_optimizer('sgd', [parameter], lr=1e-1))

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

    def test_magma_masks_parameters_and_updates_base_state(self):
        parameter = make_parameter()
        optimizer = Magma(build_optimizer('sgd', [parameter], lr=1e-1, momentum=0.9), mask_prob=0.0)

        parameter.grad = torch.ones_like(parameter)
        initial_parameter = parameter.detach().clone()
        optimizer.step()

        assert torch.equal(parameter, initial_parameter)
        assert 'momentum_buffer' in optimizer.state[parameter]
        assert 'alignment' in optimizer.state_dict()['magma_state'][(0, 0)]

    def test_magma_excludes_parameters_and_loads_state_dict(self):
        parameter = make_parameter()
        optimizer = Magma(build_optimizer('adamw', [parameter], lr=1e-1), mask_prob=0.0, exclude={parameter})

        parameter.grad = torch.ones_like(parameter)
        initial_parameter = parameter.detach().clone()
        optimizer.step()

        assert not torch.equal(parameter, initial_parameter)
        state_dict = optimizer.state_dict()

        new_parameter = make_parameter()
        new_optimizer = Magma(build_optimizer('adamw', [new_parameter], lr=1e-1))
        new_optimizer.load_state_dict(state_dict)
        assert new_optimizer.state[new_parameter].keys() == optimizer.state[parameter].keys()


class TestSAM:
    @pytest.mark.parametrize('wrapper', [WSAM, LookSAM, FriendlySAM])
    def test_checkpoint_resume(self, wrapper):
        param = make_parameter((2,), grad=1.0)

        def build(parameters):
            options = {'model': TrainingModel()} if wrapper is WSAM else {}
            return wrapper(params=parameters, base_optimizer=load_optimizer('adamw'), lr=0.01, **options)

        optimizer = build([param])
        optimizer.first_step(zero_grad=True)
        param.grad = torch.full_like(param, 0.5)
        optimizer.second_step(zero_grad=True)

        restored_param = param.detach().clone().requires_grad_()
        restored = build([restored_param])
        restored.load_state_dict(deepcopy(optimizer.state_dict()))

        for gradient in (0.25, -0.5):
            for current, parameter in ((optimizer, param), (restored, restored_param)):
                parameter.grad = torch.full_like(parameter, gradient)
                current.first_step(zero_grad=True)
                parameter.grad = torch.full_like(parameter, gradient * 0.5)
                current.second_step(zero_grad=True)

            torch.testing.assert_close(restored_param, param)
            torch.testing.assert_close(
                restored.base_optimizer.state[restored_param]['exp_avg'],
                optimizer.base_optimizer.state[param]['exp_avg'],
            )

    @pytest.mark.parametrize('adaptive', [True, False])
    @pytest.mark.parametrize('wrapper', [SAM, FriendlySAM, LookSAM])
    @pytest.mark.parametrize(('base_optimizer_name', 'use_closure'), [('asgd', False), ('adamw', True)])
    def test_sam_optimizer(self, adaptive, wrapper, base_optimizer_name, use_closure, environment):
        x_data, y_data = environment
        model, loss_fn = build_model(device=x_data.device)
        options = {} if use_closure else {'use_gc': True}

        optimizer = wrapper(
            model.parameters(), load_optimizer(base_optimizer_name), lr=5e-1, adaptive=adaptive, **options
        )

        trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
        run = trainer.run_with_closure if use_closure else trainer.run_sam_style
        run(iterations=3, threshold=2.0)

    @pytest.mark.parametrize('optimizer', [SAM, LookSAM, FriendlySAM])
    def test_sam_no_gradient(self, optimizer, environment):
        x_data, y_data = environment
        model, loss_fn = build_model(device=x_data.device)

        model.fc1.weight.requires_grad = False
        model.fc1.weight.grad = None

        optimizer = optimizer(model.parameters(), load_optimizer('adamp'))
        optimizer.zero_grad()

        loss = loss_fn(y_data, model(x_data))
        loss.backward()
        optimizer.first_step(zero_grad=True)

        loss_fn(y_data, model(x_data)).backward()
        optimizer.second_step(zero_grad=True)

    @pytest.mark.parametrize(('first_pass_active', 'second_pass_active'), [(True, False), (False, True), (True, True)])
    @pytest.mark.parametrize('wrapper', [SAM, WSAM, LookSAM, FriendlySAM])
    def test_sam_changing_gradient_availability(self, first_pass_active, second_pass_active, wrapper):
        parameter = nn.Parameter(torch.tensor([1.0]))
        always_active = nn.Parameter(torch.tensor([2.0]))
        options = {'model': TrainingModel(), 'gamma': 0.5} if wrapper is WSAM else {}
        optimizer = wrapper(
            params=[parameter, always_active], base_optimizer=load_optimizer('sgd'), lr=0.1, rho=0.1, **options
        )

        first_loss = always_active.sum() + (parameter.sum() if first_pass_active else 0.0)
        first_loss.backward()
        optimizer.first_step(zero_grad=True)

        second_loss = always_active.sum() + (parameter.sum() if second_pass_active else 0.0)
        second_loss.backward()
        optimizer.second_step(zero_grad=True)

        expected = torch.tensor([0.9 if second_pass_active else 1.0])
        torch.testing.assert_close(parameter, expected)
        torch.testing.assert_close(always_active, torch.tensor([1.9]))

        always_active.sum().backward()
        optimizer.first_step(zero_grad=True)
        always_active.sum().backward()
        optimizer.second_step(zero_grad=True)

        torch.testing.assert_close(parameter, expected)
        torch.testing.assert_close(always_active, torch.tensor([1.8]))

    @pytest.mark.parametrize('optimizer', [SAM, WSAM, LookSAM, BSAM, FriendlySAM])
    def test_sam_family_methods(self, optimizer):
        base_optimizer = load_optimizer('lion')

        opt = optimizer(params=[make_parameter()], model=None, base_optimizer=base_optimizer, num_data=1)
        opt.zero_grad()

        opt.init_group({'params': []})
        state = opt.state_dict()
        opt.load_state_dict(state)

        if 'base_optimizer' in state:
            opt.load_state_dict({key: value for key, value in state.items() if key != 'base_optimizer'})
            assert opt.param_groups is opt.base_optimizer.param_groups

        with pytest.raises(NoClosureError):
            opt.step()

        with pytest.raises(ValueError):
            optimizer(model=None, params=None, base_optimizer=base_optimizer, rho=-0.1, num_data=1)


class TestWSAM:
    @pytest.mark.parametrize('adaptive', [True, False])
    @pytest.mark.parametrize(('decouple', 'use_closure'), [(True, False), (False, False), (True, True)])
    def test_wsam_optimizer(self, adaptive, decouple, use_closure, environment):
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
        run = trainer.run_wsam_with_closure if use_closure else trainer.run_sam_style
        run(iterations=10, threshold=1.5)

    def test_wsam_no_gradient(self, environment):
        x_data, y_data = environment
        model, loss_fn = build_model(device=x_data.device)

        model.fc1.weight.requires_grad = False
        model.fc1.weight.grad = None

        optimizer = WSAM(model, model.parameters(), load_optimizer('adamp'))
        optimizer.zero_grad()

        loss = loss_fn(y_data, model(x_data))
        loss.backward()
        optimizer.first_step(zero_grad=True)

        loss_fn(y_data, model(x_data)).backward()
        optimizer.second_step(zero_grad=True)


class TestBSAM:
    @pytest.mark.parametrize('adaptive', [True, False])
    def test_bsam_optimizer(self, adaptive, environment, monkeypatch):
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


class TestScheduleFreeWrapper:
    def test_schedulefree_wrapper(self):
        params = [make_parameter(), make_parameter((1,)), make_parameter(grad=None)]
        optimizer = ScheduleFreeWrapper(build_optimizer('adamw', params, lr=1e-3, weight_decay=1e-3))

        with pytest.raises(ValueError):
            optimizer.step()

        optimizer.eval()
        optimizer.train()

        _ = optimizer.__str__
        _ = optimizer.__getstate__()
        _ = optimizer.param_groups

        optimizer.step()

        backup_state = optimizer.state_dict()

        optimizer = ScheduleFreeWrapper(build_optimizer('adamw', params, lr=1e-3, weight_decay=1e-3))
        optimizer.zero_grad()
        optimizer.train()

        optimizer.load_state_dict(backup_state)

        optimizer.step()

        optimizer.eval()
        optimizer.train()
        optimizer.train()

        optimizer.add_param_group({'params': []})

    def test_schedulefree_wrapper_legacy_state_dict(self):
        parameter = make_parameter()
        optimizer = ScheduleFreeWrapper(build_optimizer('sgd', [parameter], lr=0.1))
        optimizer.train()
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()

        legacy_state = {'schedulefree_state': optimizer.state, 'base_optimizer': optimizer.optimizer.state_dict()}

        same_parameter = ScheduleFreeWrapper(build_optimizer('sgd', [parameter], lr=0.1))
        same_parameter.load_state_dict(legacy_state)
        torch.testing.assert_close(same_parameter.state[parameter]['z'], optimizer.state[parameter]['z'])

        different_parameter = ScheduleFreeWrapper(build_optimizer('sgd', [make_parameter()], lr=0.1))
        with pytest.raises(ValueError, match='schedule-free state does not match'):
            different_parameter.load_state_dict(legacy_state)

    def test_schedulefree_sparse_gradient(self):
        param = make_sparse_parameters()[1]

        optimizer = build_optimizer('schedulefree', [param])
        optimizer.train()

        with pytest.raises(NoSparseGradientError):
            optimizer.step(lambda: 0.1)


class TestPCGrad:
    @pytest.mark.parametrize('reduction', ['mean', 'sum'])
    def test_pc_grad_optimizers(self, reduction, environment):
        torch.manual_seed(42)

        x_data, y_data = environment

        model: nn.Module = TrainingModel(output_features=2).to(x_data.device)
        loss_fn_1: nn.Module = nn.BCEWithLogitsLoss()
        loss_fn_2: nn.Module = nn.L1Loss()

        optimizer = PCGrad(build_optimizer('adamp', model.parameters(), lr=1e-1), reduction=reduction)
        optimizer.init_group()

        init_loss, loss = np.inf, np.inf
        for _ in range(5):
            optimizer.zero_grad()

            predictions = model(x_data)
            y_pred_1, y_pred_2 = predictions[:, :1], predictions[:, 1:]
            loss1, loss2 = loss_fn_1(y_pred_1, y_data), loss_fn_2(y_pred_2, y_data)

            loss = (loss1 + loss2) / 2.0
            if init_loss == np.inf:
                init_loss = loss

            optimizer.pc_backward([loss1, loss2])
            optimizer.step()

        assert tensor_to_numpy(init_loss) > 1.25 * tensor_to_numpy(loss)

    @pytest.mark.parametrize('reduction', ['mean', 'sum'])
    def test_pcgrad_preserves_unused_parameters(self, reduction):
        shared = nn.Parameter(torch.tensor([1.0]))
        task_specific = nn.Parameter(torch.tensor([2.0]))
        unused = nn.Parameter(torch.tensor([3.0]))

        optimizer = PCGrad(
            build_optimizer('sgd', [shared, task_specific, unused], lr=0.1, weight_decay=0.2), reduction
        )

        optimizer.pc_backward([(shared - 1.0).square().sum() + task_specific.sum(), (shared - 1.0).square().sum()])

        torch.testing.assert_close(shared.grad, torch.zeros_like(shared))
        torch.testing.assert_close(task_specific.grad, torch.ones_like(task_specific))
        assert unused.grad is None

        optimizer.step()
        torch.testing.assert_close(shared, torch.tensor([0.98]))
        torch.testing.assert_close(task_specific, torch.tensor([1.86]))
        torch.testing.assert_close(unused, torch.tensor([3.0]))

    def test_pcgrad_parameters(self):
        opt = build_optimizer('adamw', [make_parameter()])

        for reduction in ('mean', 'sum'):
            PCGrad(opt, reduction=reduction)

        with pytest.raises(ValueError):
            PCGrad(opt, reduction='invalid')


class TestTRAC:
    def test_trac_optimizer(self, environment):
        x_data, y_data = environment
        model, loss_fn = build_model(device=x_data.device)

        optimizer = TRAC(build_optimizer('adamw', model.parameters(), lr=1e0))

        trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
        trainer.run(iterations=3, threshold=2.0)

    def test_trac_optimizer_erf_imag(self):
        optimizer = TRAC(build_optimizer('adamw', [make_parameter()]))
        optimizer.zero_grad()

        complex_tensor = torch.complex(torch.tensor(0.0), torch.tensor(1.0))
        optimizer.erf_imag(complex_tensor)

        assert str(optimizer).lower() == 'trac'

    @pytest.mark.parametrize('checkpoint_format', ['indexed', 'legacy'])
    def test_trac_checkpoint_resumes_with_new_parameters(self, checkpoint_format):
        parameters = [nn.Parameter(torch.tensor([1.0])), nn.Parameter(torch.tensor([-1.0, 2.0]))]
        optimizer = TRAC(
            build_optimizer('sgd', [{'params': [p], 'lr': lr} for p, lr in zip(parameters, [0.1, 0.05])], momentum=0.9)
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
        restored = TRAC(build_optimizer('sgd', [{'params': [p]} for p in restored_parameters], lr=0.1, momentum=0.9))
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

    def test_trac_rejects_missing_checkpoint_reference(self):
        parameter = nn.Parameter(torch.tensor([1.0]))
        optimizer = TRAC(build_optimizer('sgd', [parameter], lr=0.1))
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()

        state_dict = deepcopy(optimizer.state_dict())
        del state_dict['state']['trac'][0]

        with pytest.raises(ValueError, match='TRAC state does not match'):
            optimizer.load_state_dict(state_dict)


class TestOrthoGrad:
    def test_orthograd_skip_conditions(self):
        param = make_parameter(requires_grad=True)
        param.grad = None

        optimizer = OrthoGrad(build_optimizer('adamw', [param]))
        optimizer.apply_orthogonal_gradients([param])
