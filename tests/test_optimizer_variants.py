from copy import deepcopy
from typing import Any
from unittest.mock import patch
from weakref import ref

import pytest
import torch

from tests.fixtures import build_model, make_parameter
from tests.optimizer_cases import FOREACH_OPTIMIZERS, GRADIENT_OPTIONS, MAXIMIZE_OPTIMIZERS, SKIP_CAPABILITY_PROBE
from tests.utils import Trainer, build_optimizer, build_optimizer_parameters, ids

ADANORM_SUPPORTED_OPTIMIZERS: list[tuple[str, dict[str, Any], int]] = [
    ('adabelief', {'lr': 5e-1, 'weight_decay': 1e-3}, 10),
    ('adamp', {'lr': 5e-1, 'weight_decay': 1e-3}, 5),
    ('adams', {'lr': 7.5e-1, 'weight_decay': 1e-3}, 5),
    ('adapnm', {'lr': 5e-1, 'weight_decay': 1e-3}, 5),
    ('diffgrad', {'lr': 5e-1, 'weight_decay': 1e-3}, 5),
    ('lamb', {'lr': 5e-2}, 10),
    ('radam', {'lr': 5e0, 'weight_decay': 1e-3}, 10),
    ('ranger', {'lr': 2.5e0, 'weight_decay': 1e-3}, 75),
    ('adan', {'lr': 5e-1, 'weight_decay': 1e-3}, 5),
    ('lion', {'lr': 5e-1, 'weight_decay': 1e-3}, 5),
    ('adamax', {'lr': 5e-1, 'weight_decay': 1e-3}, 5),
    ('aida', {'lr': 1e0, 'weight_decay': 1e-3}, 5),
]

ADAMD_SUPPORTED_OPTIMIZERS: list[tuple[str, dict[str, Any], int]] = [
    ('adabelief', {'lr': 1e1, 'weight_decay': 1e-3}, 5),
    ('adabound', {'lr': 1e0, 'gamma': 0.1, 'weight_decay': 1e-3}, 35),
    ('adamp', {'lr': 1e0, 'weight_decay': 1e-3}, 5),
    ('adams', {'lr': 2e0, 'weight_decay': 1e-3}, 5),
    ('diffgrad', {'lr': 2e0, 'weight_decay': 1e-3, 'rectify': True}, 15),
    ('diffgrad', {'lr': 2e0, 'weight_decay': 1e-3}, 5),
    ('lamb', {'lr': 1e0, 'weight_decay': 1e-3, 'rectify': True}, 30),
    ('radam', {'lr': 1e0, 'weight_decay': 1e-3}, 25),
    ('ranger', {'lr': 5e0, 'weight_decay': 1e-3}, 50),
    ('ranger21', {'lr': 5e-1, 'weight_decay': 1e-3, 'num_iterations': 125, 'disable_lr_scheduler': True}, 125),
    ('adapnm', {'lr': 1e0, 'weight_decay': 1e-3}, 10),
    ('novograd', {'lr': 1e0, 'weight_decay': 1e-3}, 5),
    ('adanorm', {'lr': 1e0, 'weight_decay': 1e-3}, 5),
    ('yogi', {'lr': 1e0, 'weight_decay': 1e-3}, 5),
    ('adamod', {'lr': 1e2, 'weight_decay': 1e-3}, 20),
    ('adamax', {'lr': 1e0, 'weight_decay': 1e-3}, 5),
    ('avagrad', {'lr': 1e1, 'weight_decay': 1e-3}, 5),
    ('adahessian', {'lr': 5e0, 'weight_decay': 1e-3}, 5),
    ('aida', {'lr': 1e1, 'weight_decay': 1e-3, 'rectify': True}, 10),
]

COPT_SUPPORTED_OPTIMIZERS: list[tuple[str, dict[str, Any], int]] = [
    ('adafactor', {'lr': 1e0, 'weight_decay': 1e-3, 'scale_parameter': False, 'relative_step': False}, 5),
    ('lion', {'lr': 5e-1, 'weight_decay': 1e-3}, 5),
    ('ademamix', {'lr': 1e0}, 2),
    ('laprop', {'lr': 1e0}, 2),
    ('adamp', {'lr': 1e0}, 2),
    ('adopt', {'lr': 1e1}, 3),
    ('adashift', {'lr': 1e0, 'keep_num': 1}, 5),
    ('mars', {'lr': 5e-1, 'lr_1d': 5e-1, 'weight_decay': 1e-3}, 3),
    ('mars', {'lr': 5e-1, 'lr_1d': 5e-1, 'weight_decay': 1e-3, 'optimize_1d': True}, 3),
    (
        'muon',
        {
            'lr': 5e-1,
            'weight_decay': 1e-3,
            'use_adjusted_lr': True,
            'adamw_lr': 5e-1,
            'adamw_betas': (0.9, 0.98),
            'adamw_wd': 1e-2,
        },
        7,
    ),
    (
        'adago',
        {'lr': 5e-1, 'use_adjusted_lr': True, 'adamw_lr': 5e-1, 'adamw_betas': (0.9, 0.98), 'adamw_wd': 1e-2},
        7,
    ),
]

STABLE_ADAMW_SUPPORTED_OPTIMIZERS: list[tuple[str, dict[str, Any], int]] = [
    ('adopt', {'lr': 1e0, 'weight_decay': 1e-3, 'stable_adamw': True}, 5),
    ('ademamix', {'lr': 1e0, 'weight_decay': 1e-3, 'stable_adamw': True}, 10),
]


class TestMaximize:
    @pytest.mark.parametrize(
        ('optimizer_name', 'foreach'),
        [
            (name, foreach)
            for name in sorted(MAXIMIZE_OPTIMIZERS)
            if name not in SKIP_CAPABILITY_PROBE
            for foreach in ([False, True] if name in FOREACH_OPTIMIZERS else [False])
        ],
    )
    def test_maximize(self, optimizer_name, foreach):
        params = [torch.full((2, 2), 2.0, requires_grad=True) for _ in range(2)]
        params[0].grad = torch.full_like(params[0], 0.5)
        params[1].grad = torch.full_like(params[1], -0.5)

        options = GRADIENT_OPTIONS.get(optimizer_name, {}).copy()
        if optimizer_name in FOREACH_OPTIMIZERS:
            options['foreach'] = foreach
        if optimizer_name == 'kron':
            options.update(pre_conditioner_update_probability=1.0, balance_prob=0.0)
        ascent = build_optimizer(optimizer_name, [params[0]], maximize=True, **options)
        descent = build_optimizer(optimizer_name, [params[1]], maximize=False, **options)

        if optimizer_name.startswith('schedulefree'):
            ascent.train()
            descent.train()

        for optimizer in (ascent, descent):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                options = {'hessian': [torch.ones_like(optimizer.param_groups[0]['params'][0])]} if (
                    optimizer_name in ('adahessian', 'sophiah')
                ) else {}
                optimizer.step(lambda: 0.1, **options)

        torch.testing.assert_close(params[0], params[1])


class TestAdaNorm:
    @pytest.mark.parametrize('optimizer_config', ADANORM_SUPPORTED_OPTIMIZERS, ids=ids)
    def test_adanorm_optimizer(self, optimizer_config, environment):
        x_data, y_data = environment
        model, loss_fn = build_model(device=x_data.device)

        optimizer_name, config, num_iterations = optimizer_config
        optimizer = build_optimizer(optimizer_name, model.parameters(), **config, adanorm=True)

        trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
        trainer.run(iterations=num_iterations, threshold=1.75)

    @pytest.mark.parametrize('optimizer_config', ADANORM_SUPPORTED_OPTIMIZERS, ids=ids)
    def test_adanorm_variant(self, optimizer_config):
        param = make_parameter()
        param.grad = torch.ones(1, 1)

        optimizer_name, _ = optimizer_config[:2]

        optimizer = build_optimizer(optimizer_name, [param], adanorm=True)
        optimizer.step()

        param.grad = torch.zeros(1, 1)
        optimizer.step()


class TestAdamDebias:
    @pytest.mark.parametrize('optimizer_config', ADAMD_SUPPORTED_OPTIMIZERS, ids=ids)
    def test_adamd_variant(self, optimizer_config, environment):
        x_data, y_data = environment
        model, loss_fn = build_model(device=x_data.device)

        optimizer_name, config, num_iterations = optimizer_config
        optimizer = build_optimizer(optimizer_name, model.parameters(), **config, adam_debias=True)

        create_graph = optimizer_name in ('adahessian',)
        trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
        trainer.run(iterations=num_iterations, create_graph=create_graph, threshold=2.0)


class TestCautious:
    @pytest.mark.parametrize('optimizer_config', COPT_SUPPORTED_OPTIMIZERS, ids=ids)
    def test_cautious_variant(self, optimizer_config, environment):
        x_data, y_data = environment
        model, loss_fn = build_model(device=x_data.device)

        optimizer_name, config, num_iterations = optimizer_config
        parameters, config = build_optimizer_parameters(model.parameters(), optimizer_name, config)
        optimizer = build_optimizer(optimizer_name, parameters, **config, cautious=True)

        trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
        trainer.run(iterations=num_iterations, threshold=1.5)


class TestStableAdamW:
    @pytest.mark.parametrize('optimizer_config', STABLE_ADAMW_SUPPORTED_OPTIMIZERS, ids=ids)
    def test_stable_adamw_variant(self, optimizer_config, environment):
        x_data, y_data = environment
        model, loss_fn = build_model(device=x_data.device)

        optimizer_name, config, num_iterations = optimizer_config
        optimizer = build_optimizer(optimizer_name, model.parameters(), **config)

        trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
        trainer.run(iterations=num_iterations, threshold=1.5)


FOREACH_CASES = [
    ('adabound', {}),
    ('adabound', {'ams_bound': True}),
    ('adamax', {}),
    ('adamax', {'adam_debias': True}),
    ('adamod', {}),
    ('adamod', {'adam_debias': True}),
    ('diffgrad', {}),
    ('diffgrad', {'ams_bound': True}),
    ('diffgrad', {'rectify': True}),
    ('diffgrad', {'rectify': True, 'degenerated_to_sgd': False, 'ams_bound': True}),
    ('padam', {}),
    ('padam', {'partial': 0.5}),
    ('radam', {}),
    ('radam', {'degenerated_to_sgd': True}),
    ('radam', {'adam_debias': True}),
    ('yogi', {}),
    ('yogi', {'adam_debias': True}),
]
FOREACH_NAMES = sorted({name for name, _ in FOREACH_CASES})


def assert_optimizer_matches(optimizer, reference):
    for group, reference_group in zip(optimizer.param_groups, reference.param_groups):
        assert group['step'] == reference_group['step']

        for param, reference_param in zip(group['params'], reference_group['params']):
            tolerances = (
                {'atol': 1e-5, 'rtol': 4 * torch.finfo(param.dtype).eps} if param.dtype == torch.float16 else {}
            )

            if param.dtype == torch.bfloat16:
                torch.testing.assert_close(param, reference_param, atol=torch.finfo(param.dtype).eps, rtol=0.016)
            elif param.dtype == torch.float16:
                torch.testing.assert_close(
                    param, reference_param, **{**tolerances, 'atol': torch.finfo(param.dtype).eps}
                )
            else:
                torch.testing.assert_close(param, reference_param)

            torch.testing.assert_close(param.grad, reference_param.grad, **tolerances)
            torch.testing.assert_close(
                optimizer.state.get(param, {}), reference.state.get(reference_param, {}), **tolerances
            )


class TestForeach:
    @pytest.mark.parametrize(('optimizer_name', 'options'), FOREACH_CASES)
    @pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float16])
    @pytest.mark.parametrize('foreach', [True, None])
    @pytest.mark.parametrize(('weight_decouple', 'fixed_decay'), [(False, False), (True, False), (True, True)])
    def test_matches_per_parameter(
        self, optimizer_name, options, dtype, foreach, weight_decouple, fixed_decay, device
    ):
        params = [
            make_parameter((2, 3), dtype=dtype, device=device, grad=None),
            make_parameter(
                (2,), dtype=torch.float32 if dtype != torch.float32 else torch.bfloat16, device=device, grad=None
            ),
            make_parameter((), dtype=dtype, device=device, grad=None),
            make_parameter((2, 3), dtype=dtype, device=device, grad=None).t().detach().requires_grad_(),
            make_parameter(device=device, grad=None),
        ]

        with torch.no_grad():
            for param in params:
                param.fill_(0.5)

        reference_params = [param.detach().clone().requires_grad_() for param in params]

        config = {
            'lr': 0.01,
            'betas': (0.6, 0.8, 0.7) if optimizer_name == 'adamod' else (0.6, 0.8),
            'weight_decay': 0.1,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'maximize': True,
            **({'eps': 1e-4} if dtype == torch.float16 else {}),
            **options,
        }
        optimizer = build_optimizer(optimizer_name, params, foreach=foreach, **config)
        reference = build_optimizer(optimizer_name, reference_params, foreach=False, **config)

        with patch.object(optimizer, '_step_foreach', wraps=optimizer._step_foreach) as batched_step:
            for step in range(12):
                for index, (param, reference_param) in enumerate(zip(params, reference_params)):
                    if step in (0, 11) or index == 4 or (index == 1 and step % 2) or (index == 3 and step < 4):
                        param.grad = reference_param.grad = None
                    else:
                        grad = torch.arange(param.numel(), dtype=param.dtype, device=param.device).reshape(param.shape)
                        grad = grad.mul(0.1).add(0.25 if step % 2 else -0.35)
                        param.grad, reference_param.grad = grad, grad.clone()

                optimizer.step()
                reference.step()

                assert_optimizer_matches(optimizer, reference)

            assert batched_step.call_count > 0

            for call in batched_step.call_args_list:
                assert len({param.dtype for param in call.args[1]}) == 1

        assert params[-1] not in optimizer.state

    @pytest.mark.parametrize('optimizer_name', FOREACH_NAMES)
    def test_group_override(self, optimizer_name):
        params = [make_parameter(grad=0.2) for _ in range(3)]
        reference_params = [make_parameter(grad=0.2) for _ in range(3)]
        groups = [{'params': [param], 'foreach': option} for param, option in zip(params, [False, True, None])]
        optimizer = build_optimizer(optimizer_name, groups, foreach=False)
        reference = build_optimizer(optimizer_name, reference_params, foreach=False)

        with patch.object(optimizer, '_step_foreach', wraps=optimizer._step_foreach) as batched_step:
            optimizer.step()
            reference.step()

            assert batched_step.call_count == 2
            assert [call.args[0]['foreach'] for call in batched_step.call_args_list] == [True, None]

        for param, reference_param in zip(params, reference_params):
            torch.testing.assert_close(param, reference_param)
            torch.testing.assert_close(optimizer.state[param], reference.state[reference_param])

    @pytest.mark.parametrize('optimizer_name', FOREACH_NAMES)
    def test_complex_fallback(self, optimizer_name):
        params = [make_parameter((2,), dtype=dtype, grad=0.2) for dtype in (torch.float32, torch.complex64)]
        reference_params = [make_parameter((2,), dtype=dtype, grad=0.2) for dtype in (torch.float32, torch.complex64)]
        optimizer = build_optimizer(optimizer_name, params, foreach=True)
        reference = build_optimizer(optimizer_name, reference_params, foreach=False)

        with patch.object(optimizer, '_step_foreach', wraps=optimizer._step_foreach) as batched_step:
            optimizer.step()
            reference.step()

            batched_step.assert_not_called()

        assert_optimizer_matches(optimizer, reference)

    @pytest.mark.parametrize('optimizer_name', ['adamax', 'diffgrad', 'radam'])
    def test_adanorm_fallback(self, optimizer_name):
        param, reference_param = make_parameter(grad=0.2), make_parameter(grad=0.2)
        optimizer = build_optimizer(optimizer_name, [param], foreach=True, adanorm=True)
        reference = build_optimizer(optimizer_name, [reference_param], foreach=False, adanorm=True)

        with patch.object(optimizer, '_step_foreach', wraps=optimizer._step_foreach) as batched_step:
            for grad in (0.2, 0.01):
                param.grad.fill_(grad)
                reference_param.grad.fill_(grad)

                optimizer.step()
                reference.step()

            batched_step.assert_not_called()

        assert_optimizer_matches(optimizer, reference)

    @pytest.mark.parametrize('optimizer_name', FOREACH_NAMES)
    @pytest.mark.parametrize('foreach', [False, True])
    def test_checkpoint_switch(self, optimizer_name, foreach):
        params = [make_parameter((2,), grad=0.2), make_parameter(grad=None)]
        optimizer = build_optimizer(optimizer_name, params, foreach=foreach)

        for _ in range(6):
            optimizer.step()

        restored_params = [param.detach().clone().requires_grad_() for param in params]
        restored = build_optimizer(optimizer_name, restored_params, foreach=not foreach)
        restored.load_state_dict(deepcopy(optimizer.state_dict()))
        restored.param_groups[0]['foreach'] = not foreach

        for step in range(3):
            for param, restored_param in zip(params, restored_params):
                param.grad = torch.full_like(param, 0.1 if step % 2 else -0.2)
                restored_param.grad = param.grad.clone()

            optimizer.step()
            restored.step()

            assert_optimizer_matches(restored, optimizer)

    def test_yogi_releases_scratch_before_denominator(self, device, monkeypatch):
        param = make_parameter((2,), grad=0.2, device=device)
        reference_param = make_parameter((2,), grad=0.2, device=device)
        optimizer = build_optimizer('yogi', [param], foreach=True)
        reference = build_optimizer('yogi', [reference_param], foreach=False)

        scratch_refs = []
        original_sqrt = torch._foreach_sqrt

        def track_scratch(operation):
            def tracked(*args, **kwargs):
                result = operation(*args, **kwargs)
                scratch_refs.extend(ref(tensor) for tensor in result)

                return result

            return tracked

        def check_scratch_released(tensors):
            assert len(scratch_refs) == 2
            assert all(tensor_ref() is None for tensor_ref in scratch_refs)

            return original_sqrt(tensors)

        monkeypatch.setattr(torch, '_foreach_mul', track_scratch(torch._foreach_mul))
        monkeypatch.setattr(torch, '_foreach_sub', track_scratch(torch._foreach_sub))

        with patch('torch._foreach_sqrt', side_effect=check_scratch_released) as square_root:
            optimizer.step()

            square_root.assert_called_once()

        reference.step()

        assert_optimizer_matches(optimizer, reference)
