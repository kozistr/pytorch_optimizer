from copy import deepcopy
from unittest.mock import patch
from weakref import ref

import pytest
import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer

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
