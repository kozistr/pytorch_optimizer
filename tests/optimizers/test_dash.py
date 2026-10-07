import math
from copy import deepcopy

import pytest
import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


class TestDASHUpdates:
    @pytest.mark.parametrize('method', ['newton_db', 'eigh'])
    def test_blockwise_grafting(self, method, device):
        param = make_parameter((2, 4), dtype=torch.float64, device=device)
        gradient = torch.tensor([[1.0, 0.0, 3.0, 0.0], [0.0, 2.0, 0.0, 4.0]], dtype=param.dtype, device=device)
        param.grad = gradient.clone()

        with torch.no_grad():
            param.fill_(1.0)

        optimizer = build_optimizer(
            'dash',
            [param],
            lr=0.1,
            betas=(0.0, 0.0),
            weight_decay=0.1,
            block_size=2,
            eps=0.0,
            matrix_eps=0.25,
            inverse_root_method=method,
            matrix_scaling='frobenius',
            newton_steps=20,
        )
        damping = 0.5 if method == 'eigh' else 0.25
        direction = gradient / (gradient.square() + damping).sqrt()

        for block in direction.split(2, dim=1):
            block.mul_(2.0**0.5 / block.norm())

        optimizer.step()

        torch.testing.assert_close(param, 0.99 - 0.1 * direction)

    def test_update_momentum(self, device):
        param = make_parameter((2, 2), dtype=torch.float64, device=device, grad=1.0)
        expected = param.detach().clone().requires_grad_()
        optimizer = build_optimizer(
            'dash',
            [param],
            lr=0.1,
            momentum=0.6,
            precondition_frequency=4,
            start_preconditioning_step=3,
            matrix_eps=0.01,
            eps=0.0,
        )
        sgd = torch.optim.SGD([expected], lr=0.1, momentum=0.6, nesterov=True)

        for _ in range(5):
            expected.grad = torch.ones_like(expected)

            optimizer.step()
            sgd.step()

            torch.testing.assert_close(param, expected)

    def test_zero_gradient(self, device):
        param = make_parameter((3, 5), device=device)
        optimizer = build_optimizer('dash', [param], block_size=2, eps=0.0)

        optimizer.step()

        torch.testing.assert_close(param, torch.zeros_like(param), atol=0.0, rtol=0.0)

    def test_noncontiguous_parameter_and_gradient(self, device):
        shape = (2, 3, 4)
        size = math.prod(shape)
        param = torch.arange(size, dtype=torch.float64, device=device).reshape(shape).transpose(0, 1).requires_grad_()
        param.grad = torch.linspace(-1.0, 1.0, size, dtype=param.dtype, device=device).reshape(shape).transpose(0, 1)
        contiguous = param.detach().contiguous().requires_grad_()
        contiguous.grad = param.grad.contiguous()
        optimizer = build_optimizer(
            'dash', [param, contiguous], block_size=2, inverse_root_method='eigh', matrix_eps=0.01
        )

        optimizer.step()

        torch.testing.assert_close(param, contiguous)

    def test_parameter_groups_and_missing_gradients(self, device):
        first = make_parameter((), grad=1.0, device=device)
        second = make_parameter((), grad=None, device=device)
        third = make_parameter((), grad=1.0, device=device)
        optimizer = build_optimizer('dash', [{'params': [first, second]}, {'params': [third], 'lr': 0.1}], lr=0.2)

        optimizer.step()

        assert second not in optimizer.state

        second.grad, third.grad = torch.ones_like(second), None
        optimizer.step()

        assert all(group['step'] == 2 for group in optimizer.param_groups)
        assert all('step' not in optimizer.state[param] for param in (first, second, third))
        torch.testing.assert_close(first, torch.tensor(-0.4, device=device))
        torch.testing.assert_close(second, torch.tensor(-0.2 * math.sqrt(1.95) / 1.9, device=device))
        torch.testing.assert_close(third, torch.tensor(-0.1, device=device))

    def test_checkpoint_preserves_precision(self, device):
        param = make_parameter((3, 5), dtype=torch.float16, grad=0.1, device=device)
        options = {'block_size': 2, 'momentum': 0.8, 'precondition_frequency': 2}
        optimizer = build_optimizer('dash', [param], **options)

        optimizer.step()

        restored_param = param.detach().clone().requires_grad_()
        restored = build_optimizer('dash', [restored_param], **options)
        restored.load_state_dict(deepcopy(optimizer.state_dict()))

        assert optimizer.param_groups[0]['step'] == restored.param_groups[0]['step'] == 1

        for block in restored.state[restored_param]['blocks']:
            assert block['exp_avg_sq'].dtype == block['exp_avg'].dtype == block['momentum'].dtype == torch.float32
            assert all(t.dtype == torch.float32 for t in block['statistics'] + block['inverse_roots'])

        for _ in range(3):
            param.grad = torch.full_like(param, 0.2)
            restored_param.grad = param.grad.clone()

            with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
                torch.manual_seed(42)
                optimizer.step()

            with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
                torch.manual_seed(42)
                restored.step()

            torch.testing.assert_close(param, restored_param, atol=0.0, rtol=0.0)
