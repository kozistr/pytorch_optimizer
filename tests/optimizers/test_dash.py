import math
from copy import deepcopy

import pytest
import torch

from pytorch_optimizer import DASH
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


class TestDASHRoots:
    @pytest.mark.parametrize('root', [2, 4])
    @pytest.mark.parametrize(
        ('method', 'scaling'), [('eigh', 'power'), ('newton_db', 'power'), ('newton_db', 'frobenius')]
    )
    def test_inverse_root(self, root, method, scaling, device):
        optimizer = build_optimizer(
            'dash',
            [make_parameter(device=device)],
            inverse_root_method=method,
            matrix_scaling=scaling,
            newton_steps=20,
            matrix_eps=1e-4,
        )
        matrices = torch.tensor(
            [[[4.0, 1.0], [1.0, 2.0]], [[1.0, 0.0], [0.0, 3.0]]], dtype=torch.float64, device=device
        )
        original = matrices.clone()

        result = optimizer.inverse_root(matrices, root, optimizer.param_groups[0])

        eps = (2 if method == 'eigh' else 1) * 1e-4
        values, vectors = torch.linalg.eigh(matrices + eps * torch.eye(2, device=device))
        expected = (vectors * values.pow(-1.0 / root).unsqueeze(-2)) @ vectors.transpose(-2, -1)

        torch.testing.assert_close(result, expected, atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(matrices, original, atol=0.0, rtol=0.0)


class TestDASHUpdates:
    def test_partition_edge_blocks(self, device):
        gradient = torch.arange(15, device=device).reshape(3, 5)
        expected = [
            ((0, 0, 2, 4), [[[0, 1], [5, 6]], [[2, 3], [7, 8]]]),
            ((0, 4, 2, 1), [[[4], [9]]]),
            ((2, 0, 1, 4), [[[10, 11]], [[12, 13]]]),
            ((2, 4, 1, 1), [[[14]]]),
        ]

        partitions = list(DASH.partition(gradient, 2))

        assert len(partitions) == len(expected)

        for (bounds, block), (expected_bounds, values) in zip(partitions, expected):
            assert bounds == expected_bounds
            torch.testing.assert_close(block, torch.tensor(values, device=device))

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
        torch.testing.assert_close(param.grad, gradient, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize('shape', [(), (1, 5, 1), (5, 7), (2, 2, 2)])
    def test_grafting_warmup(self, shape, device):
        param = make_parameter(shape, dtype=torch.float64, device=device)
        expected = param.detach().clone().requires_grad_()
        optimizer = build_optimizer(
            'dash', [param], lr=0.1, grafting_beta=0.7, weight_decay=0.1, block_size=2, start_preconditioning_step=3
        )
        adamw = torch.optim.AdamW([expected], lr=0.1, betas=(0.9, 0.7), weight_decay=0.1)

        for step in range(2):
            gradient = torch.arange(param.numel(), dtype=param.dtype, device=device).reshape(shape) + step + 1
            param.grad = gradient.clone()
            expected.grad = gradient.clone()

            optimizer.step()
            adamw.step()

            torch.testing.assert_close(param, expected)
            torch.testing.assert_close(param.grad, gradient, atol=0.0, rtol=0.0)

    def test_uncorrected_grafting(self, device):
        param = make_parameter((2, 2), dtype=torch.float64, device=device)
        gradient = torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=param.dtype, device=device)
        optimizer = build_optimizer(
            'dash', [param], lr=0.1, grafting_beta=0.7, correct_bias=False, maximize=True, start_preconditioning_step=3
        )
        first_update = 0.1 * gradient / (0.3**0.5 * gradient.abs() + 1e-8)
        second_update = 0.19 * gradient / (0.51**0.5 * gradient.abs() + 1e-8)

        for _ in range(2):
            param.grad = gradient.clone()
            optimizer.step()

        torch.testing.assert_close(param, 0.1 * (first_update + second_update))
        torch.testing.assert_close(param.grad, gradient, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize('nesterov', [False, True])
    def test_update_momentum(self, nesterov, device):
        param = make_parameter((2, 2), dtype=torch.float64, device=device, grad=1.0)
        expected = param.detach().clone().requires_grad_()
        optimizer = build_optimizer(
            'dash',
            [param],
            lr=0.1,
            momentum=0.6,
            nesterov=nesterov,
            precondition_frequency=4,
            start_preconditioning_step=3,
            matrix_eps=0.01,
            eps=0.0,
        )
        sgd = torch.optim.SGD([expected], lr=0.1, momentum=0.6, nesterov=nesterov)

        for _ in range(5):
            expected.grad = torch.ones_like(expected)

            optimizer.step()
            sgd.step()

            torch.testing.assert_close(param, expected)

    def test_precondition_frequency(self, device):
        param = make_parameter((2, 2), device=device, grad=1.0)
        optimizer = build_optimizer('dash', [param], inverse_root_method='eigh', precondition_frequency=3)

        optimizer.step()
        cached = optimizer.state[param]['blocks'][0]['inverse_roots'][0].clone()

        param.grad.mul_(2.0)
        optimizer.step()

        torch.testing.assert_close(optimizer.state[param]['blocks'][0]['inverse_roots'][0], cached, atol=0.0, rtol=0.0)

        optimizer.step()

        assert not torch.equal(optimizer.state[param]['blocks'][0]['inverse_roots'][0], cached)

    @pytest.mark.parametrize('method', ['newton_db', 'eigh'])
    @pytest.mark.parametrize('eps', [0.0, 1e-8])
    def test_zero_gradient(self, method, eps, device):
        param = make_parameter((3, 5), device=device)
        optimizer = build_optimizer('dash', [param], block_size=2, inverse_root_method=method, eps=eps)

        optimizer.step()

        torch.testing.assert_close(param, torch.zeros_like(param), atol=0.0, rtol=0.0)

    @pytest.mark.parametrize('shape', [(3, 5), (2, 3, 4)])
    def test_noncontiguous_parameter_and_gradient(self, shape, device):
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
        assert not param.is_contiguous()

    def test_parameter_groups_and_missing_gradients(self, device):
        first, second = make_parameter((), grad=1.0, device=device), make_parameter((), grad=None, device=device)
        optimizer = build_optimizer('dash', [{'params': [first]}, {'params': [second], 'lr': 0.1}], lr=0.2)

        optimizer.step()

        assert second not in optimizer.state

        first.grad, second.grad = None, torch.ones_like(second)
        optimizer.step()

        assert optimizer.state[first]['step'] == optimizer.state[second]['step'] == 1
        torch.testing.assert_close(first, torch.tensor(-0.2, device=device))
        torch.testing.assert_close(second, torch.tensor(-0.1, device=device))

    def test_state_storage(self, device):
        param = make_parameter((4, 4), grad=1.0, device=device)
        optimizer = build_optimizer('dash', [param], block_size=2, betas=(0.0, 0.9))

        optimizer.step()
        blocks = optimizer.state[param]['blocks']

        assert len(blocks) == 1
        assert set(blocks[0]) == {'exp_avg_sq', 'statistics', 'inverse_roots'}
        assert len(blocks[0]['statistics']) == len(blocks[0]['inverse_roots']) == 1
        assert blocks[0]['statistics'][0].shape == (8, 2, 2)

    @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float64])
    def test_checkpoint_preserves_precision(self, dtype, device):
        param = make_parameter((3, 5), dtype=dtype, grad=0.1, device=device)
        options = {'block_size': 2, 'momentum': 0.8, 'precondition_frequency': 2}
        optimizer = build_optimizer('dash', [param], **options)

        optimizer.step()

        restored_param = param.detach().clone().requires_grad_()
        restored = build_optimizer('dash', [restored_param], **options)
        restored.load_state_dict(deepcopy(optimizer.state_dict()))
        expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32

        for block in restored.state[restored_param]['blocks']:
            assert block['exp_avg_sq'].dtype == block['exp_avg'].dtype == block['momentum'].dtype == expected_dtype
            assert all(t.dtype == expected_dtype for t in block['statistics'] + block['inverse_roots'])

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
