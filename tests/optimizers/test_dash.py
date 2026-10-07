from copy import deepcopy

import pytest
import torch

from pytorch_optimizer import DASH, load_optimizer
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def reference_newton_db(matrix, scale, steps, inverse):
    identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    y, z = matrix / scale, identity.expand_as(matrix).clone()
    for _ in range(steps):
        correction = (3.0 * identity - z @ y) / 2.0
        y, z = y @ correction, correction @ z
    return z / scale.sqrt() if inverse else y * scale.sqrt()


def reference_inverse_root(matrix, root, options):
    identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    regularized = matrix + options['matrix_eps'] * identity
    if options['inverse_root_method'] == 'eigh':
        values, vectors = torch.linalg.eigh(regularized)
        values = values + options['matrix_eps'] - values.min().clamp_max(0.0)
        return vectors @ torch.diag(values.pow(-1.0 / root)) @ vectors.T
    scale = regularized.bfloat16().norm().to(matrix.dtype)
    if root == 4:
        regularized = reference_newton_db(regularized, scale, options['newton_steps'], inverse=False)
        scale = scale.sqrt()
    return reference_newton_db(regularized, scale, options['newton_steps'], inverse=True)


def reference_step(param, grad, state, step, options):
    shape = param.squeeze().shape
    one_sided = len(shape) < 2
    matrix_shape = (shape[0], param.numel() // shape[0]) if not one_sided else (param.numel(), 1)
    matrix, gradient = param.reshape(matrix_shape), grad.reshape(matrix_shape)
    update = torch.empty_like(matrix)
    beta1, beta2 = options['betas']
    grafting_beta = options['grafting_beta'] if options['grafting_beta'] is not None else beta2
    for row in range(0, matrix.shape[0], options['block_size']):
        for col in range(0, matrix.shape[1], options['block_size']):
            block = gradient[row : row + options['block_size'], col : col + options['block_size']]
            if options.get('maximize', False):
                block = -block
            key = (row, col)
            if key not in state:
                state[key] = {
                    'left': torch.zeros(block.shape[0], block.shape[0], dtype=block.dtype, device=block.device),
                    'right': torch.zeros(block.shape[1], block.shape[1], dtype=block.dtype, device=block.device),
                    'first': torch.zeros_like(block),
                    'second': torch.zeros_like(block),
                    'momentum': torch.zeros_like(block),
                }
            item = state[key]
            item['left'] = beta2 * item['left'] + (1.0 - beta2) * block @ block.T
            item['right'] = beta2 * item['right'] + (1.0 - beta2) * block.T @ block
            item['second'] = grafting_beta * item['second'] + (1.0 - grafting_beta) * block.square()
            item['first'] = beta1 * item['first'] + (1.0 - beta1) * block
            chosen = item['first'] if beta1 else block
            bias1 = 1.0 - beta1**step if options['correct_bias'] else 1.0
            bias2 = 1.0 - grafting_beta**step if options['correct_bias'] else 1.0
            graft = (chosen / bias1) / ((item['second'] / bias2).sqrt() + options['eps'])
            if step >= options['start_preconditioning_step']:
                if step == options['start_preconditioning_step'] or step % options['precondition_frequency'] == 0:
                    item['inverse_left'] = reference_inverse_root(item['left'], 2 if one_sided else 4, options)
                    item['inverse_right'] = reference_inverse_root(item['right'], 4, options)
                direction = item['inverse_left'] @ chosen
                if not one_sided:
                    direction = direction @ item['inverse_right']
                direction = direction * graft.norm() / (direction.norm() + 1e-16)
            else:
                direction = graft
            if options['momentum']:
                item['momentum'] = options['momentum'] * item['momentum'] + direction
                direction = (
                    direction + options['momentum'] * item['momentum'] if options['nesterov'] else item['momentum']
                )
            update[row : row + block.shape[0], col : col + block.shape[1]] = direction
    return (matrix * (1.0 - options['lr'] * options['weight_decay']) - options['lr'] * update).reshape_as(param)


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

    @pytest.mark.parametrize('steps', [1, 3])
    @pytest.mark.parametrize('inverse', [False, True])
    def test_finite_newton_iterations(self, steps, inverse, device):
        matrices = torch.tensor([[[1.0, 0.2], [0.2, 2.0]]], dtype=torch.float64, device=device)
        scale = torch.tensor([[[3.0]]], dtype=torch.float64, device=device)
        expected = reference_newton_db(matrices, scale, steps, inverse)
        torch.testing.assert_close(DASH.newton_db(matrices, scale, steps, inverse), expected)

    @pytest.mark.parametrize('vectors', [1, 16])
    def test_power_iteration(self, vectors, device):
        optimizer = build_optimizer('dash', [make_parameter(device=device)], power_iteration_vectors=vectors)
        matrices = torch.diag_embed(torch.tensor([[1.0, 4.0, 2.0], [3.0, 1.0, 2.0]], device=device))
        original = matrices.clone()
        with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
            torch.manual_seed(42)
            scale = optimizer.matrix_scale(matrices, optimizer.param_groups[0])
        torch.testing.assert_close(scale.flatten(), torch.tensor([8.0, 6.0], device=device), atol=0.05, rtol=0.01)
        torch.testing.assert_close(matrices, original, atol=0.0, rtol=0.0)


class TestDASHUpdates:
    @pytest.mark.parametrize('shape', [(), (1, 5, 1), (5, 7), (2, 2, 2)])
    @pytest.mark.parametrize(
        'options',
        [
            {'inverse_root_method': 'eigh'},
            {'inverse_root_method': 'newton_db', 'matrix_scaling': 'frobenius', 'betas': (0.0, 0.8)},
            {
                'inverse_root_method': 'eigh',
                'momentum': 0.6,
                'nesterov': False,
                'grafting_beta': 0.7,
                'correct_bias': False,
                'maximize': True,
            },
            {
                'inverse_root_method': 'newton_db',
                'matrix_scaling': 'frobenius',
                'momentum': 0.6,
                'start_preconditioning_step': 3,
            },
        ],
    )
    def test_matches_unbatched_reference(self, shape, options, device):
        param = make_parameter(shape, dtype=torch.float64, device=device)
        optimizer = build_optimizer(
            'dash',
            [param],
            lr=0.02,
            block_size=2,
            precondition_frequency=4,
            weight_decay=0.1,
            matrix_eps=0.01,
            **options,
        )
        expected, reference_state = param.detach().clone().fill_(1.0), {}
        with torch.no_grad():
            param.fill_(1.0)
        reference_options = {**optimizer.param_groups[0], 'maximize': options.get('maximize', False)}
        for step in range(1, 9):
            gradient = (torch.arange(param.numel(), dtype=param.dtype, device=device).reshape(shape) + step).sin()
            param.grad = gradient.clone()
            expected = reference_step(expected, gradient, reference_state, step, reference_options)
            optimizer.step()
            torch.testing.assert_close(param, expected, atol=1e-8, rtol=1e-8)
            torch.testing.assert_close(param.grad, gradient, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize('method', ['newton_db', 'eigh'])
    @pytest.mark.parametrize('eps', [0.0, 1e-8])
    def test_zero_gradient(self, method, eps, device):
        param = make_parameter((3, 5), device=device)
        optimizer = build_optimizer('dash', [param], block_size=2, inverse_root_method=method, eps=eps)
        optimizer.step()
        torch.testing.assert_close(param, torch.zeros_like(param), atol=0.0, rtol=0.0)

    def test_noncontiguous_parameter_and_gradient(self, device):
        param = torch.arange(15, dtype=torch.float64, device=device).reshape(3, 5).T.requires_grad_()
        param.grad = torch.linspace(-1.0, 1.0, 15, dtype=param.dtype, device=device).reshape(3, 5).T
        optimizer = build_optimizer('dash', [param], block_size=2, inverse_root_method='eigh', matrix_eps=0.01)
        expected = reference_step(param.detach(), param.grad, {}, 1, optimizer.param_groups[0])
        optimizer.step()
        torch.testing.assert_close(param, expected)
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


@pytest.mark.parametrize(
    ('name', 'value'),
    [
        ('block_size', 0),
        ('precondition_frequency', 0),
        ('start_preconditioning_step', 0),
        ('newton_steps', 0),
        ('power_iteration_steps', 0),
        ('power_iteration_vectors', 0),
        ('grafting_beta', 1.0),
        ('momentum', 1.0),
        ('matrix_eps', 0.0),
        ('inverse_root_method', 'invalid'),
        ('matrix_scaling', 'invalid'),
    ],
)
def test_invalid_options(name, value):
    with pytest.raises(ValueError):
        load_optimizer('dash')([make_parameter()], **{name: value})
