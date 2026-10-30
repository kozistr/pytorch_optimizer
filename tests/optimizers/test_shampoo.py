from copy import deepcopy
from functools import reduce
from importlib import import_module

import pytest
import torch

from pytorch_optimizer.optimizer.shampoo.utils import BlockPartitioner, PreConditioner
from pytorch_optimizer.optimizer.utils.precision import to_real
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


@pytest.mark.parametrize('pre_conditioner_type', [0, 1, 2])
def test_scalable_shampoo_pre_conditioner_with_svd(pre_conditioner_type):
    params = [make_parameter(shape, grad=1.0) for shape in ((8, 2), (4, 8), (1, 4))]
    reference_params = [param.detach().clone().requires_grad_() for param in params]
    reference = torch.optim.SGD(reference_params, lr=0.1, momentum=0.9, nesterov=True, weight_decay=0.01)
    optimizer = build_optimizer(
        'scalableshampoo',
        params,
        lr=0.1,
        weight_decay=0.01,
        block_size=4,
        start_preconditioning_step=1,
        preconditioning_compute_steps=1,
        pre_conditioner_type=pre_conditioner_type,
        matrix_eps=1e-2,
        use_svd=True,
    )
    for _ in range(2):
        for param in reference_params:
            param.grad = torch.ones_like(param)
        reference.step()
        optimizer.step()

        for param, reference_param in zip(params, reference_params):
            torch.testing.assert_close(param, reference_param)
            torch.testing.assert_close(
                optimizer.state[param]['momentum'], reference.state[reference_param]['momentum_buffer']
            )
            torch.testing.assert_close(param.grad, torch.ones_like(param))


class TestShampooUtils:
    def test_legacy_imports(self):
        utils = import_module('pytorch_optimizer.optimizer.utils')
        shampoo_utils = import_module('pytorch_optimizer.optimizer.shampoo.utils')
        for module_name in ('graft', 'partition', 'preconditioner'):
            module = import_module(f'pytorch_optimizer.optimizer.utils.{module_name}')
            for name in module.__all__:
                assert getattr(module, name) is getattr(utils, name) is getattr(shampoo_utils, name)
        with pytest.raises(AttributeError):
            _ = utils.unknown_shampoo_helper

    def test_to_real(self):
        complex_tensor = torch.tensor(1.0j + 2.0, dtype=torch.complex64)
        assert to_real(complex_tensor) == 2.0

        real_tensor = torch.tensor(1.0, dtype=torch.float32)
        assert to_real(real_tensor) == 1.0

    def test_block_partitioner(self):
        var = torch.zeros((2, 2))
        target_var = torch.zeros((1, 1))

        partitioner = BlockPartitioner(var, block_size=2, rank=2, pre_conditioner_type=0)
        with pytest.raises(ValueError):
            partitioner.partition(target_var)

    def test_pre_conditioner(self):
        var = torch.zeros((16, 4))

        pre_conditioner = PreConditioner(var, 0.9, 0, 4, 1, 64, True, 0)
        pre_conditioner.add_statistics(torch.zeros_like(var))
        statistics = [statistic.clone() for statistic in pre_conditioner.statistics]
        pre_conditioner.compute_pre_conditioners()

        for statistic, original in zip(pre_conditioner.statistics, statistics):
            torch.testing.assert_close(statistic, original, rtol=0.0, atol=0.0)

    def test_invalid_pre_conditioner_type(self):
        with pytest.raises(ValueError):
            PreConditioner(torch.zeros((4, 4, 32)), 0.9, 0, 128, 1, 8192, True, pre_conditioner_type=3)


class TestKLShampoo:
    @pytest.mark.parametrize('optimizer_name', ['klshampoo', 'klsoap'])
    def test_batched_updates_match_single(self, optimizer_name, device):
        shapes = [(2, 3), (2, 3), (3, 2), (2, 2), (2, 2), (2, 2, 2), (), (3,)]
        params = [make_parameter(shape, dtype=torch.float64, device=device) for shape in shapes]
        expected = [p.detach().clone().requires_grad_() for p in params]
        options = {'maximize': True, 'precondition_frequency': 4, 'init_factor': 0.2}
        lr = torch.tensor(0.01, device=device)
        optimizer = build_optimizer(optimizer_name, params, lr=lr, **options, foreach=True)
        reference = build_optimizer(optimizer_name, expected, lr=lr.double(), **options, foreach=False)
        generator = torch.Generator(device=device).manual_seed(42)
        for step in range(16):
            for index, (p, other) in enumerate(zip(params, expected)):
                gradient = torch.randn(p.shape, dtype=p.dtype, device=device, generator=generator)
                inactive = (index == 1 and step < 3) or (index == 0 and step == 11)
                p.grad = None if inactive else gradient.clone()
                other.grad = None if inactive else gradient.clone()
            optimizer.step()
            reference.step()
            torch.testing.assert_close(params, expected, atol=1e-11, rtol=1e-11)
            torch.testing.assert_close([p.grad for p in params], [p.grad for p in expected])

    @pytest.mark.parametrize(
        ('optimizer_name', 'expected'),
        [
            (
                'klshampoo',
                [[-0.542267561, -0.484244078, -0.027349863], [-0.019204151, 0.124741323, 0.396915585]],
            ),
            (
                'klsoap',
                [[-0.560616136, -0.512286186, 0.010716311], [-0.017210223, 0.074154854, 0.358525932]],
            ),
        ],
    )
    def test_reference_trajectory(self, optimizer_name, expected, device):
        param = make_parameter((2, 3), device=device)
        with torch.no_grad():
            param.copy_(torch.linspace(-0.5, 0.5, 6, device=device).reshape_as(param))
        optimizer = build_optimizer(
            optimizer_name,
            [param],
            lr=0.02,
            betas=(0.6, 0.8),
            shampoo_beta=0.7,
            init_factor=0.2,
            eps=0.03,
            weight_decay=0.1,
            precondition_frequency=4,
            max_preconditioner_value=1.5,
        )
        gradients = torch.tensor(
            [
                [[0.6, -0.2, 0.1], [0.3, 0.8, -0.5]],
                [[-0.4, 0.7, 0.2], [0.9, -0.1, 0.6]],
                [[0.2, 0.5, -0.8], [-0.6, 0.4, 0.3]],
            ],
            device=device,
        )
        for step in range(16):
            param.grad = gradients[step % len(gradients)].clone()
            optimizer.step()

        # Generated with the authors' KLOpt in FP32, without torch.compile.
        torch.testing.assert_close(param, torch.tensor(expected, device=device), atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize('optimizer_name', ['klshampoo', 'klsoap'])
    def test_tensor_update_matches_dense_kronecker(self, optimizer_name, device):
        param = make_parameter((2, 2, 2), dtype=torch.float64, device=device)
        gradient = torch.tensor(
            [[[0.2, 0.7], [-0.4, 0.5]], [[0.9, -0.3], [0.1, -0.8]]], dtype=param.dtype, device=device
        )
        param.grad = gradient.clone()
        optimizer = build_optimizer(
            optimizer_name,
            [param],
            lr=0.02,
            betas=(0.6, 0.8),
            init_factor=0.2,
            eps=0.03,
            weight_decay=0.1,
            max_preconditioner_value=None,
        )
        statistics, bases = [], []
        for dim in range(3):
            unfolded = gradient.movedim(dim, 0).reshape(2, -1)
            statistic = unfolded @ unfolded.T
            _, basis = torch.linalg.eigh(statistic)
            statistics.append(statistic / 4 + 1e-8 * torch.eye(2, dtype=param.dtype, device=device))
            bases.append(basis.flip(dims=(1,)))

        eigenvalues = [torch.full((2,), 0.2, dtype=param.dtype, device=device) for _ in range(3)]
        expected, momentum, second_moment = torch.zeros_like(param).flatten(), torch.zeros(8, device=device), 0.0
        optimizer.step()
        torch.testing.assert_close(param, expected.reshape_as(param))

        for step in range(1, 4):
            grad = gradient.roll(step, dims=0) + 0.03 * step
            full_basis = reduce(torch.kron, [basis.contiguous() for basis in bases])
            projected = full_basis.T @ grad.flatten()
            momentum = 0.6 * momentum + 0.4 * grad.flatten()
            if optimizer_name == 'klsoap':
                second_moment = 0.8 * second_moment + 0.2 * projected.square()
                direction = full_basis @ ((full_basis.T @ momentum) / (second_moment.sqrt() + 0.03))
            else:
                spectrum = reduce(torch.kron, eigenvalues).sqrt()
                direction = full_basis @ ((full_basis.T @ momentum) / (spectrum + 0.03))
            expected = expected * 0.998 - 0.02 * direction

            new_eigenvalues = []
            for dim in range(3):
                other_inverse = reduce(
                    torch.kron,
                    [
                        (basis @ torch.diag(values.reciprocal()) @ basis.T).contiguous()
                        for other, (basis, values) in enumerate(zip(bases, eigenvalues))
                        if other != dim
                    ],
                )
                unfolded = grad.movedim(dim, 0).reshape(2, -1)
                target = unfolded @ other_inverse @ unfolded.T / 4
                statistics[dim] = 0.8 * statistics[dim] + 0.2 * target
                new_eigenvalues.append(0.8 * eigenvalues[dim] + 0.2 * (bases[dim].T @ target @ bases[dim]).diag())
            eigenvalues = new_eigenvalues
            for dim in range(3):
                q, r = torch.linalg.qr(statistics[dim] @ bases[dim])
                bases[dim] = q @ torch.diag(r.diag().sign())

            param.grad = grad
            optimizer.step()
            torch.testing.assert_close(param, expected.reshape_as(param), atol=1e-12, rtol=1e-12)

    @pytest.mark.parametrize('shape', [(), (3,), (2, 3)])
    def test_diagonal_fallback(self, shape, device):
        param = make_parameter(shape, dtype=torch.float64, device=device)
        lr = torch.tensor(0.1, device=device)
        optimizer = build_optimizer(
            'klshampoo', [param], lr=lr, betas=(0.6, 0.8), max_precondition_dim=2, eps=0.03, weight_decay=0.1
        )
        expected, momentum, second_moment = torch.zeros_like(param), torch.zeros_like(param), torch.zeros_like(param)
        for step in range(1, 4):
            grad = torch.linspace(-0.5, 0.7, param.numel(), dtype=param.dtype, device=device).reshape(shape)
            grad = grad + step * 0.1
            momentum = 0.6 * momentum + 0.4 * grad
            second_moment = 0.8 * second_moment + 0.2 * grad.square()
            expected = expected * (1.0 - lr.double() * 0.1) - lr.double() * momentum / (second_moment.sqrt() + 0.03)
            param.grad = grad
            optimizer.step()
            torch.testing.assert_close(param, expected, atol=1e-12, rtol=1e-12)

    @pytest.mark.parametrize('optimizer_name', ['klshampoo', 'klsoap'])
    def test_checkpoint_preserves_precision(self, optimizer_name, device):
        params = [make_parameter((2, 3), dtype=torch.bfloat16, device=device) for _ in range(2)]
        optimizer = build_optimizer(optimizer_name, params, lr=0.01)
        gradient = torch.tensor([[0.6, -0.2, 0.1], [0.3, 0.8, -0.5]], dtype=params[0].dtype, device=device)
        for index, param in enumerate(params):
            param.grad = gradient.roll(index, dims=1)
        for _ in range(2):
            optimizer.step()

        restored_params = [param.detach().clone().requires_grad_() for param in params]
        restored = build_optimizer(optimizer_name, restored_params, lr=0.01)
        checkpoint = deepcopy(optimizer.state_dict())
        del checkpoint['param_groups'][0]['foreach']
        restored.load_state_dict(checkpoint)
        for param in restored_params:
            assert restored.state[param]['exp_avg'].dtype == torch.float32
            assert all(t.dtype == torch.float32 for t in restored.state[param]['Q'])

        for step in range(12):
            for param, other in zip(params, restored_params):
                param.grad = param.grad.roll(1, dims=1) + 0.01 * step
                other.grad = param.grad.clone()
            optimizer.step()
            restored.step()
            torch.testing.assert_close(params, restored_params, atol=0.0, rtol=0.0)
