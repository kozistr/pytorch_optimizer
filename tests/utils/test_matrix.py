import math

import pytest
import torch

from pytorch_optimizer.optimizer.utils.matrix import (
    NS_COEFFICIENTS,
    batched_power_iteration,
    compute_power_newton_db,
    compute_power_schur_newton,
    compute_power_svd,
    power_iteration,
    zero_power_via_gram_newton_schulz,
    zero_power_via_newton_schulz_5,
)
from pytorch_optimizer.optimizer.utils.shape import merge_small_dims
from tests.utils.distributed import run_distributed


@pytest.mark.parametrize('num_iters', [0, 1, 2, 5, 100])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_power_iteration(num_iters, dtype, device, monkeypatch):
    eigenvalues = torch.tensor([5.0, 3.0, 0.0], dtype=dtype, device=device)
    matrix = eigenvalues.diag().T
    original = matrix.clone()
    initial = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, device=device)
    monkeypatch.setattr(torch, 'randn', lambda *_args, **_kwargs: initial.clone())

    vector = initial.double() * eigenvalues.double().pow(num_iters)
    expected = (vector.square() * eigenvalues).sum()
    if num_iters > 0:
        expected.div_(vector.square().sum())

    result = power_iteration(matrix, num_iters=num_iters)

    torch.testing.assert_close(result, expected.to(dtype))
    torch.testing.assert_close(matrix, original)


@pytest.mark.parametrize('vectors', [1, 16])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_batched_power_iteration(vectors, dtype, device):
    matrices = torch.diag_embed(torch.tensor([[1.0, 4.0, 2.0], [3.0, 1.0, 2.0]], dtype=dtype, device=device))
    original = matrices.clone()

    with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
        torch.manual_seed(42)
        result = batched_power_iteration(matrices, num_vectors=vectors)

    expected = torch.tensor([4.0, 3.0], dtype=dtype, device=device).view(-1, 1, 1)

    torch.testing.assert_close(result, expected, atol=0.025, rtol=0.01)
    torch.testing.assert_close(matrices, original, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(('num_iters', 'value'), [(1, 0.6875), (3, 0.9752996308188813), (20, 1.0)])
@pytest.mark.parametrize('inverse', [False, True])
def test_compute_power_newton_db(num_iters, value, inverse, device):
    matrix = torch.tensor([[[2.5, -1.5], [-1.5, 2.5]]], dtype=torch.float64, device=device)
    original = matrix.clone()
    scale = torch.tensor([[[4.0]]], dtype=matrix.dtype, device=device)
    other = 0.5 if inverse else 2.0
    diagonal, off_diagonal = (value + other) / 2.0, (value - other) / 2.0
    expected = torch.tensor([[diagonal, off_diagonal], [off_diagonal, diagonal]], dtype=matrix.dtype, device=device)

    result = compute_power_newton_db(matrix, scale, num_iters, inverse)

    torch.testing.assert_close(result, expected.unsqueeze(0))
    torch.testing.assert_close(matrix, original, atol=0.0, rtol=0.0)
    torch.testing.assert_close(scale, torch.full_like(scale, 4.0), atol=0.0, rtol=0.0)


@pytest.mark.parametrize('batch', [False, True])
@pytest.mark.parametrize('power', [2, 4])
def test_compute_power_svd(batch, power, device):
    matrix = torch.tensor(
        [[1.0, 1.0, 0.0], [1.0, 1.0 + 1e-10, 0.0], [0.0, 0.0, 3.0]], device=device, dtype=torch.float64
    )
    if batch:
        matrix = torch.stack([matrix, 2.0 * matrix])
    original = matrix.clone()

    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    expected = (eigenvectors * eigenvalues.pow(-1.0 / power).unsqueeze(-2)) @ eigenvectors.mT

    result = compute_power_svd(matrix, power)

    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(matrix, original)


@pytest.mark.parametrize('size', [2, 8])
def test_compute_power_converges(size):
    matrix = 3.0 * torch.eye(size, dtype=torch.float64) + 1.0
    matrix[-1, -1] = 3.0

    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    expected = (eigenvectors * eigenvalues.rsqrt()) @ eigenvectors.T

    result = compute_power_schur_newton(matrix, p=2, ridge_epsilon=0.0, error_tolerance=1e-12)

    torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)


def test_compute_power_regularization(device, monkeypatch):
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64, device=device)
    original = matrix.clone()
    identity = torch.eye(2, dtype=matrix.dtype, device=device)
    damped = matrix + 0.1 * identity
    expected = identity * (3.0 / (2.0 * torch.linalg.norm(damped))).sqrt()
    monkeypatch.setattr('pytorch_optimizer.optimizer.utils.matrix.power_iteration', lambda _: matrix.new_tensor(5.0))

    result = compute_power_schur_newton(matrix, p=2, ridge_epsilon=0.02, max_iters=0)

    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(matrix, original, rtol=0.0, atol=0.0)


def test_compute_power(monkeypatch):
    x = compute_power_schur_newton(torch.zeros((1,)), p=1)
    torch.testing.assert_close(x, torch.tensor([1000000.0]))

    x = compute_power_schur_newton(torch.tensor([[4.0]]), p=2, ridge_epsilon=0.0)
    torch.testing.assert_close(x, torch.tensor([[0.5]]))

    matrix = torch.ones(2, 2, dtype=torch.float64)
    monkeypatch.setattr('pytorch_optimizer.optimizer.utils.matrix.power_iteration', lambda _: 2.0)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix + 2e-6 * torch.eye(2))
    for power in (1, 2):
        expected = (eigenvectors * eigenvalues.pow(-1.0 / power)) @ eigenvectors.T
        result = compute_power_schur_newton(matrix, p=power, error_tolerance=1e-10)
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

    result = compute_power_schur_newton(matrix, p=16, max_error_ratio=0.0)
    torch.testing.assert_close(result, 1.0946 * torch.eye(2, dtype=matrix.dtype), atol=1e-4, rtol=0.0)


def test_merge_small_dims():
    assert merge_small_dims([1, 2, 512, 1, 2048, 1, 3, 4], max_dim=1024) == [1024, 2048, 12]
    assert merge_small_dims([1, 2, 768, 1, 2048], max_dim=1024) == [2, 768, 2048]
    assert merge_small_dims([1, 1, 1], max_dim=1) == [1]


def test_zero_power_via_newton_schulz_5():
    x = torch.FloatTensor(([[-1.5724165, 1.5850062], [-0.87536967, 0.31970903], [-0.18436244, -0.16805087]]))
    output = zero_power_via_newton_schulz_5(x).float()
    expected_output = torch.tensor([[-0.3359375, 0.671875], [-0.734375, -0.38671875], [-0.3828125, -0.3984375]])
    torch.testing.assert_close(output, expected_output, rtol=3e-2, atol=3e-2)

    with pytest.raises(ValueError):
        zero_power_via_newton_schulz_5(x[0])

    for name, schedule in NS_COEFFICIENTS.items():
        torch.testing.assert_close(
            zero_power_via_newton_schulz_5(x, weights=name), zero_power_via_newton_schulz_5(x, weights=schedule)
        )

    for weights in ('invalid', [], [(1.0, 2.0)]):
        with pytest.raises(ValueError):
            zero_power_via_newton_schulz_5(x, weights=weights)
        with pytest.raises(ValueError):
            zero_power_via_gram_newton_schulz(x, weights=weights)

    with pytest.raises(ValueError):
        zero_power_via_gram_newton_schulz(x[0])


def _check_dtensor_newton_schulz(_rank):
    dtensor = pytest.importorskip('torch.distributed.tensor')
    mesh = torch.distributed.device_mesh.init_device_mesh('cpu', (1, 2))
    cases = [
        ((3, 2), dtensor.Shard(0), 5),
        ((2, 3), dtensor.Shard(1), 0),
        ((2, 3, 2), dtensor.Shard(0), 3),
        ((1, 2), dtensor.Replicate(), 1),
    ]
    helpers = [(zero_power_via_newton_schulz_5, torch.float64), (zero_power_via_gram_newton_schulz, torch.float32)]
    for helper, dtype in helpers:
        for shape, placement, num_steps in cases:
            matrix = torch.arange(math.prod(shape), dtype=dtype).reshape(shape) / 10.0
            original = matrix.clone()
            sharded = dtensor.distribute_tensor(matrix, mesh, [dtensor.Replicate(), placement])
            options = {'num_steps': num_steps, 'safety_factor': 1.5, 'eps': 1e-5, 'dtype': dtype}
            expected = helper(matrix, **options)
            result = helper(sharded, **options)
            torch.testing.assert_close(result.full_tensor(), expected, atol=0.0, rtol=0.0)
            torch.testing.assert_close(sharded.full_tensor(), original, atol=0.0, rtol=0.0)
            assert result.placements == sharded.placements


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason='Gloo is unavailable')
def test_dtensor_newton_schulz(tmp_path):
    pytest.importorskip('torch.distributed.tensor')
    run_distributed(_check_dtensor_newton_schulz, tmp_path / 'store')


@pytest.mark.parametrize('shape', [(2, 3), (3, 2), (2, 2, 3), (2, 3, 2)])
@pytest.mark.parametrize('num_steps', [0, 1, 5])
@pytest.mark.parametrize('safety_factor', [1.0, 1.5])
@pytest.mark.parametrize(
    ('helper', 'dtype'),
    [(zero_power_via_newton_schulz_5, torch.float64), (zero_power_via_gram_newton_schulz, torch.float32)],
)
def test_newton_schulz_singular_values(shape, num_steps, safety_factor, helper, dtype, device):
    matrix = torch.arange(1, math.prod(shape) + 1, dtype=dtype, device=device)
    matrix = matrix.reshape(*shape[:-2], shape[-1], shape[-2]).mT
    original = matrix.clone()
    weights = [(3.4445, -4.7750, 2.0315), (2.8366, -3.0525, 1.2012)]

    reference = matrix.double()
    u, s, vh = torch.linalg.svd(reference, full_matrices=False)
    s.div_(torch.linalg.vector_norm(reference, dim=(-2, -1), keepdim=False).unsqueeze(-1) * safety_factor)
    for index in range(num_steps):
        w0, w1, w2 = weights[min(index, len(weights) - 1)]
        s = w0 * s + w1 * s.pow(3) + w2 * s.pow(5)
    expected = ((u * s.unsqueeze(-2)) @ vh).to(dtype)

    result = helper(
        matrix, num_steps=num_steps, safety_factor=safety_factor, weights=weights, dtype=matrix.dtype
    )

    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(matrix, original)


@pytest.mark.parametrize('reset_steps', [(), (1, 3)])
def test_gram_newton_schulz_restarts(reset_steps, device):
    matrix = torch.arange(1, 25, dtype=torch.float32, device=device).reshape(2, 2, 2, 3).mT
    matrix[..., :2, :].add_(10.0 * torch.eye(2, device=device))
    expected = zero_power_via_newton_schulz_5(matrix, dtype=matrix.dtype)
    result = zero_power_via_gram_newton_schulz(matrix, reset_steps=reset_steps, dtype=matrix.dtype)
    torch.testing.assert_close(result, expected)


def test_gram_newton_schulz_half_precision(device):
    matrix = torch.tensor([[[1e8, 0.0, 0.0], [0.0, 2e8, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], device=device)
    original = matrix.clone()
    expected = zero_power_via_newton_schulz_5(matrix, dtype=torch.float64).half()
    result = zero_power_via_gram_newton_schulz(matrix)
    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(matrix, original)


@pytest.mark.parametrize(
    ('helper', 'dtype'),
    [
        (power_iteration, torch.float64),
        (zero_power_via_newton_schulz_5, torch.float64),
        (zero_power_via_gram_newton_schulz, torch.float32),
    ],
)
@pytest.mark.skipif(not torch._dynamo.is_dynamo_supported(), reason='torch.compile is unavailable in this runtime')
def test_matrix_helpers_compile(helper, dtype, device):
    matrix = torch.tensor([[5.0, 1.0], [1.0, 3.0]], dtype=dtype, device=device)
    kwargs = {} if helper is power_iteration else {'dtype': matrix.dtype}
    expected = torch.linalg.eigvalsh(matrix)[-1] if helper is power_iteration else helper(matrix, **kwargs)
    compiled = torch.compile(helper, backend='eager', fullgraph=True)

    torch.testing.assert_close(compiled(matrix, **kwargs), expected)
