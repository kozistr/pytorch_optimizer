import math

import numpy as np
import pytest
import torch

from pytorch_optimizer.optimizer.utils.matrix import (
    compute_power_schur_newton,
    compute_power_svd,
    power_iteration,
    zero_power_via_newton_schulz_5,
)
from pytorch_optimizer.optimizer.utils.shape import merge_small_dims


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
    assert result.dtype == dtype
    assert result.device == matrix.device


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
    assert result.dtype == matrix.dtype


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


def test_compute_power():
    x = compute_power_schur_newton(torch.zeros((1,)), p=1)
    assert torch.tensor([1000000.0]) == x

    x = compute_power_schur_newton(torch.tensor([[4.0]]), p=2, ridge_epsilon=0.0)
    torch.testing.assert_close(x, torch.tensor([[0.5]]))

    x = compute_power_schur_newton(torch.ones((2, 2)), p=1)
    assert np.sum(x.numpy() - np.asarray([[252206.4062, -252205.8750], [-252205.8750, 252206.4062]])) < 200

    x = compute_power_schur_newton(torch.ones((2, 2)), p=16, max_error_ratio=0.0)
    np.testing.assert_array_almost_equal(
        np.asarray([[1.0946, 0.0000], [0.0000, 1.0946]]),
        x.numpy(),
        decimal=2,
    )

    x = compute_power_schur_newton(torch.ones((2, 2)), p=2)
    assert np.sum(x.numpy() - np.asarray([[359.1108, -358.4036], [-358.4036, 359.1108]])) < 50


def test_merge_small_dims():
    case1 = [1, 2, 512, 1, 2048, 1, 3, 4]
    expected_case1 = [1024, 2048, 12]
    assert expected_case1 == merge_small_dims(case1, max_dim=1024)

    case2 = [1, 2, 768, 1, 2048]
    expected_case2 = [2, 768, 2048]
    assert expected_case2 == merge_small_dims(case2, max_dim=1024)

    case3 = [1, 1, 1]
    expected_case3 = [1]
    assert expected_case3 == merge_small_dims(case3, max_dim=1)


def test_zero_power_via_newton_schulz_5():
    x = torch.FloatTensor(([[-1.5724165, 1.5850062], [-0.87536967, 0.31970903], [-0.18436244, -0.16805087]]))
    output = zero_power_via_newton_schulz_5(x).float().numpy()

    expected_output = np.asarray([[-0.3359375, 0.671875], [-0.734375, -0.38671875], [-0.3828125, -0.3984375]])

    np.testing.assert_allclose(output, expected_output, rtol=3e-2, atol=3e-2)

    with pytest.raises(ValueError):
        zero_power_via_newton_schulz_5(x[0])

    output_by_name = zero_power_via_newton_schulz_5(x, weights='original')
    output_by_tuple = zero_power_via_newton_schulz_5(x, weights=(3.4445, -4.7750, 2.0315))

    torch.testing.assert_close(output_by_name, output_by_tuple)

    quintic_output = zero_power_via_newton_schulz_5(x, weights='quintic')
    polar_express_output = zero_power_via_newton_schulz_5(x, weights='polar_express')
    polar_express_safer_output = zero_power_via_newton_schulz_5(x, weights='polar_express_safer')
    custom_schedule_output = zero_power_via_newton_schulz_5(
        x,
        weights=[(4.0848, -6.8946, 2.9270), (3.9505, -6.3029, 2.6377)],
    )

    assert quintic_output.shape == x.shape
    assert polar_express_output.shape == x.shape
    assert polar_express_safer_output.shape == x.shape
    assert custom_schedule_output.shape == x.shape

    with pytest.raises(ValueError):
        zero_power_via_newton_schulz_5(x, weights='invalid')

    with pytest.raises(ValueError):
        zero_power_via_newton_schulz_5(x, weights=[])

    with pytest.raises(ValueError):
        zero_power_via_newton_schulz_5(x, weights=[(1.0, 2.0)])


@pytest.mark.parametrize('shape', [(2, 3), (3, 2), (2, 2, 3), (2, 3, 2)])
@pytest.mark.parametrize('num_steps', [0, 1, 5])
@pytest.mark.parametrize('safety_factor', [1.0, 1.5])
def test_newton_schulz_singular_values(shape, num_steps, safety_factor, device):
    matrix = torch.arange(1, math.prod(shape) + 1, dtype=torch.float64, device=device)
    matrix = matrix.reshape(*shape[:-2], shape[-1], shape[-2]).mT
    original = matrix.clone()
    weights = [(3.4445, -4.7750, 2.0315), (2.8366, -3.0525, 1.2012)]

    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    s.div_(torch.linalg.vector_norm(matrix, dim=(-2, -1), keepdim=False).unsqueeze(-1) * safety_factor)
    for index in range(num_steps):
        w0, w1, w2 = weights[min(index, len(weights) - 1)]
        s = w0 * s + w1 * s.pow(3) + w2 * s.pow(5)
    expected = (u * s.unsqueeze(-2)) @ vh

    result = zero_power_via_newton_schulz_5(
        matrix, num_steps=num_steps, safety_factor=safety_factor, weights=weights, dtype=matrix.dtype
    )

    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(matrix, original)
    assert result.shape == matrix.shape


@pytest.mark.parametrize('helper', [power_iteration, zero_power_via_newton_schulz_5])
@pytest.mark.skipif(not torch._dynamo.is_dynamo_supported(), reason='torch.compile is unavailable in this runtime')
def test_matrix_helpers_compile(helper, device):
    matrix = torch.tensor([[5.0, 1.0], [1.0, 3.0]], dtype=torch.float64, device=device)
    kwargs = {} if helper is power_iteration else {'dtype': matrix.dtype}
    expected = torch.linalg.eigvalsh(matrix)[-1] if helper is power_iteration else helper(matrix, **kwargs)
    compiled = torch.compile(helper, backend='eager', fullgraph=True)

    torch.testing.assert_close(compiled(matrix, **kwargs), expected)
