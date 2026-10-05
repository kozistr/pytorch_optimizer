import numpy as np
import pytest
import torch

from pytorch_optimizer.optimizer.utils.matrix import compute_power_schur_newton, zero_power_via_newton_schulz_5
from pytorch_optimizer.optimizer.utils.shape import merge_small_dims


@pytest.mark.parametrize('size', [2, 8])
def test_compute_power_converges(size):
    matrix = 3.0 * torch.eye(size, dtype=torch.float64) + 1.0
    matrix[-1, -1] = 3.0

    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    expected = (eigenvectors * eigenvalues.rsqrt()) @ eigenvectors.T

    result = compute_power_schur_newton(matrix, p=2, ridge_epsilon=0.0, error_tolerance=1e-12)

    torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)


def test_compute_power():
    x = compute_power_schur_newton(torch.zeros((1,)), p=1)
    assert torch.tensor([1000000.0]) == x

    x = compute_power_schur_newton(torch.zeros((1, 2)), p=1)
    assert torch.tensor([1.0]) == x

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
