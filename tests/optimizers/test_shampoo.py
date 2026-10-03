import numpy as np
import pytest
import torch

from pytorch_optimizer.optimizer.shampoo_utils import (
    BlockPartitioner,
    PreConditioner,
    compute_power_schur_newton,
    merge_small_dims,
    zero_power_via_newton_schulz_5,
)
from pytorch_optimizer.optimizer.utils import to_real
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


@pytest.mark.parametrize('pre_conditioner_type', [0, 1, 2])
def test_scalable_shampoo_pre_conditioner_with_svd(pre_conditioner_type):
    params = [make_parameter(shape) for shape in ((8, 2), (4, 8), (1, 4))]
    optimizer = build_optimizer(
        'scalableshampoo',
        params,
        block_size=4,
        start_preconditioning_step=1,
        preconditioning_compute_steps=1,
        pre_conditioner_type=pre_conditioner_type,
        use_svd=True,
    )
    optimizer.step()
    assert all(torch.isfinite(param).all() for param in params)


class TestShampooUtils:
    @pytest.mark.parametrize('size', [2, 8])
    def test_compute_power_converges(self, size):
        matrix = 3.0 * torch.eye(size, dtype=torch.float64) + 1.0
        matrix[-1, -1] = 3.0

        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        expected = (eigenvectors * eigenvalues.rsqrt()) @ eigenvectors.T

        result = compute_power_schur_newton(matrix, p=2, ridge_epsilon=0.0, error_tolerance=1e-12)

        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

    def test_compute_power(self):
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

    def test_merge_small_dims(self):
        case1 = [1, 2, 512, 1, 2048, 1, 3, 4]
        expected_case1 = [1024, 2048, 12]
        assert expected_case1 == merge_small_dims(case1, max_dim=1024)

        case2 = [1, 2, 768, 1, 2048]
        expected_case2 = [2, 768, 2048]
        assert expected_case2 == merge_small_dims(case2, max_dim=1024)

        case3 = [1, 1, 1]
        expected_case3 = [1]
        assert expected_case3 == merge_small_dims(case3, max_dim=1)

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
        grad = torch.zeros((16, 4))

        pre_conditioner = PreConditioner(var, 0.9, 0, 4, 1, 64, True, 0)
        pre_conditioner.add_statistics(grad)
        pre_conditioner.compute_pre_conditioners()

    @pytest.mark.parametrize('pre_conditioner_type', [0, 1, 2, 3])
    def test_pre_conditioner_type(self, pre_conditioner_type):
        var = torch.zeros((4, 4, 32))
        if pre_conditioner_type in (0, 1, 2):
            PreConditioner(var, 0.9, 0, 128, 1, 8192, True, pre_conditioner_type=pre_conditioner_type)
        else:
            with pytest.raises(ValueError):
                PreConditioner(var, 0.9, 0, 128, 1, 8192, True, pre_conditioner_type=pre_conditioner_type)

    def test_zero_power_via_newton_schulz_5(self):
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
