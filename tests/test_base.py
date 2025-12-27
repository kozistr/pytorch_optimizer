import pytest
import torch

from pytorch_optimizer.base.optimizer import BaseOptimizer


class TestValidationMethods:
    """Tests for validation methods."""

    @pytest.mark.parametrize('range_type', ['[]', '[)', '(]', '()'])
    def test_validate_range(self, range_type):
        with pytest.raises(ValueError):
            BaseOptimizer.validate_range(-1.0, 'x', 0.0, 1.0, range_type)

    @pytest.mark.parametrize(
        ('x', 'bound', 'bound_type'),
        [
            (-1.0, -2.0, 'upper'),
            (-1.0, 1.0, 'lower'),
        ],
    )
    def test_validate_boundary(self, x, bound, bound_type):
        with pytest.raises(ValueError):
            BaseOptimizer.validate_boundary(x, bound, bound_type)

    def test_validate_non_positive(self):
        with pytest.raises(ValueError):
            BaseOptimizer.validate_non_positive(1.0, 'x')

    def test_validate_mod(self):
        with pytest.raises(ValueError):
            BaseOptimizer.validate_mod(10, 3)


class TestHessianMethods:
    """Tests for Hessian-related methods."""

    def test_set_hessian(self, param_groups):
        hessian = [torch.zeros(2, 1)]
        with pytest.raises(ValueError):
            BaseOptimizer.set_hessian(param_groups, {'dummy': param_groups[0]['params']}, hessian)

    def test_compute_hutchinson_hessian(self):
        with pytest.raises(NotImplementedError):
            BaseOptimizer.compute_hutchinson_hessian({}, {}, distribution='dummy')


class TestGradientMethods:
    """Tests for gradient-related methods."""

    @pytest.mark.parametrize('maximize', [True, False])
    def test_maximize_gradient(self, maximize: bool):
        grad = torch.ones(1)
        expected = -grad if maximize else grad
        BaseOptimizer.maximize_gradient(grad, maximize)

        torch.testing.assert_close(grad, expected)
