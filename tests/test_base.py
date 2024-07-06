import pytest
import torch

from pytorch_optimizer.base.optimizer import BaseOptimizer
from tests.utils import simple_parameter


def test_set_hessian():
    param = simple_parameter()

    param_groups = [{'params': param}]
    hessian = [torch.zeros(2, 1)]

    with pytest.raises(ValueError):
        BaseOptimizer.set_hessian(param_groups, {'dummy': param}, hessian)


def test_compute_hutchinson_hessian():
    with pytest.raises(NotImplementedError):
        BaseOptimizer.compute_hutchinson_hessian({}, {}, distribution='dummy')


def test_validate_boundary():
    x: float = -1.0

    with pytest.raises(ValueError):
        BaseOptimizer.validate_boundary(x, -2.0, bound_type='upper')

    with pytest.raises(ValueError):
        BaseOptimizer.validate_boundary(x, 1.0, bound_type='lower')


@pytest.mark.parametrize('range_type', ['[]', '[)', '(]', '()'])
def test_validate_range(range_type):
    with pytest.raises(ValueError):
        BaseOptimizer.validate_range(-1.0, 'x', 0.0, 1.0, range_type=range_type)


def test_mod():
    with pytest.raises(ValueError):
        BaseOptimizer.validate_mod(10, 3)
