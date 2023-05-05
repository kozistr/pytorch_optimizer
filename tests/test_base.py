import pytest

from pytorch_optimizer.base.optimizer import BaseOptimizer


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
