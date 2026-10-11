from unittest.mock import patch

import pytest
import torch

from pytorch_optimizer.base.optimizer import BaseOptimizer
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


class TestValidationMethods:
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
    def test_set_hessian(self, param_groups):
        hessian = [torch.zeros(2, 1)]
        with pytest.raises(ValueError):
            BaseOptimizer.set_hessian(param_groups, {'dummy': param_groups[0]['params']}, hessian)

    def test_compute_hutchinson_hessian(self):
        with pytest.raises(NotImplementedError):
            BaseOptimizer.compute_hutchinson_hessian({}, {}, distribution='dummy')

    def test_rademacher_hessian_with_cross_terms(self):
        parameter = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        hessian = torch.tensor([[2.0, 3.0], [3.0, 4.0]], dtype=torch.float64)
        loss = 0.5 * parameter @ hessian @ parameter
        parameter.grad = torch.autograd.grad(loss, parameter, create_graph=True)[0]
        state = {parameter: {'hessian': torch.zeros_like(parameter)}}

        with torch.random.fork_rng(devices=[]):
            # These four seeded directions cancel the off-diagonal Hessian terms.
            torch.manual_seed(0)
            BaseOptimizer.compute_hutchinson_hessian(
                [{'params': [parameter]}], state, num_samples=4, distribution='rademacher'
            )

        torch.testing.assert_close(state[parameter]['hessian'], hessian.diagonal())


class TestGradientMethods:
    @pytest.mark.parametrize(
        ('gradient', 'expected', 'expected_norm'), [(0.0, 0.0, 1.9), (1.0, 1.95, 1.95), (4.0, 4.0, 2.1)]
    )
    def test_adanorm_gradient(self, gradient, expected, expected_norm):
        grad = torch.tensor([gradient])
        norm = torch.tensor([2.0])
        result = BaseOptimizer.get_adanorm_gradient(grad, True, norm, r=None)

        torch.testing.assert_close(result, torch.tensor([expected]))
        torch.testing.assert_close(norm, torch.tensor([expected_norm]))

    @pytest.mark.parametrize('maximize', [True, False])
    def test_maximize_gradient(self, maximize: bool):
        grad = torch.ones(1)
        BaseOptimizer.maximize_gradient(grad, maximize)

        torch.testing.assert_close(grad, torch.tensor([-1.0 if maximize else 1.0]))


def test_can_use_foreach():
    assert BaseOptimizer.can_use_foreach({}, foreach=False) is False
    assert BaseOptimizer.collect_trainable_params({'params': []}, {}, None) == ([], [], {})


def test_compile_foreach():
    optimizer = build_optimizer('radam', [make_parameter()])
    step = optimizer._step_foreach
    compile_kwargs = {'dynamic': False}

    with patch('pytorch_optimizer.base.optimizer.compile_foreach_step') as compile_step:
        optimizer._compile_foreach(compile_kwargs)

    compile_step.assert_called_once_with(step, compile_kwargs)
    assert optimizer._step_foreach is compile_step.return_value
