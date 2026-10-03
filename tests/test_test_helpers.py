import numpy as np
import pytest
import torch

from pytorch_optimizer.optimizer import OPTIMIZERS
from tests.fixtures import build_model, make_parameter, make_sparse_parameters
from tests.optimizer_cases import optimizers_with_argument, supports_gradient
from tests.utils import build_optimizer, build_optimizer_parameters


class TestParameterHelpers:
    def test_sparse_parameters_have_independent_storage(self):
        dense, sparse = make_sparse_parameters()

        torch.testing.assert_close(dense, sparse)
        torch.testing.assert_close(dense.grad, sparse.grad.to_dense())

        with torch.no_grad():
            dense.add_(1.0)

        assert not torch.equal(dense, sparse)

    def test_training_config_does_not_retain_model_parameters(self):
        recipe = {'max_lr': 0.1}

        first, _ = build_model()
        second, _ = build_model()

        first_params, first_config = build_optimizer_parameters(first.parameters(), 'alig', recipe)
        second_params, second_config = build_optimizer_parameters(second.parameters(), 'alig', recipe)

        with torch.no_grad():
            for param in (*first_params, *second_params):
                param.fill_(2.0)

        first_config['projection_fn']()

        torch.testing.assert_close(sum(param.norm().square() for param in first_params), torch.tensor(1.0))
        assert all(torch.all(param == 2.0) for param in second_params)

        second_config['projection_fn']()

        torch.testing.assert_close(sum(param.norm().square() for param in second_params), torch.tensor(1.0))
        assert recipe == {'max_lr': 0.1}

    def test_optimizer_options_override_test_defaults(self):
        optimizer = build_optimizer('ranger21', [make_parameter()], num_iterations=75, lookahead_merge_time=5)

        assert optimizer.start_warm_down + optimizer.num_warm_down_iterations == 75
        assert optimizer.lookahead_merge_time == 5


class TestCapabilityDetection:
    def test_inherited_constructor_arguments(self, monkeypatch):
        class InheritedSGD(torch.optim.SGD):
            pass

        monkeypatch.setitem(OPTIMIZERS, 'inheritedsgd', InheritedSGD)

        assert 'inheritedsgd' in optimizers_with_argument('foreach')
        assert 'inheritedsgd' in optimizers_with_argument('maximize')

    @pytest.mark.parametrize('kind', ['sparse', 'complex'])
    def test_probe_preserves_random_generators(self, kind, monkeypatch):
        supports_gradient.cache_clear()

        original_step = torch.optim.SGD.step

        def random_step(optimizer, *args, **kwargs):
            torch.rand(1)
            np.random.random()
            return original_step(optimizer, *args, **kwargs)

        monkeypatch.setattr(torch.optim.SGD, 'step', random_step)

        torch_state = torch.random.get_rng_state().clone()
        numpy_state = np.random.get_state()

        assert supports_gradient('sgd', kind)

        torch.testing.assert_close(torch.random.get_rng_state(), torch_state)

        after = np.random.get_state()
        assert after[0] == numpy_state[0]

        np.testing.assert_array_equal(after[1], numpy_state[1])
        assert after[2:] == numpy_state[2:]

    def test_probe_propagates_unexpected_errors(self, monkeypatch):
        supports_gradient.cache_clear()

        def broken_step(*_args, **_kwargs):
            raise RuntimeError('unexpected optimizer failure')

        monkeypatch.setattr(torch.optim.SGD, 'step', broken_step)

        with pytest.raises(RuntimeError, match='unexpected optimizer failure'):
            supports_gradient('sgd', 'sparse')

    def test_invalid_gradient_kind(self):
        with pytest.raises(ValueError, match='Unknown gradient kind'):
            supports_gradient('sgd', 'unknown')
