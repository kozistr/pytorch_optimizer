import pytest

from pytorch_optimizer.optimizer import AdaFactor
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


class TestAdaFactorParameters:
    @pytest.mark.parametrize('recipe', [(1.0, 1, True, True, 1e-6), (1.0, 1, False, True, 1.0)])
    def test_get_relative_step_size(self, recipe):
        assert AdaFactor.get_relative_step_size(*recipe[:-1]) == recipe[-1]

    @pytest.mark.parametrize('recipe', [(1.0, 1.0, True, 1.0), (2.0, 1.0, False, 2.0)])
    def test_get_lr(self, recipe):
        optimizer = build_optimizer('adafactor', [make_parameter()])
        assert optimizer.get_lr(*recipe[:-1]) == recipe[-1]
