import pytest

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


class TestAdaFactorParameters:
    @pytest.mark.parametrize('recipe', [(1.0, 1, True, True, 1e-6), (1.0, 1, False, True, 1.0)])
    def test_get_relative_step_size(self, recipe):
        opt = build_optimizer('adafactor', [make_parameter()])

        expected = opt.get_relative_step_size(*recipe[:-1])

        assert expected == recipe[-1]

    @pytest.mark.parametrize('recipe', [(1.0, 1.0, True, 1.0), (2.0, 1.0, False, 2.0)])
    def test_get_lr(self, recipe):
        opt = build_optimizer('adafactor', [make_parameter()])

        expected = opt.get_lr(*recipe[:-1])

        assert expected == recipe[-1]
