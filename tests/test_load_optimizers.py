import pytest

from pytorch_optimizer import get_supported_optimizers, load_optimizer
from tests.constants import INVALID_OPTIMIZER_NAMES, VALID_OPTIMIZER_NAMES


@pytest.mark.parametrize('valid_optimizer_names', VALID_OPTIMIZER_NAMES)
def test_load_optimizers_valid(valid_optimizer_names):
    load_optimizer(valid_optimizer_names)


@pytest.mark.parametrize('invalid_optimizer_names', INVALID_OPTIMIZER_NAMES)
def test_load_optimizers_invalid(invalid_optimizer_names):
    with pytest.raises(NotImplementedError):
        load_optimizer(invalid_optimizer_names)


def test_get_supported_optimizers():
    assert len(get_supported_optimizers()) == 24
