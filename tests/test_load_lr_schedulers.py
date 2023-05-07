import pytest

from pytorch_optimizer import get_supported_lr_schedulers, load_lr_scheduler
from tests.constants import VALID_LR_SCHEDULER_NAMES


@pytest.mark.parametrize('valid_lr_scheduler_names', VALID_LR_SCHEDULER_NAMES)
def test_load_optimizers_valid(valid_lr_scheduler_names):
    load_lr_scheduler(valid_lr_scheduler_names)


def test_load_optimizers_invalid():
    with pytest.raises(NotImplementedError):
        load_lr_scheduler('dummy')


def test_get_supported_lr_schedulers():
    assert len(get_supported_lr_schedulers()) == 10
