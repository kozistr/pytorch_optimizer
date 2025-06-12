import pytest

from pytorch_optimizer.loss import get_supported_loss_functions
from pytorch_optimizer.lr_scheduler import get_supported_lr_schedulers, load_lr_scheduler
from pytorch_optimizer.optimizer import get_supported_optimizers, load_optimizer
from tests.constants import (
    INVALID_LR_SCHEDULER_NAMES,
    INVALID_OPTIMIZER_NAMES,
    VALID_LR_SCHEDULER_NAMES,
    VALID_OPTIMIZER_NAMES,
)


@pytest.mark.parametrize('valid_optimizer_names', VALID_OPTIMIZER_NAMES)
def test_load_optimizer_valid(valid_optimizer_names):
    load_optimizer(valid_optimizer_names)


@pytest.mark.parametrize('invalid_optimizer_names', INVALID_OPTIMIZER_NAMES)
def test_load_optimizer_invalid(invalid_optimizer_names):
    with pytest.raises(NotImplementedError):
        load_optimizer(invalid_optimizer_names)


@pytest.mark.parametrize('valid_lr_scheduler_names', VALID_LR_SCHEDULER_NAMES)
def test_load_lr_scheduler_valid(valid_lr_scheduler_names):
    load_lr_scheduler(valid_lr_scheduler_names)


@pytest.mark.parametrize('invalid_lr_scheduler_names', INVALID_LR_SCHEDULER_NAMES)
def test_load_lr_scheduler_invalid(invalid_lr_scheduler_names):
    with pytest.raises(NotImplementedError):
        load_lr_scheduler(invalid_lr_scheduler_names)


def test_get_supported_optimizers():
    assert len(get_supported_optimizers()) == 106
    assert len(get_supported_optimizers('adam*')) == 10
    assert len(get_supported_optimizers(['adam*', 'ranger*'])) == 13


def test_get_supported_lr_schedulers():
    assert len(get_supported_lr_schedulers()) == 16
    assert len(get_supported_lr_schedulers('cosine*')) == 4
    assert len(get_supported_lr_schedulers(['cosine*', '*warm*'])) == 5


def test_get_supported_loss_functions():
    assert len(get_supported_loss_functions()) == 13
    assert len(get_supported_loss_functions('*focal*')) == 4
    assert len(get_supported_loss_functions(['*focal*', 'bce*'])) == 5
