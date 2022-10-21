from typing import List

import pytest

from pytorch_optimizer import get_supported_lr_schedulers, load_lr_scheduler

VALID_LR_SCHEDULER_NAMES: List[str] = [
    'CosineAnnealingWarmupRestarts',
]

INVALID_LR_SCHEDULER_NAMES: List[str] = [
    'dummy',
]


@pytest.mark.parametrize('valid_lr_scheduler_names', VALID_LR_SCHEDULER_NAMES)
def test_load_optimizers_valid(valid_lr_scheduler_names):
    load_lr_scheduler(valid_lr_scheduler_names)


@pytest.mark.parametrize('invalid_lr_scheduler_names', INVALID_LR_SCHEDULER_NAMES)
def test_load_optimizers_invalid(invalid_lr_scheduler_names):
    with pytest.raises(NotImplementedError):
        load_lr_scheduler(invalid_lr_scheduler_names)


def test_get_supported_lr_schedulers():
    assert len(get_supported_lr_schedulers()) == 1
