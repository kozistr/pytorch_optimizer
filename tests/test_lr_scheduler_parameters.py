import numpy as np
import pytest

from pytorch_optimizer import AdamP, get_chebyshev_schedule
from pytorch_optimizer.base.exception import NegativeLRError, NegativeStepError
from pytorch_optimizer.lr_scheduler.cosine_anealing import CosineAnnealingWarmupRestarts
from pytorch_optimizer.lr_scheduler.linear_warmup import PolyScheduler
from tests.utils import Example


def test_cosine_annealing_warmup_restarts_params():
    optimizer = AdamP(Example().parameters())

    with pytest.raises(ValueError) as error_info:
        CosineAnnealingWarmupRestarts(
            optimizer=optimizer,
            first_cycle_steps=10,
            warmup_steps=20,
        )

    assert str(error_info.value) == '[-] warmup_steps must be smaller than first_cycle_steps. 20 < 10'

    min_lr: float = 1e-6
    first_cycle_steps: int = 5
    lr_scheduler = CosineAnnealingWarmupRestarts(
        optimizer=optimizer,
        min_lr=min_lr,
        first_cycle_steps=first_cycle_steps,
        warmup_steps=0,
    )
    lr_scheduler.step_in_cycle = -1
    expected_max_lr: float = round(lr_scheduler.get_lr()[0], 6)
    np.testing.assert_almost_equal(min_lr, expected_max_lr)

    for _ in range(first_cycle_steps + 1):
        lr_scheduler.step(epoch=None)


def test_linear_warmup_lr_scheduler_params():
    optimizer = AdamP(Example().parameters())

    with pytest.raises(ValueError) as error_info:
        PolyScheduler(poly_order=-1, optimizer=optimizer, t_max=1, max_lr=1)

    assert str(error_info.value) == '[-] poly_order must be positive. -1'

    with pytest.raises(NegativeLRError):
        PolyScheduler(optimizer=optimizer, t_max=1, max_lr=-1)

    with pytest.raises(NegativeLRError):
        PolyScheduler(optimizer=optimizer, t_max=1, max_lr=1, min_lr=-1)

    with pytest.raises(NegativeLRError):
        PolyScheduler(optimizer=optimizer, t_max=1, max_lr=1, min_lr=1, init_lr=-1)

    with pytest.raises(NegativeStepError):
        PolyScheduler(optimizer=optimizer, t_max=-1, max_lr=1, min_lr=1, init_lr=1)

    with pytest.raises(NegativeStepError):
        PolyScheduler(optimizer=optimizer, t_max=1, max_lr=1, min_lr=1, init_lr=1, warmup_steps=-1)


def test_chebyshev_params():
    with pytest.raises(ValueError):
        get_chebyshev_schedule(0)
