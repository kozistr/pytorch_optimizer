import numpy as np
import pytest

from pytorch_optimizer import AdamP, get_chebyshev_schedule
from pytorch_optimizer.lr_scheduler.cosine_anealing import CosineAnnealingWarmupRestarts
from tests.utils import Example


def test_cosine_annealing_warmup_restarts_params():
    model = Example()
    optimizer = AdamP(model.parameters())

    with pytest.raises(ValueError):
        CosineAnnealingWarmupRestarts(
            optimizer=optimizer,
            first_cycle_steps=10,
            warmup_steps=20,
        )

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


def test_chebyshev_params():
    with pytest.raises(IndexError):
        get_chebyshev_schedule(2)
