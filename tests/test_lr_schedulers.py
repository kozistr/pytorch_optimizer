import numpy as np
import pytest

from pytorch_optimizer import AdamP, get_chebyshev_schedule
from pytorch_optimizer.lr_scheduler.chebyshev import chebyshev_perm
from pytorch_optimizer.lr_scheduler.cosine_anealing import CosineAnnealingWarmupRestarts
from tests.utils import Example


def test_cosine_annealing_warmup_restarts():
    model = Example()
    optimizer = AdamP(model.parameters())

    # case 1
    lr_scheduler = CosineAnnealingWarmupRestarts(
        optimizer, first_cycle_steps=500, cycle_mult=1.0, max_lr=0.1, min_lr=0.001, warmup_steps=100, gamma=1.0
    )

    expected_lrs = [
        0.001000,
        0.001990,
        0.002980,
        0.003970,
        0.004960,
    ]

    for epoch in range(5):
        lr_scheduler.step(epoch)

        np.testing.assert_almost_equal(expected_lrs[epoch], lr_scheduler.get_lr()[0])

    # case 2
    lr_scheduler = CosineAnnealingWarmupRestarts(
        optimizer, first_cycle_steps=10, cycle_mult=1.0, max_lr=0.1, min_lr=0.001, warmup_steps=5, gamma=0.5
    )

    expected_lrs = [
        0.001,
        0.0208,
        0.0406,
        0.0604,
        0.0802,
        0.1,
        0.090546,
        0.065796,
        0.035204,
        0.010454,
        0.001,
        0.0108,
        0.0206,
        0.0304,
        0.0402,
        0.05,
        0.045321,
        0.033071,
        0.017929,
        0.005679,
    ]

    for epoch in range(20):
        lr_scheduler.step(epoch)

        np.testing.assert_almost_equal(expected_lrs[epoch], round(lr_scheduler.get_lr()[0], 6))


def test_get_chebyshev_schedule():
    np.testing.assert_almost_equal(get_chebyshev_schedule(3), 1.81818182, decimal=6)
    np.testing.assert_array_equal(chebyshev_perm(5), np.asarray([0, 7, 3, 4, 1, 6, 2, 5]))

    with pytest.raises(IndexError):
        get_chebyshev_schedule(2)
