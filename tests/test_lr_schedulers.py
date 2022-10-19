import numpy as np
import pytest

from pytorch_optimizer import AdamP, get_chebyshev_schedule
from pytorch_optimizer.lr_scheduler.chebyshev import chebyshev_perm
from pytorch_optimizer.lr_scheduler.cosine_anealing import CosineAnnealingWarmupRestarts
from tests.utils import Example

CAWR_RECIPES = [
    (
        10,
        1.0,
        1e-3,
        1e-6,
        5,
        1.0,
        20,
        [
            1e-06,
            0.000201,
            0.000401,
            0.0006,
            0.0008,
            0.001,
            0.000905,
            0.000655,
            0.000346,
            9.6e-05,
            1e-06,
            0.000201,
            0.000401,
            0.0006,
            0.0008,
            0.001,
            0.000905,
            0.000655,
            0.000346,
            9.6e-05,
        ],
    ),
    (
        10,
        0.9,
        1e-3,
        1e-6,
        5,
        0.5,
        20,
        [
            1e-06,
            0.000201,
            0.000401,
            0.0006,
            0.0008,
            0.001,
            0.000905,
            0.000655,
            0.000346,
            9.6e-05,
            1e-6,
            0.000101,
            0.000201,
            0.0003,
            0.0004,
            0.0005,
            0.000427,
            0.000251,
            7.4e-05,
            1e-06,
        ],
    ),
]


@pytest.mark.parametrize('cosine_annealing_warmup_restart_param', CAWR_RECIPES)
def test_cosine_annealing_warmup_restarts(cosine_annealing_warmup_restart_param):
    model = Example()
    optimizer = AdamP(model.parameters())

    (
        first_cycle_steps,
        cycle_mult,
        max_lr,
        min_lr,
        warmup_steps,
        gamma,
        max_epochs,
        expected_lrs,
    ) = cosine_annealing_warmup_restart_param

    lr_scheduler = CosineAnnealingWarmupRestarts(
        optimizer=optimizer,
        first_cycle_steps=first_cycle_steps,
        cycle_mult=cycle_mult,
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        gamma=gamma,
    )

    if warmup_steps > 0:
        np.testing.assert_almost_equal(min_lr, round(lr_scheduler.get_lr()[0], 6))

    for epoch in range(max_epochs):
        lr_scheduler.step(epoch)

        lr: float = round(lr_scheduler.get_lr()[0], 6)
        np.testing.assert_almost_equal(expected_lrs[epoch], lr)


def test_get_chebyshev_schedule():
    np.testing.assert_almost_equal(get_chebyshev_schedule(3), 1.81818182, decimal=6)
    np.testing.assert_array_equal(chebyshev_perm(5), np.asarray([0, 7, 3, 4, 1, 6, 2, 5]))
