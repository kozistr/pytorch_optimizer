from typing import List

import numpy as np

from pytorch_optimizer import AdamP, get_chebyshev_schedule
from pytorch_optimizer.lr_scheduler.chebyshev import chebyshev_perm
from pytorch_optimizer.lr_scheduler.cosine_anealing import CosineAnnealingWarmupRestarts
from tests.utils import Example


def test_cosine_annealing_warmup_restarts():
    model = Example()
    optimizer = AdamP(model.parameters())

    def test_scheduler(
        first_cycle_steps: int,
        cycle_mult: float,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        gamma: float,
        max_epochs: int,
        expected_lrs: List[float],
    ):
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

    # case 1
    test_scheduler(
        first_cycle_steps=10,
        cycle_mult=1.0,
        max_lr=1e-3,
        min_lr=1e-6,
        warmup_steps=5,
        gamma=1.0,
        max_epochs=20,
        expected_lrs=[
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
    )

    # case 2
    test_scheduler(
        first_cycle_steps=10,
        cycle_mult=0.9,
        max_lr=1e-3,
        min_lr=1e-6,
        warmup_steps=5,
        gamma=0.5,
        max_epochs=20,
        expected_lrs=[
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
    )


def test_get_chebyshev_schedule():
    np.testing.assert_almost_equal(get_chebyshev_schedule(3), 1.81818182, decimal=6)
    np.testing.assert_array_equal(chebyshev_perm(5), np.asarray([0, 7, 3, 4, 1, 6, 2, 5]))
