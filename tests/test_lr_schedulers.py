from typing import Tuple

import numpy as np
import pytest
from torch import nn

from pytorch_optimizer.lr_scheduler.chebyshev import (
    get_chebyshev_perm_steps,
    get_chebyshev_permutation,
    get_chebyshev_schedule,
)
from pytorch_optimizer.lr_scheduler.cosine_anealing import CosineAnnealingWarmupRestarts
from pytorch_optimizer.lr_scheduler.experimental.deberta_v3_lr_scheduler import deberta_v3_large_lr_scheduler
from pytorch_optimizer.lr_scheduler.linear_warmup import CosineScheduler, LinearScheduler, PolyScheduler
from pytorch_optimizer.lr_scheduler.proportion import ProportionScheduler
from pytorch_optimizer.lr_scheduler.rex import REXScheduler
from pytorch_optimizer.lr_scheduler.wsd import get_wsd_schedule
from pytorch_optimizer.optimizer import AdamW
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
LWL_RECIPE = [
    0.001,
    0.0028,
    0.0046,
    0.0064,
    0.0082,
    0.01,
    0.00802,
    0.00604,
    0.00406,
    0.00208,
]
LWC_RECIPE = [
    0.001,
    0.00280,
    0.00460,
    0.00640,
    0.00820,
    0.01000,
    0.00905,
    0.00658,
    0.00352,
    0.00105,
]
LWP_RECIPE = [
    0.001,
    0.002800,
    0.004600,
    0.006400,
    0.008200,
    0.010000,
    0.010000,
    0.014101,
    0.017247,
    0.019900,
]
PROPORTION_LEARNING_RATES = [(1e-1, 1e-1, 2.0), (1e-1, 1e-3, 1.090909)]


@pytest.mark.parametrize('cosine_annealing_warmup_restart_param', CAWR_RECIPES)
def test_cosine_annealing_warmup_restarts(cosine_annealing_warmup_restart_param):
    model = Example()
    optimizer = AdamW(model.parameters())

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


def test_get_chebyshev_scheduler():
    # test the first nontrivial permutations sigma_{T}
    recipes = {
        2: np.asarray([0, 1]),
        4: np.asarray([0, 3, 1, 2]),
        8: np.asarray([0, 7, 3, 4, 1, 6, 2, 5]),
        16: np.asarray([0, 15, 7, 8, 3, 12, 4, 11, 1, 14, 6, 9, 2, 13, 5, 10]),
    }

    for k, v in recipes.items():
        np.testing.assert_array_equal(get_chebyshev_permutation(k), v)

    np.testing.assert_almost_equal(get_chebyshev_perm_steps(1), 1.904762, decimal=6)
    np.testing.assert_almost_equal(get_chebyshev_perm_steps(3), 8.799878, decimal=6)


def test_get_chebyshev_lr():
    recipes = [
        0.019125119558059765,
        0.019125119558059765,
        0.0010022924983586518,
        0.0020901181252459123,
        0.0017496032811320122,
        0.006336331139456458,
        0.0011208500962143087,
        0.004471008393917827,
        0.0012101602977446309,
        0.014193791132074378,
        0.0010208804147606497,
        0.0025832131864890117,
        0.0015085567867114075,
        0.009426190153875151,
        0.0010594201194061095,
        0.0033213041232648503,
        0.001335267780289186,
        0.001335267780289186,
        0.001335267780289186,
    ]

    optimizer = AdamW(Example().parameters())
    optimizer.step()

    lr_scheduler = get_chebyshev_schedule(optimizer, num_epochs=16, is_warmup=True)
    lr_scheduler.last_epoch = 0
    lr_scheduler.step()

    np.testing.assert_almost_equal(lr_scheduler.get_last_lr(), 1e-3)

    optimizer = AdamW(Example().parameters())
    optimizer.step()

    lr_scheduler = get_chebyshev_schedule(optimizer, num_epochs=16, is_warmup=False)
    lr_scheduler.last_epoch = 0

    for expected_lr in recipes:
        lr_scheduler.step()
        np.testing.assert_almost_equal(lr_scheduler.get_last_lr(), expected_lr)


def test_linear_warmup_linear_scheduler():
    optimizer = AdamW(Example().parameters())

    lr_scheduler = LinearScheduler(optimizer, t_max=10, max_lr=1e-2, min_lr=1e-4, init_lr=1e-3, warmup_steps=5)

    for expected_lr in LWL_RECIPE:
        lr_scheduler.step()
        np.testing.assert_almost_equal(expected_lr, lr_scheduler.get_lr())


def test_linear_warmup_cosine_scheduler():
    optimizer = AdamW(Example().parameters())
    lr_scheduler = CosineScheduler(optimizer, t_max=10, max_lr=1e-2, min_lr=1e-4, init_lr=1e-3, warmup_steps=5)

    for expected_lr in LWC_RECIPE:
        lr_scheduler.step()
        np.testing.assert_almost_equal(expected_lr, lr_scheduler.get_lr(), 5)


def test_linear_warmup_poly_scheduler():
    optimizer = AdamW(Example().parameters())
    lr_scheduler = PolyScheduler(optimizer=optimizer, t_max=10, max_lr=1e-2, min_lr=1e-4, init_lr=1e-3, warmup_steps=5)

    for expected_lr in LWP_RECIPE:
        lr_scheduler.step()
        np.testing.assert_almost_equal(expected_lr, lr_scheduler.get_lr(), 6)


@pytest.mark.parametrize('proportion_learning_rate', PROPORTION_LEARNING_RATES)
def test_proportion_scheduler(proportion_learning_rate: Tuple[float, float, float]):
    base_optimizer = AdamW(Example().parameters())
    lr_scheduler = CosineScheduler(
        base_optimizer, t_max=10, max_lr=proportion_learning_rate[0], min_lr=proportion_learning_rate[1], init_lr=1e-2
    )
    rho_scheduler = ProportionScheduler(
        lr_scheduler,
        max_lr=proportion_learning_rate[0],
        min_lr=proportion_learning_rate[1],
        max_value=2.0,
        min_value=1.0,
    )

    for _ in range(10):
        _ = rho_scheduler.step()
        np.testing.assert_almost_equal(proportion_learning_rate[2], rho_scheduler.get_lr(), 6)


def test_proportion_no_last_lr_scheduler():
    base_optimizer = AdamW(Example().parameters())
    lr_scheduler = CosineAnnealingWarmupRestarts(
        base_optimizer,
        first_cycle_steps=10,
        max_lr=1e-2,
        min_lr=1e-2,
    )
    rho_scheduler = ProportionScheduler(
        lr_scheduler,
        max_lr=1e-2,
        min_lr=1e-2,
        max_value=2.0,
        min_value=1.0,
    )

    for _ in range(10):
        _ = rho_scheduler.step()
        np.testing.assert_almost_equal(2.0, rho_scheduler.get_lr(), 6)


def test_rex_lr_scheduler():
    lrs = [
        0.888888,
        0.749999,
        0.571428,
        0.333333,
        0.0,
    ]

    base_optimizer = AdamW(Example().parameters())

    lr_scheduler = REXScheduler(
        base_optimizer,
        total_steps=5,
        max_lr=1.0,
        min_lr=0.0,
    )

    for expected_lr in lrs:
        _ = lr_scheduler.step()
        np.testing.assert_almost_equal(expected_lr, lr_scheduler.get_lr(), 6)


@pytest.mark.parametrize(
    'recipe',
    [
        ('cosine', [0.0005, 0.001, 0.001, 0.001, 0.000775, 0.000325, 0.0001, 0.0001, 0.0001]),
        ('1-sqrt', [0.0005, 0.001, 0.001, 0.001, 0.0004226, 0.0001835, 0.0001, 0.0001, 0.0001]),
        ('1-square', [0.0005, 0.001, 0.001, 0.001, 0.0008888, 0.0005555, 0.0001, 0.0001, 0.0001]),
        ('linear', [0.0005, 0.001, 0.001, 0.001, 0.0006666, 0.0003333, 0.0001, 0.0001, 0.0001]),
    ],
)
def test_wsd_lr_scheduler(recipe):
    optimizer = AdamW(Example().parameters())
    optimizer.step()

    cooldown_type, expected_lrs = recipe

    lr_scheduler = get_wsd_schedule(optimizer, 2, 2, 3, min_lr_ratio=0.1, cooldown_type=cooldown_type)

    for expected_lr in expected_lrs:
        lr_scheduler.step()
        np.testing.assert_almost_equal(expected_lr, lr_scheduler.get_last_lr()[0], 7)


def test_deberta_v3_large_lr_scheduler():
    model = nn.Sequential(*[nn.Linear(1, 1, bias=False) for _ in range(400)])
    deberta_v3_large_lr_scheduler(model)
