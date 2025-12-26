from typing import Tuple

import numpy as np
import pytest
from torch import nn

from pytorch_optimizer.base.exception import NegativeLRError, NegativeStepError
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
from tests.constants import CAWR_RECIPES, LWC_RECIPE, LWL_RECIPE, LWP_RECIPE, PROPORTION_LEARNING_RATES
from tests.utils import Example, LRSchedulerAssertions


@pytest.mark.parametrize('cosine_annealing_warmup_restart_param', CAWR_RECIPES)
def test_cosine_annealing_warmup_restarts(cosine_annealing_warmup_restart_param, optimizer_factory):
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
        optimizer=optimizer_factory,
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


class TestWarmupSchedulers:

    def test_linear_warmup_linear_scheduler(self, optimizer_factory):
        lr_scheduler = LinearScheduler(
            optimizer_factory,
            t_max=10,
            max_lr=1e-2,
            min_lr=1e-4,
            init_lr=1e-3,
            warmup_steps=5,
        )
        LRSchedulerAssertions.assert_lr_sequence(lr_scheduler, LWL_RECIPE)

    def test_linear_warmup_cosine_scheduler(self, optimizer_factory):
        lr_scheduler = CosineScheduler(
            optimizer_factory,
            t_max=10,
            max_lr=1e-2,
            min_lr=1e-4,
            init_lr=1e-3,
            warmup_steps=5,
        )
        LRSchedulerAssertions.assert_lr_sequence(lr_scheduler, LWC_RECIPE, decimals=5)

    def test_linear_warmup_poly_scheduler(self, optimizer_factory):
        lr_scheduler = PolyScheduler(
            optimizer_factory,
            t_max=10,
            max_lr=1e-2,
            min_lr=1e-4,
            init_lr=1e-3,
            warmup_steps=5,
        )
        LRSchedulerAssertions.assert_lr_sequence(lr_scheduler, LWP_RECIPE, decimals=6)


@pytest.mark.parametrize('proportion_learning_rate', PROPORTION_LEARNING_RATES)
def test_proportion_scheduler(proportion_learning_rate: Tuple[float, float, float], optimizer_factory):
    lr_scheduler = CosineScheduler(
        optimizer_factory,
        t_max=10,
        max_lr=proportion_learning_rate[0],
        min_lr=proportion_learning_rate[1],
        init_lr=1e-2,
    )

    rho_scheduler = ProportionScheduler(
        lr_scheduler,
        max_lr=proportion_learning_rate[0],
        min_lr=proportion_learning_rate[1],
        max_value=2.0,
        min_value=1.0,
    )

    LRSchedulerAssertions.assert_lr_sequence(rho_scheduler, [proportion_learning_rate[2]] * 10, decimals=6)


def test_proportion_no_last_lr_scheduler(optimizer_factory):
    lr_scheduler = CosineAnnealingWarmupRestarts(
        optimizer_factory,
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

    LRSchedulerAssertions.assert_lr_sequence(rho_scheduler, [2.0] * 10, decimals=6)


def test_rex_lr_scheduler(optimizer_factory):
    lrs = [
        0.888888,
        0.749999,
        0.571428,
        0.333333,
        0.0,
    ]

    lr_scheduler = REXScheduler(optimizer_factory, total_steps=5, max_lr=1.0, min_lr=0.0)

    LRSchedulerAssertions.assert_lr_sequence(lr_scheduler, lrs, decimals=6)


@pytest.mark.parametrize(
    'recipe',
    [
        ('cosine', [0.0005, 0.001, 0.001, 0.001, 0.000775, 0.000325, 0.0001, 0.0001, 0.0001]),
        ('1-sqrt', [0.0005, 0.001, 0.001, 0.001, 0.0004226, 0.0001835, 0.0001, 0.0001, 0.0001]),
        ('1-square', [0.0005, 0.001, 0.001, 0.001, 0.0008888, 0.0005555, 0.0001, 0.0001, 0.0001]),
        ('linear', [0.0005, 0.001, 0.001, 0.001, 0.0006666, 0.0003333, 0.0001, 0.0001, 0.0001]),
    ],
)
def test_wsd_lr_scheduler(recipe, optimizer_factory):
    optimizer_factory.step()

    cooldown_type, expected_lrs = recipe

    lr_scheduler = get_wsd_schedule(
        optimizer_factory,
        num_warmup_steps=2,
        num_stable_steps=2,
        num_decay_steps=3,
        min_lr_ratio=0.1,
        cooldown_type=cooldown_type,
    )

    LRSchedulerAssertions.assert_lr_sequence(lr_scheduler, expected_lrs, decimals=7)


def test_deberta_v3_large_lr_scheduler():
    model = nn.Sequential(*[nn.Linear(1, 1, bias=False) for _ in range(400)])
    deberta_v3_large_lr_scheduler(model)


class TestLRSchedulerParameters:

    def test_cosine_annealing_warmup_restarts_params(self, optimizer_factory):
        with pytest.raises(ValueError) as error_info:
            CosineAnnealingWarmupRestarts(
                optimizer=optimizer_factory,
                first_cycle_steps=10,
                warmup_steps=20,
            )

        assert str(error_info.value) == 'warmup_steps must be smaller than first_cycle_steps. 20 < 10'

        min_lr: float = 1e-6
        first_cycle_steps: int = 5
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer=optimizer_factory,
            min_lr=min_lr,
            first_cycle_steps=first_cycle_steps,
            warmup_steps=0,
        )
        lr_scheduler.step_in_cycle = -1
        expected_max_lr: float = round(lr_scheduler.get_lr()[0], 6)
        np.testing.assert_almost_equal(min_lr, expected_max_lr)

        for _ in range(first_cycle_steps + 1):
            lr_scheduler.step(epoch=None)

    def test_linear_warmup_lr_scheduler_params(self, optimizer_factory):
        with pytest.raises(ValueError) as error_info:
            PolyScheduler(poly_order=-1, optimizer=optimizer_factory, t_max=1, max_lr=1)

        assert str(error_info.value) == 'poly_order must be positive. -1'

        with pytest.raises(NegativeLRError):
            PolyScheduler(optimizer=optimizer_factory, t_max=1, max_lr=-1)

        with pytest.raises(NegativeLRError):
            PolyScheduler(optimizer=optimizer_factory, t_max=1, max_lr=1, min_lr=-1)

        with pytest.raises(NegativeLRError):
            PolyScheduler(optimizer=optimizer_factory, t_max=1, max_lr=1, min_lr=1, init_lr=-1)

        with pytest.raises(NegativeStepError):
            PolyScheduler(optimizer=optimizer_factory, t_max=-1, max_lr=1, min_lr=1, init_lr=1)

        with pytest.raises(NegativeStepError):
            PolyScheduler(optimizer=optimizer_factory, t_max=1, max_lr=1, min_lr=1, init_lr=1, warmup_steps=-1)

    def test_chebyshev_params(self):
        with pytest.raises(IndexError):
            get_chebyshev_perm_steps(0)
