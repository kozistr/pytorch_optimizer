import numpy as np
import pytest

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
from tests.fixtures import TrainingModel, make_parameter
from tests.utils import assert_lr_sequence, build_optimizer

CAWR_RECIPES: tuple[tuple, ...] = (
    (1.0, 1.0, [1e-6, 0.000201, 0.000401, 0.0006, 0.0008, 0.001, 0.000905, 0.000655, 0.000346, 9.6e-5] * 2),
    (
        0.9,
        0.5,
        [
            1e-6, 0.000201, 0.000401, 0.0006, 0.0008, 0.001, 0.000905, 0.000655, 0.000346, 9.6e-5,
            1e-6, 0.000101, 0.000201, 0.0003, 0.0004, 0.0005, 0.000427, 0.000251, 7.4e-5, 1e-6,
        ],
    ),
)


LWL_RECIPE = (0.001, 0.0028, 0.0046, 0.0064, 0.0082, 0.01, 0.00802, 0.00604, 0.00406, 0.00208)


LWC_RECIPE = (0.001, 0.0028, 0.0046, 0.0064, 0.0082, 0.01, 0.00905, 0.00658, 0.00352, 0.00105)


LWP_RECIPE = (0.001, 0.0028, 0.0046, 0.0064, 0.0082, 0.01, 0.01, 0.014101, 0.017247, 0.0199)


PROPORTION_LEARNING_RATES: tuple[tuple[float, float, float], ...] = (
    (1e-1, 1e-1, 2.0),
    (1e-1, 1e-3, 1.090909),
)


@pytest.mark.parametrize(('cycle_mult', 'gamma', 'expected_lrs'), CAWR_RECIPES)
def test_cosine_annealing_warmup_restarts(cycle_mult, gamma, expected_lrs, scheduler_optimizer):
    options = {
        'first_cycle_steps': 10,
        'cycle_mult': cycle_mult,
        'max_lr': 1e-3,
        'min_lr': 1e-6,
        'warmup_steps': 5,
        'gamma': gamma,
    }
    lr_scheduler = CosineAnnealingWarmupRestarts(scheduler_optimizer, **options)
    incremental = CosineAnnealingWarmupRestarts(build_optimizer('sgd', [make_parameter()]), **options)

    for epoch, expected_lr in enumerate(expected_lrs):
        lr_scheduler.step(epoch)
        if epoch > 0:
            incremental.step()

        assert lr_scheduler.get_last_lr() == pytest.approx(incremental.get_last_lr())
        assert round(lr_scheduler.get_lr()[0], 6) == pytest.approx(expected_lr, abs=1e-7)


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

    optimizer = build_optimizer('adamw', [make_parameter(grad=None)])
    optimizer.step()

    lr_scheduler = get_chebyshev_schedule(optimizer, num_epochs=16, is_warmup=True)
    lr_scheduler.last_epoch = 0
    optimizer.step()
    lr_scheduler.step()

    np.testing.assert_almost_equal(lr_scheduler.get_last_lr(), 1e-3)

    optimizer = build_optimizer('adamw', [make_parameter(grad=None)])
    optimizer.step()

    lr_scheduler = get_chebyshev_schedule(optimizer, num_epochs=16, is_warmup=False)
    lr_scheduler.last_epoch = 0

    for expected_lr in recipes:
        optimizer.step()
        lr_scheduler.step()
        np.testing.assert_almost_equal(lr_scheduler.get_last_lr(), expected_lr)


@pytest.mark.parametrize(
    ('scheduler', 'expected_lrs', 'decimals'),
    [(LinearScheduler, LWL_RECIPE, 7), (CosineScheduler, LWC_RECIPE, 5), (PolyScheduler, LWP_RECIPE, 6)],
)
def test_linear_warmup_scheduler(scheduler, expected_lrs, decimals, scheduler_optimizer):
    lr_scheduler = scheduler(
        scheduler_optimizer, t_max=10, max_lr=1e-2, min_lr=1e-4, init_lr=1e-3, warmup_steps=5
    )
    assert_lr_sequence(lr_scheduler, expected_lrs, decimals=decimals)


@pytest.mark.parametrize('proportion_learning_rate', PROPORTION_LEARNING_RATES)
def test_proportion_scheduler(proportion_learning_rate: tuple[float, float, float], scheduler_optimizer):
    lr_scheduler = CosineScheduler(
        scheduler_optimizer,
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

    assert_lr_sequence(rho_scheduler, [proportion_learning_rate[2]] * 10, decimals=6)


def test_proportion_no_last_lr_scheduler(scheduler_optimizer):
    lr_scheduler = CosineAnnealingWarmupRestarts(
        scheduler_optimizer,
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

    assert_lr_sequence(rho_scheduler, [2.0] * 10, decimals=6)


def test_rex_lr_scheduler(scheduler_optimizer):
    lrs = [
        0.888888,
        0.749999,
        0.571428,
        0.333333,
        0.0,
    ]

    lr_scheduler = REXScheduler(scheduler_optimizer, total_steps=5, max_lr=1.0, min_lr=0.0)

    assert_lr_sequence(lr_scheduler, lrs, decimals=6)


@pytest.mark.parametrize(
    'recipe',
    [
        ('cosine', [0.0005, 0.001, 0.001, 0.001, 0.000775, 0.000325, 0.0001, 0.0001, 0.0001]),
        ('1-sqrt', [0.0005, 0.001, 0.001, 0.001, 0.0004226, 0.0001835, 0.0001, 0.0001, 0.0001]),
        ('1-square', [0.0005, 0.001, 0.001, 0.001, 0.0008888, 0.0005555, 0.0001, 0.0001, 0.0001]),
        ('linear', [0.0005, 0.001, 0.001, 0.001, 0.0006666, 0.0003333, 0.0001, 0.0001, 0.0001]),
    ],
)
def test_wsd_lr_scheduler(recipe, scheduler_optimizer):
    scheduler_optimizer.step()

    cooldown_type, expected_lrs = recipe

    lr_scheduler = get_wsd_schedule(
        scheduler_optimizer,
        num_warmup_steps=2,
        num_stable_steps=2,
        num_decay_steps=3,
        min_lr_ratio=0.1,
        cooldown_type=cooldown_type,
    )

    assert_lr_sequence(lr_scheduler, expected_lrs, decimals=7)


def test_deberta_v3_large_lr_scheduler():
    groups = deberta_v3_large_lr_scheduler(
        TrainingModel(), layer_low_threshold=1, layer_middle_threshold=2, head_param_start=3
    )

    assert [group['lr'] for group in groups] == pytest.approx([1e-4, 8e-6, 2e-5, 4e-5])
    assert groups[2]['weight_decay'] == 0.0


class TestLRSchedulerParameters:
    def test_cosine_annealing_warmup_restarts_params(self, scheduler_optimizer):
        with pytest.raises(ValueError, match='warmup_steps must be smaller than first_cycle_steps'):
            CosineAnnealingWarmupRestarts(
                optimizer=scheduler_optimizer,
                first_cycle_steps=10,
                warmup_steps=20,
            )

        min_lr: float = 1e-6
        first_cycle_steps: int = 5
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer=scheduler_optimizer,
            min_lr=min_lr,
            first_cycle_steps=first_cycle_steps,
            warmup_steps=0,
        )
        lr_scheduler.step_in_cycle = -1
        expected_max_lr: float = round(lr_scheduler.get_lr()[0], 6)
        np.testing.assert_almost_equal(min_lr, expected_max_lr)

        for _ in range(first_cycle_steps + 1):
            lr_scheduler.step(epoch=None)

    def test_linear_warmup_lr_scheduler_params(self, scheduler_optimizer):
        with pytest.raises(ValueError, match='poly_order must be positive'):
            PolyScheduler(poly_order=-1, optimizer=scheduler_optimizer, t_max=1, max_lr=1)

        with pytest.raises(NegativeLRError):
            PolyScheduler(optimizer=scheduler_optimizer, t_max=1, max_lr=-1)

        with pytest.raises(NegativeLRError):
            PolyScheduler(optimizer=scheduler_optimizer, t_max=1, max_lr=1, min_lr=-1)

        with pytest.raises(NegativeLRError):
            PolyScheduler(optimizer=scheduler_optimizer, t_max=1, max_lr=1, min_lr=1, init_lr=-1)

        with pytest.raises(NegativeStepError):
            PolyScheduler(optimizer=scheduler_optimizer, t_max=-1, max_lr=1, min_lr=1, init_lr=1)

        with pytest.raises(NegativeStepError):
            PolyScheduler(optimizer=scheduler_optimizer, t_max=1, max_lr=1, min_lr=1, init_lr=1, warmup_steps=-1)

    def test_chebyshev_params(self):
        with pytest.raises(IndexError):
            get_chebyshev_perm_steps(0)
