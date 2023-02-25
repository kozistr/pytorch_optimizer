from typing import Tuple

import numpy as np
import pytest
from torch import nn

from pytorch_optimizer import AdamP, deberta_v3_large_lr_scheduler, get_chebyshev_schedule
from pytorch_optimizer.lr_scheduler.chebyshev import chebyshev_perm
from pytorch_optimizer.lr_scheduler.cosine_anealing import CosineAnnealingWarmupRestarts
from pytorch_optimizer.lr_scheduler.linear_warmup import CosineScheduler, LinearScheduler, PolyScheduler
from pytorch_optimizer.lr_scheduler.proportion import ProportionScheduler
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


@pytest.mark.lr_scheduler
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


@pytest.mark.lr_scheduler
def test_get_chebyshev_scheduler():
    np.testing.assert_almost_equal(get_chebyshev_schedule(3), 1.81818182, decimal=6)
    np.testing.assert_array_equal(chebyshev_perm(5), np.asarray([0, 7, 3, 4, 1, 6, 2, 5]))


@pytest.mark.lr_scheduler
def test_linear_warmup_linear_scheduler():
    optimizer = AdamP(Example().parameters())
    lr_scheduler = LinearScheduler(optimizer, t_max=10, max_lr=1e-2, min_lr=1e-4, init_lr=1e-3, warmup_steps=5)

    for expected_lr in LWL_RECIPE:
        lr_scheduler.step()
        np.testing.assert_almost_equal(expected_lr, lr_scheduler.get_lr())


@pytest.mark.lr_scheduler
def test_linear_warmup_cosine_scheduler():
    optimizer = AdamP(Example().parameters())
    lr_scheduler = CosineScheduler(optimizer, t_max=10, max_lr=1e-2, min_lr=1e-4, init_lr=1e-3, warmup_steps=5)

    for expected_lr in LWC_RECIPE:
        lr_scheduler.step()
        np.testing.assert_almost_equal(expected_lr, lr_scheduler.get_lr(), 5)


@pytest.mark.lr_scheduler
def test_linear_warmup_poly_scheduler():
    optimizer = AdamP(Example().parameters())
    lr_scheduler = PolyScheduler(optimizer=optimizer, t_max=10, max_lr=1e-2, min_lr=1e-4, init_lr=1e-3, warmup_steps=5)

    for expected_lr in LWP_RECIPE:
        lr_scheduler.step()
        np.testing.assert_almost_equal(expected_lr, lr_scheduler.get_lr(), 6)


@pytest.mark.lr_scheduler
@pytest.mark.parametrize('proportion_learning_rate', PROPORTION_LEARNING_RATES)
def test_proportion_scheduler(proportion_learning_rate: Tuple[float, float, float]):
    base_optimizer = AdamP(Example().parameters())
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


@pytest.mark.lr_scheduler
def test_proportion_no_last_lr_scheduler():
    base_optimizer = AdamP(Example().parameters())
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


@pytest.mark.lr_scheduler
def test_deberta_v3_large_lr_scheduler():
    try:
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained('microsoft/deberta-v3-large', pretrained=False)
        model = AutoModel.from_config(config)
    except ImportError:
        model = nn.Sequential(*[nn.Linear(1, 1, bias=False) for _ in range(400)])

    deberta_v3_large_lr_scheduler(model)
