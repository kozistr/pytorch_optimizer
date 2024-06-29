# ruff: noqa
from enum import Enum

from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    MultiplicativeLR,
    MultiStepLR,
    OneCycleLR,
    StepLR,
)


class SchedulerType(Enum):
    CONSTANT = 'constant'
    LINEAR = 'linear'
    PROPORTION = 'proportion'
    STEP = 'step'
    MULTI_STEP = 'multi_step'
    MULTIPLICATIVE = 'multiplicative'
    CYCLIC = 'cyclic'
    ONE_CYCLE = 'one_cycle'
    COSINE = 'cosine'
    POLY = 'poly'
    COSINE_ANNEALING = 'cosine_annealing'
    COSINE_ANNEALING_WITH_WARM_RESTART = 'cosine_annealing_with_warm_restart'
    COSINE_ANNEALING_WITH_WARMUP = 'cosine_annealing_with_warmup'
    CHEBYSHEV = 'chebyshev'
    REX = 'rex'
    WARMUP_STABLE_DECAY = 'warmup_stable_decay'

    def __str__(self) -> str:
        return self.value
