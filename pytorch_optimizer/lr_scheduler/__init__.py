# ruff: noqa
import fnmatch
from collections.abc import Sequence
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

from pytorch_optimizer.base.type import SchedulerClass
from pytorch_optimizer.lr_scheduler.chebyshev import get_chebyshev_perm_steps, get_chebyshev_schedule
from pytorch_optimizer.lr_scheduler.cosine_anealing import CosineAnnealingWarmupRestarts
from pytorch_optimizer.lr_scheduler.experimental.deberta_v3_lr_scheduler import deberta_v3_large_lr_scheduler
from pytorch_optimizer.lr_scheduler.linear_warmup import CosineScheduler, LinearScheduler, PolyScheduler
from pytorch_optimizer.lr_scheduler.proportion import ProportionScheduler
from pytorch_optimizer.lr_scheduler.rex import REXScheduler
from pytorch_optimizer.lr_scheduler.wsd import get_wsd_schedule

__all__ = [
    'ConstantLR',
    'CosineAnnealingLR',
    'CosineAnnealingWarmRestarts',
    'CosineAnnealingWarmupRestarts',
    'CosineScheduler',
    'CyclicLR',
    'LinearScheduler',
    'MultiStepLR',
    'MultiplicativeLR',
    'OneCycleLR',
    'PolyScheduler',
    'ProportionScheduler',
    'REXScheduler',
    'StepLR',
    'deberta_v3_large_lr_scheduler',
    'get_chebyshev_perm_steps',
    'get_chebyshev_schedule',
    'get_supported_lr_schedulers',
    'get_wsd_schedule',
    'load_lr_scheduler',
]


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


LR_SCHEDULER_LIST: dict = {
    SchedulerType.CONSTANT: ConstantLR,
    SchedulerType.STEP: StepLR,
    SchedulerType.MULTI_STEP: MultiStepLR,
    SchedulerType.CYCLIC: CyclicLR,
    SchedulerType.MULTIPLICATIVE: MultiplicativeLR,
    SchedulerType.ONE_CYCLE: OneCycleLR,
    SchedulerType.COSINE: CosineScheduler,
    SchedulerType.POLY: PolyScheduler,
    SchedulerType.LINEAR: LinearScheduler,
    SchedulerType.PROPORTION: ProportionScheduler,
    SchedulerType.COSINE_ANNEALING: CosineAnnealingLR,
    SchedulerType.COSINE_ANNEALING_WITH_WARMUP: CosineAnnealingWarmupRestarts,
    SchedulerType.COSINE_ANNEALING_WITH_WARM_RESTART: CosineAnnealingWarmRestarts,
    SchedulerType.CHEBYSHEV: get_chebyshev_schedule,
    SchedulerType.REX: REXScheduler,
    SchedulerType.WARMUP_STABLE_DECAY: get_wsd_schedule,
}
LR_SCHEDULERS: dict[str, SchedulerClass] = {
    str(lr_scheduler_name).lower(): lr_scheduler for lr_scheduler_name, lr_scheduler in LR_SCHEDULER_LIST.items()
}


def load_lr_scheduler(lr_scheduler_name: str) -> SchedulerClass:
    """Return a learning rate scheduler class by name.

    Args:
        lr_scheduler_name: Case insensitive name from `get_supported_lr_schedulers()`.

    Returns:
        Scheduler: Registered scheduler class.

    Raises:
        NotImplementedError: The scheduler name is unsupported.

    """
    lrs_name: str = lr_scheduler_name.lower()

    if lrs_name not in LR_SCHEDULERS:
        raise NotImplementedError(f'not implemented lr_scheduler {lrs_name}')

    return LR_SCHEDULERS[lrs_name]


def get_supported_lr_schedulers(filters: str | list[str] | None = None) -> list[str]:
    """List registered scheduler names in alphabetical order.

    Args:
        filters: Wildcard pattern or list of patterns, such as `'*cosine*'`. `None` selects all names.

    Returns:
        list[str]: Matching names in lowercase, without duplicates.
    """
    if filters is None:
        return sorted(LR_SCHEDULERS.keys())

    include_filters: Sequence[str] = filters if isinstance(filters, (tuple, list)) else [filters]

    filtered_list: set[str] = set()
    for include_filter in include_filters:
        filtered_list.update(fnmatch.filter(LR_SCHEDULERS.keys(), include_filter))

    return sorted(filtered_list)
