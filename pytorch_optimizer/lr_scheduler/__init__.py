# ruff: noqa
import fnmatch
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Union

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

from pytorch_optimizer.base.types import SCHEDULER
from pytorch_optimizer.lr_scheduler.chebyshev import get_chebyshev_perm_steps, get_chebyshev_schedule
from pytorch_optimizer.lr_scheduler.cosine_anealing import CosineAnnealingWarmupRestarts
from pytorch_optimizer.lr_scheduler.experimental.deberta_v3_lr_scheduler import deberta_v3_large_lr_scheduler
from pytorch_optimizer.lr_scheduler.linear_warmup import CosineScheduler, LinearScheduler, PolyScheduler
from pytorch_optimizer.lr_scheduler.proportion import ProportionScheduler
from pytorch_optimizer.lr_scheduler.rex import REXScheduler
from pytorch_optimizer.lr_scheduler.wsd import get_wsd_schedule


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


LR_SCHEDULER_LIST: Dict = {
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
LR_SCHEDULERS: Dict[str, SCHEDULER] = {
    str(lr_scheduler_name).lower(): lr_scheduler for lr_scheduler_name, lr_scheduler in LR_SCHEDULER_LIST.items()
}


def load_lr_scheduler(lr_scheduler: str) -> SCHEDULER:
    lr_scheduler: str = lr_scheduler.lower()

    if lr_scheduler not in LR_SCHEDULERS:
        raise NotImplementedError(f'[-] not implemented lr_scheduler : {lr_scheduler}')

    return LR_SCHEDULERS[lr_scheduler]


def get_supported_lr_schedulers(filters: Optional[Union[str, List[str]]] = None) -> List[str]:
    r"""Return list of available lr scheduler names, sorted alphabetically.

    :param filters: Optional[Union[str, List[str]]]. wildcard filter string that works with fmatch. if None, it will
        return the whole list.
    """
    if filters is None:
        return sorted(LR_SCHEDULERS.keys())

    include_filters: Sequence[str] = filters if isinstance(filters, (tuple, list)) else [filters]

    filtered_list: Set[str] = set()
    for include_filter in include_filters:
        filtered_list.update(fnmatch.filter(LR_SCHEDULERS.keys(), include_filter))

    return sorted(filtered_list)
