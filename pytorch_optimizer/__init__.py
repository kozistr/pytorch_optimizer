# pylint: disable=unused-import
from typing import Dict, List

from pytorch_optimizer.base.types import OPTIMIZER, SCHEDULER
from pytorch_optimizer.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    OneCycleLR,
)
from pytorch_optimizer.lr_scheduler.chebyshev import get_chebyshev_schedule
from pytorch_optimizer.lr_scheduler.cosine_anealing import CosineAnnealingWarmupRestarts
from pytorch_optimizer.lr_scheduler.linear_warmup import CosineScheduler, LinearScheduler, PolyScheduler
from pytorch_optimizer.lr_scheduler.proportion import ProportionScheduler
from pytorch_optimizer.optimizer.adabelief import AdaBelief
from pytorch_optimizer.optimizer.adabound import AdaBound
from pytorch_optimizer.optimizer.adai import Adai
from pytorch_optimizer.optimizer.adamp import AdamP
from pytorch_optimizer.optimizer.adan import Adan
from pytorch_optimizer.optimizer.adapnm import AdaPNM
from pytorch_optimizer.optimizer.agc import agc
from pytorch_optimizer.optimizer.diffgrad import DiffGrad
from pytorch_optimizer.optimizer.diffrgrad import DiffRGrad
from pytorch_optimizer.optimizer.fp16 import DynamicLossScaler, SafeFP16Optimizer
from pytorch_optimizer.optimizer.gc import centralize_gradient
from pytorch_optimizer.optimizer.gsam import GSAM
from pytorch_optimizer.optimizer.lamb import Lamb
from pytorch_optimizer.optimizer.lars import LARS
from pytorch_optimizer.optimizer.lookahead import Lookahead
from pytorch_optimizer.optimizer.madgrad import MADGRAD
from pytorch_optimizer.optimizer.nero import Nero
from pytorch_optimizer.optimizer.pcgrad import PCGrad
from pytorch_optimizer.optimizer.pnm import PNM
from pytorch_optimizer.optimizer.radam import RAdam
from pytorch_optimizer.optimizer.ralamb import RaLamb
from pytorch_optimizer.optimizer.ranger import Ranger
from pytorch_optimizer.optimizer.ranger21 import Ranger21
from pytorch_optimizer.optimizer.sam import SAM
from pytorch_optimizer.optimizer.sgdp import SGDP
from pytorch_optimizer.optimizer.shampoo import Shampoo
from pytorch_optimizer.optimizer.utils import (
    clip_grad_norm,
    disable_running_stats,
    enable_running_stats,
    get_optimizer_parameters,
    matrix_power,
    normalize_gradient,
    unit_norm,
)

OPTIMIZER_LIST: List[OPTIMIZER] = [
    AdaBelief,
    AdaBound,
    AdamP,
    Adai,
    Adan,
    AdaPNM,
    DiffGrad,
    DiffRGrad,
    Lamb,
    LARS,
    MADGRAD,
    Nero,
    PNM,
    RAdam,
    RaLamb,
    Ranger,
    Ranger21,
    SGDP,
    Shampoo,
]
OPTIMIZERS: Dict[str, OPTIMIZER] = {str(optimizer.__name__).lower(): optimizer for optimizer in OPTIMIZER_LIST}

LR_SCHEDULER_LIST: List[SCHEDULER] = [
    CosineAnnealingWarmupRestarts,
    ConstantLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    OneCycleLR,
    CosineScheduler,
    PolyScheduler,
    LinearScheduler,
    ProportionScheduler,
]
LR_SCHEDULERS: Dict[str, SCHEDULER] = {
    str(lr_scheduler.__name__).lower(): lr_scheduler for lr_scheduler in LR_SCHEDULER_LIST
}


def load_optimizer(optimizer: str) -> OPTIMIZER:
    optimizer: str = optimizer.lower()

    if optimizer not in OPTIMIZERS:
        raise NotImplementedError(f'[-] not implemented optimizer : {optimizer}')

    return OPTIMIZERS[optimizer]


def load_lr_scheduler(lr_scheduler: str) -> SCHEDULER:
    lr_scheduler: str = lr_scheduler.lower()

    if lr_scheduler not in LR_SCHEDULERS:
        raise NotImplementedError(f'[-] not implemented lr_scheduler : {lr_scheduler}')

    return LR_SCHEDULERS[lr_scheduler]


def get_supported_optimizers() -> List[OPTIMIZER]:
    return OPTIMIZER_LIST


def get_supported_lr_schedulers() -> List[SCHEDULER]:
    return LR_SCHEDULER_LIST
