# pylint: disable=unused-import
from typing import Dict, List, Type

from torch.optim import Optimizer

from pytorch_optimizer.optimizer.adabelief import AdaBelief
from pytorch_optimizer.optimizer.adabound import AdaBound
from pytorch_optimizer.optimizer.adamp import AdamP
from pytorch_optimizer.optimizer.adan import Adan
from pytorch_optimizer.optimizer.adapnm import AdaPNM
from pytorch_optimizer.optimizer.agc import agc
from pytorch_optimizer.optimizer.chebyshev_schedule import get_chebyshev_schedule
from pytorch_optimizer.optimizer.diffgrad import DiffGrad
from pytorch_optimizer.optimizer.diffrgrad import DiffRGrad
from pytorch_optimizer.optimizer.fp16 import DynamicLossScaler, SafeFP16Optimizer
from pytorch_optimizer.optimizer.gc import centralize_gradient
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
    get_optimizer_parameters,
    matrix_power,
    normalize_gradient,
    unit_norm,
)

OPTIMIZER_LIST: List[Type[Optimizer]] = [
    AdaBelief,
    AdaBound,
    AdamP,
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
OPTIMIZERS: Dict[str, Type[Optimizer]] = {str(optimizer.__name__).lower(): optimizer for optimizer in OPTIMIZER_LIST}


def load_optimizer(optimizer: str) -> Type[Optimizer]:
    optimizer: str = optimizer.lower()

    if optimizer not in OPTIMIZERS:
        raise NotImplementedError(f'[-] not implemented optimizer : {optimizer}')

    return OPTIMIZERS[optimizer]


def get_supported_optimizers() -> List[Type[Optimizer]]:
    return OPTIMIZER_LIST
