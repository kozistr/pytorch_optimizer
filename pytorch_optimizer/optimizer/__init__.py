import fnmatch
from importlib.util import find_spec
from typing import Dict, List, Optional, Sequence, Set, Union

import torch
from torch import nn
from torch.optim import AdamW

from pytorch_optimizer.base.types import OPTIMIZER, PARAMETERS
from pytorch_optimizer.optimizer.a2grad import A2Grad
from pytorch_optimizer.optimizer.adabelief import AdaBelief
from pytorch_optimizer.optimizer.adabound import AdaBound
from pytorch_optimizer.optimizer.adadelta import AdaDelta
from pytorch_optimizer.optimizer.adafactor import AdaFactor
from pytorch_optimizer.optimizer.adahessian import AdaHessian
from pytorch_optimizer.optimizer.adai import Adai
from pytorch_optimizer.optimizer.adalite import Adalite
from pytorch_optimizer.optimizer.adam_mini import AdamMini
from pytorch_optimizer.optimizer.adamax import AdaMax
from pytorch_optimizer.optimizer.adamg import AdamG
from pytorch_optimizer.optimizer.adamod import AdaMod
from pytorch_optimizer.optimizer.adamp import AdamP
from pytorch_optimizer.optimizer.adams import AdamS
from pytorch_optimizer.optimizer.adamw import StableAdamW
from pytorch_optimizer.optimizer.adan import Adan
from pytorch_optimizer.optimizer.adanorm import AdaNorm
from pytorch_optimizer.optimizer.adapnm import AdaPNM
from pytorch_optimizer.optimizer.adashift import AdaShift
from pytorch_optimizer.optimizer.adasmooth import AdaSmooth
from pytorch_optimizer.optimizer.ademamix import AdEMAMix
from pytorch_optimizer.optimizer.adopt import ADOPT
from pytorch_optimizer.optimizer.agc import agc
from pytorch_optimizer.optimizer.aggmo import AggMo
from pytorch_optimizer.optimizer.aida import Aida
from pytorch_optimizer.optimizer.alig import AliG
from pytorch_optimizer.optimizer.amos import Amos
from pytorch_optimizer.optimizer.apollo import APOLLO, ApolloDQN
from pytorch_optimizer.optimizer.avagrad import AvaGrad
from pytorch_optimizer.optimizer.came import CAME
from pytorch_optimizer.optimizer.dadapt import DAdaptAdaGrad, DAdaptAdam, DAdaptAdan, DAdaptLion, DAdaptSGD
from pytorch_optimizer.optimizer.demo import DeMo
from pytorch_optimizer.optimizer.diffgrad import DiffGrad
from pytorch_optimizer.optimizer.experimental.ranger25 import Ranger25
from pytorch_optimizer.optimizer.fadam import FAdam
from pytorch_optimizer.optimizer.fp16 import DynamicLossScaler, SafeFP16Optimizer
from pytorch_optimizer.optimizer.fromage import Fromage
from pytorch_optimizer.optimizer.ftrl import FTRL
from pytorch_optimizer.optimizer.galore import GaLore
from pytorch_optimizer.optimizer.gc import centralize_gradient
from pytorch_optimizer.optimizer.grams import Grams
from pytorch_optimizer.optimizer.gravity import Gravity
from pytorch_optimizer.optimizer.grokfast import GrokFastAdamW
from pytorch_optimizer.optimizer.kate import Kate
from pytorch_optimizer.optimizer.lamb import Lamb
from pytorch_optimizer.optimizer.laprop import LaProp
from pytorch_optimizer.optimizer.lars import LARS
from pytorch_optimizer.optimizer.lion import Lion
from pytorch_optimizer.optimizer.lomo import LOMO, AdaLOMO
from pytorch_optimizer.optimizer.lookahead import Lookahead
from pytorch_optimizer.optimizer.madgrad import MADGRAD
from pytorch_optimizer.optimizer.mars import MARS
from pytorch_optimizer.optimizer.msvag import MSVAG
from pytorch_optimizer.optimizer.muon import Muon
from pytorch_optimizer.optimizer.nero import Nero
from pytorch_optimizer.optimizer.novograd import NovoGrad
from pytorch_optimizer.optimizer.orthograd import OrthoGrad
from pytorch_optimizer.optimizer.padam import PAdam
from pytorch_optimizer.optimizer.pcgrad import PCGrad
from pytorch_optimizer.optimizer.pid import PID
from pytorch_optimizer.optimizer.pnm import PNM
from pytorch_optimizer.optimizer.prodigy import Prodigy
from pytorch_optimizer.optimizer.qhadam import QHAdam
from pytorch_optimizer.optimizer.qhm import QHM
from pytorch_optimizer.optimizer.radam import RAdam
from pytorch_optimizer.optimizer.ranger import Ranger
from pytorch_optimizer.optimizer.ranger21 import Ranger21
from pytorch_optimizer.optimizer.rotograd import RotoGrad
from pytorch_optimizer.optimizer.sam import BSAM, GSAM, SAM, WSAM
from pytorch_optimizer.optimizer.schedulefree import ScheduleFreeAdamW, ScheduleFreeRAdam, ScheduleFreeSGD
from pytorch_optimizer.optimizer.sgd import ASGD, SGDW, AccSGD, SGDSaI, SignSGD
from pytorch_optimizer.optimizer.sgdp import SGDP
from pytorch_optimizer.optimizer.shampoo import ScalableShampoo, Shampoo
from pytorch_optimizer.optimizer.sm3 import SM3
from pytorch_optimizer.optimizer.soap import SOAP
from pytorch_optimizer.optimizer.sophia import SophiaH
from pytorch_optimizer.optimizer.srmm import SRMM
from pytorch_optimizer.optimizer.swats import SWATS
from pytorch_optimizer.optimizer.tiger import Tiger
from pytorch_optimizer.optimizer.trac import TRAC
from pytorch_optimizer.optimizer.yogi import Yogi

HAS_BNB: bool = find_spec('bitsandbytes') is not None
HAS_Q_GALORE: bool = find_spec('q-galore-torch') is not None
HAS_TORCHAO: bool = find_spec('torchao') is not None


def load_bnb_optimizer(optimizer: str) -> OPTIMIZER:  # pragma: no cover  # noqa: PLR0911
    r"""Load bnb optimizer instance."""
    from bitsandbytes import optim

    if 'sgd8bit' in optimizer:
        return optim.SGD8bit
    if 'adam8bit' in optimizer:
        return optim.Adam8bit
    if 'paged_adam8bit' in optimizer:
        return optim.PagedAdam8bit
    if 'adamw8bit' in optimizer:
        return optim.AdamW8bit
    if 'paged_adamw8bit' in optimizer:
        return optim.PagedAdamW8bit
    if 'lamb8bit' in optimizer:
        return optim.LAMB8bit
    if 'lars8bit' in optimizer:
        return optim.LARS8bit
    if 'lion8bit' in optimizer:
        return optim.Lion8bit
    if 'adagrad8bit' in optimizer:
        return optim.Adagrad8bit
    if 'rmsprop8bit' in optimizer:
        return optim.RMSprop8bit
    if 'adagrad32bit' in optimizer:
        return optim.Adagrad32bit
    if 'adam32bit' in optimizer:
        return optim.Adam32bit
    if 'paged_adam32bit' in optimizer:
        return optim.PagedAdam32bit
    if 'adamw32bit' in optimizer:
        return optim.AdamW32bit
    if 'lamb32bit' in optimizer:
        return optim.LAMB32bit
    if 'lars32bit' in optimizer:
        return optim.LARS32bit
    if 'lion32bit' in optimizer:
        return optim.Lion32bit
    if 'paged_lion32bit' in optimizer:
        return optim.PagedLion32bit
    if 'rmsprop32bit' in optimizer:
        return optim.RMSprop32bit
    if 'sgd32bit' in optimizer:
        return optim.SGD32bit
    if 'ademamix8bit' in optimizer:
        return optim.AdEMAMix8bit
    if 'ademamix32bit' in optimizer:
        return optim.AdEMAMix32bit
    if 'paged_ademamix8bit' in optimizer:
        return optim.PagedAdEMAMix8bit
    if 'paged_ademamix32bit' in optimizer:
        return optim.PagedAdEMAMix32bit

    raise NotImplementedError(f'[-] not implemented optimizer : {optimizer}')


def load_q_galore_optimizer(optimizer: str) -> OPTIMIZER:  # pragma: no cover
    r"""Load Q-GaLore optimizer instance."""
    import q_galore_torch

    if 'adamw8bit' in optimizer:
        return q_galore_torch.QGaLoreAdamW8bit

    raise NotImplementedError(f'[-] not implemented optimizer : {optimizer}')


def load_ao_optimizer(optimizer: str) -> OPTIMIZER:  # pragma: no cover
    r"""Load TorchAO optimizer instance."""
    from torchao.prototype import low_bit_optim

    if 'adamw8bit' in optimizer:
        return low_bit_optim.AdamW8bit
    if 'adamw4bit' in optimizer:
        return low_bit_optim.AdamW4bit
    if 'adamwfp8' in optimizer:
        return low_bit_optim.AdamWFp8

    raise NotImplementedError(f'[-] not implemented optimizer : {optimizer}')


def load_optimizer(optimizer: str) -> OPTIMIZER:
    r"""Load optimizers."""
    optimizer: str = optimizer.lower()

    if optimizer.startswith('bnb'):
        if HAS_BNB and torch.cuda.is_available():
            return load_bnb_optimizer(optimizer)  # pragma: no cover
        raise ImportError(f'bitsandbytes and CUDA required for the optimizer {optimizer}')
    if optimizer.startswith('q_galore'):
        if HAS_Q_GALORE and torch.cuda.is_available():
            return load_q_galore_optimizer(optimizer)  # pragma: no cover
        raise ImportError(f'bitsandbytes, q-galore-torch, and CUDA required for the optimizer {optimizer}')
    if optimizer.startswith('torchao'):
        if HAS_TORCHAO and torch.cuda.is_available():
            return load_ao_optimizer(optimizer)  # pragma: no cover
        raise ImportError(
            f'torchao required for the optimizer {optimizer}. '
            'usage: https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim#usage'
        )
    if optimizer not in OPTIMIZERS:
        raise NotImplementedError(f'not implemented optimizer : {optimizer}')

    return OPTIMIZERS[optimizer]


OPTIMIZER_LIST: List[OPTIMIZER] = [
    AdamW,
    AdaBelief,
    AdaBound,
    PID,
    AdamP,
    Adai,
    Adan,
    AdaMod,
    AdaPNM,
    DiffGrad,
    Lamb,
    LARS,
    QHAdam,
    QHM,
    MADGRAD,
    Nero,
    PNM,
    MSVAG,
    RAdam,
    Ranger,
    Ranger21,
    SGDP,
    Shampoo,
    ScalableShampoo,
    DAdaptAdaGrad,
    Fromage,
    AggMo,
    DAdaptAdam,
    DAdaptSGD,
    DAdaptAdan,
    AdamS,
    AdaFactor,
    ApolloDQN,
    APOLLO,
    SWATS,
    NovoGrad,
    Lion,
    AliG,
    SM3,
    AdaNorm,
    A2Grad,
    AccSGD,
    SGDW,
    Yogi,
    ASGD,
    AdaMax,
    Gravity,
    AdaSmooth,
    SRMM,
    AvaGrad,
    AdaShift,
    AdaDelta,
    Amos,
    AdaHessian,
    SophiaH,
    SignSGD,
    Prodigy,
    PAdam,
    LOMO,
    Tiger,
    CAME,
    DAdaptLion,
    Aida,
    GaLore,
    Adalite,
    BSAM,
    ScheduleFreeSGD,
    ScheduleFreeAdamW,
    FAdam,
    GrokFastAdamW,
    Kate,
    StableAdamW,
    AdamMini,
    AdaLOMO,
    AdamG,
    AdEMAMix,
    SOAP,
    ADOPT,
    FTRL,
    DeMo,
    Muon,
    ScheduleFreeRAdam,
    LaProp,
    MARS,
    SGDSaI,
    Grams,
    Ranger25,
]
OPTIMIZERS: Dict[str, OPTIMIZER] = {str(optimizer.__name__).lower(): optimizer for optimizer in OPTIMIZER_LIST}


def create_optimizer(
    model: nn.Module,
    optimizer_name: str,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    wd_ban_list: List[str] = ('bias', 'LayerNorm.bias', 'LayerNorm.weight'),
    use_lookahead: bool = False,
    **kwargs,
):
    r"""Build optimizer.

    :param model: nn.Module. model.
    :param optimizer_name: str. name of optimizer.
    :param lr: float. learning rate.
    :param weight_decay: float. weight decay.
    :param wd_ban_list: List[str]. weight decay ban list by layer.
    :param use_lookahead: bool. use lookahead.
    """
    optimizer_name = optimizer_name.lower()

    parameters = (
        get_optimizer_parameters(model, weight_decay, wd_ban_list) if weight_decay > 0.0 else model.parameters()
    )

    optimizer = load_optimizer(optimizer_name)

    if optimizer_name == 'alig':
        optimizer = optimizer(parameters, max_lr=lr, **kwargs)
    elif optimizer_name in {'lomo', 'adalomo', 'adammini'}:
        optimizer = optimizer(model, lr=lr, **kwargs)
    else:
        optimizer = optimizer(parameters, lr=lr, **kwargs)

    if use_lookahead:
        optimizer = Lookahead(
            optimizer,
            k=kwargs.get('k', 5),
            alpha=kwargs.get('alpha', 0.5),
            pullback_momentum=kwargs.get('pullback_momentum', 'none'),
        )

    return optimizer


def get_optimizer_parameters(
    model_or_parameter: Union[nn.Module, List],
    weight_decay: float,
    wd_ban_list: List[str] = ('bias', 'LayerNorm.bias', 'LayerNorm.weight'),
) -> PARAMETERS:
    r"""Get optimizer parameters while filtering specified modules.

    Notice that, You can also ban by a module name level (e.g. LayerNorm) if you pass nn.Module instance. You just only
    need to input `LayerNorm` to exclude weight decay from the layer norm layer(s).

    :param model_or_parameter: Union[nn.Module, List]. model or parameters.
    :param weight_decay: float. weight_decay.
    :param wd_ban_list: List[str]. ban list not to set weight decay.
    :returns: PARAMETERS. new parameter list.
    """
    banned_parameter_patterns: Set[str] = set()

    if isinstance(model_or_parameter, nn.Module):
        for module_name, module in model_or_parameter.named_modules():
            for param_name, _ in module.named_parameters(recurse=False):
                full_param_name: str = f'{module_name}.{param_name}' if module_name else param_name
                if any(
                    banned in pattern for banned in wd_ban_list for pattern in (full_param_name, module._get_name())
                ):
                    banned_parameter_patterns.add(full_param_name)

        model_or_parameter = list(model_or_parameter.named_parameters())
    else:
        banned_parameter_patterns.update(wd_ban_list)

    return [
        {
            'params': [
                p
                for n, p in model_or_parameter
                if p.requires_grad and not any(nd in n for nd in banned_parameter_patterns)
            ],
            'weight_decay': weight_decay,
        },
        {
            'params': [
                p
                for n, p in model_or_parameter
                if p.requires_grad and any(nd in n for nd in banned_parameter_patterns)
            ],
            'weight_decay': 0.0,
        },
    ]


def get_supported_optimizers(filters: Optional[Union[str, List[str]]] = None) -> List[str]:
    r"""Return list of available optimizer names, sorted alphabetically.

    :param filters: Optional[Union[str, List[str]]]. wildcard filter string that works with fmatch. if None, it will
        return the whole list.
    """
    if filters is None:
        return sorted(OPTIMIZERS.keys())

    include_filters: Sequence[str] = filters if isinstance(filters, (tuple, list)) else [filters]

    filtered_list: Set[str] = set()
    for include_filter in include_filters:
        filtered_list.update(fnmatch.filter(OPTIMIZERS.keys(), include_filter))

    return sorted(filtered_list)
