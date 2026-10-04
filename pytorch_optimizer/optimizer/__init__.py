import fnmatch
from collections.abc import Callable, Sequence
from importlib.util import find_spec
from types import MethodType
from typing import cast
from warnings import warn

import torch
from torch import nn
from torch.optim import LBFGS, SGD, Adam, AdamW, NAdam, Optimizer, RMSprop

from pytorch_optimizer.base.type import OptimizerType, ParamsT
from pytorch_optimizer.optimizer.a2grad import A2Grad
from pytorch_optimizer.optimizer.adabelief import AdaBelief
from pytorch_optimizer.optimizer.adabound import AdaBound
from pytorch_optimizer.optimizer.adadelta import AdaDelta
from pytorch_optimizer.optimizer.adafactor import AdaFactor
from pytorch_optimizer.optimizer.adagc import AdaGC
from pytorch_optimizer.optimizer.adahessian import AdaHessian
from pytorch_optimizer.optimizer.adai import Adai
from pytorch_optimizer.optimizer.adalite import Adalite
from pytorch_optimizer.optimizer.adam_mini import AdamMini
from pytorch_optimizer.optimizer.adamax import AdaMax
from pytorch_optimizer.optimizer.adamc import AdamC
from pytorch_optimizer.optimizer.adamg import AdamG
from pytorch_optimizer.optimizer.adamod import AdaMod
from pytorch_optimizer.optimizer.adamp import SGDP, AdamP
from pytorch_optimizer.optimizer.adams import AdamS
from pytorch_optimizer.optimizer.adamw import StableAdamW
from pytorch_optimizer.optimizer.adan import Adan
from pytorch_optimizer.optimizer.adanorm import AdaNorm
from pytorch_optimizer.optimizer.adapnm import AdaPNM
from pytorch_optimizer.optimizer.adashift import AdaShift
from pytorch_optimizer.optimizer.adasmooth import AdaSmooth
from pytorch_optimizer.optimizer.ademamix import AdEMAMix, SimplifiedAdEMAMix
from pytorch_optimizer.optimizer.adopt import ADOPT
from pytorch_optimizer.optimizer.agc import agc
from pytorch_optimizer.optimizer.aggmo import AggMo
from pytorch_optimizer.optimizer.aida import Aida
from pytorch_optimizer.optimizer.alig import AliG
from pytorch_optimizer.optimizer.amos import Amos
from pytorch_optimizer.optimizer.ano import Ano
from pytorch_optimizer.optimizer.apollo import APOLLO, ApolloDQN
from pytorch_optimizer.optimizer.avagrad import AvaGrad
from pytorch_optimizer.optimizer.bcos import BCOS
from pytorch_optimizer.optimizer.came import CAME
from pytorch_optimizer.optimizer.conda import Conda
from pytorch_optimizer.optimizer.dadapt import DAdaptAdaGrad, DAdaptAdam, DAdaptAdan, DAdaptLion, DAdaptSGD
from pytorch_optimizer.optimizer.demo import DeMo
from pytorch_optimizer.optimizer.diffgrad import DiffGrad
from pytorch_optimizer.optimizer.dual_adam import DualAdam
from pytorch_optimizer.optimizer.emonavi import EmoFact, EmoLynx, EmoNavi
from pytorch_optimizer.optimizer.exadam import EXAdam
from pytorch_optimizer.optimizer.experimental.ranger25 import Ranger25
from pytorch_optimizer.optimizer.fadam import FAdam
from pytorch_optimizer.optimizer.fira import Fira
from pytorch_optimizer.optimizer.flash_adamw import FlashAdamW
from pytorch_optimizer.optimizer.focus import FOCUS
from pytorch_optimizer.optimizer.fp16 import DynamicLossScaler, SafeFP16Optimizer
from pytorch_optimizer.optimizer.fromage import Fromage
from pytorch_optimizer.optimizer.ftrl import FTRL
from pytorch_optimizer.optimizer.galore import GaLore
from pytorch_optimizer.optimizer.gradient_centralization import centralize_gradient
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
from pytorch_optimizer.optimizer.lora_rite import LoRARite
from pytorch_optimizer.optimizer.madgrad import MADGRAD
from pytorch_optimizer.optimizer.magma import Magma
from pytorch_optimizer.optimizer.mars import MARS
from pytorch_optimizer.optimizer.msvag import MSVAG
from pytorch_optimizer.optimizer.muon import AdaGO, AdaMuon, DistributedMuon, Muon, NorMuon, prepare_muon_parameters
from pytorch_optimizer.optimizer.nero import Nero
from pytorch_optimizer.optimizer.novograd import NovoGrad
from pytorch_optimizer.optimizer.orthograd import OrthoGrad
from pytorch_optimizer.optimizer.padam import PAdam
from pytorch_optimizer.optimizer.pcgrad import PCGrad
from pytorch_optimizer.optimizer.pid import PID
from pytorch_optimizer.optimizer.pnm import PNM
from pytorch_optimizer.optimizer.prodigy import Prodigy
from pytorch_optimizer.optimizer.psgd import Kron
from pytorch_optimizer.optimizer.qhadam import QHAdam
from pytorch_optimizer.optimizer.qhm import QHM
from pytorch_optimizer.optimizer.racs import RACS, Alice
from pytorch_optimizer.optimizer.radam import RAdam
from pytorch_optimizer.optimizer.ranger import Ranger
from pytorch_optimizer.optimizer.ranger21 import Ranger21
from pytorch_optimizer.optimizer.rose import ROSE
from pytorch_optimizer.optimizer.rotograd import RotoGrad
from pytorch_optimizer.optimizer.sam import BSAM, GSAM, SAM, WSAM, FriendlySAM, LookSAM
from pytorch_optimizer.optimizer.sara import SaRA
from pytorch_optimizer.optimizer.schedulefree import (
    ScheduleFreeAdamW,
    ScheduleFreeRAdam,
    ScheduleFreeSGD,
    ScheduleFreeWrapper,
)
from pytorch_optimizer.optimizer.scion import SCION, SCIONLight
from pytorch_optimizer.optimizer.sgd import ASGD, SGDW, VSGD, AccSGD, SGDSaI, SignSGD
from pytorch_optimizer.optimizer.shampoo import ScalableShampoo, Shampoo
from pytorch_optimizer.optimizer.sm3 import SM3
from pytorch_optimizer.optimizer.snsm import AdamWSN
from pytorch_optimizer.optimizer.soap import SOAP
from pytorch_optimizer.optimizer.sophia import SophiaH
from pytorch_optimizer.optimizer.spam import SPAM, StableSPAM
from pytorch_optimizer.optimizer.splus import SPlus
from pytorch_optimizer.optimizer.srmm import SRMM
from pytorch_optimizer.optimizer.sso import SpectralSphere
from pytorch_optimizer.optimizer.swats import SWATS
from pytorch_optimizer.optimizer.tam import TAM, AdaTAM
from pytorch_optimizer.optimizer.tiger import Tiger
from pytorch_optimizer.optimizer.trac import TRAC
from pytorch_optimizer.optimizer.yogi import Yogi

__all__ = [
    'ADOPT',
    'APOLLO',
    'ASGD',
    'BCOS',
    'BSAM',
    'CAME',
    'FOCUS',
    'FTRL',
    'GSAM',
    'LARS',
    'LBFGS',
    'LOMO',
    'MADGRAD',
    'MARS',
    'MSVAG',
    'PID',
    'PNM',
    'QHM',
    'RACS',
    'ROSE',
    'SAM',
    'SCION',
    'SGD',
    'SGDP',
    'SGDW',
    'SM3',
    'SOAP',
    'SPAM',
    'SRMM',
    'SWATS',
    'TAM',
    'TRAC',
    'VSGD',
    'WSAM',
    'A2Grad',
    'AccSGD',
    'AdEMAMix',
    'AdaBelief',
    'AdaBound',
    'AdaDelta',
    'AdaFactor',
    'AdaGC',
    'AdaGO',
    'AdaHessian',
    'AdaLOMO',
    'AdaMax',
    'AdaMod',
    'AdaMuon',
    'AdaNorm',
    'AdaPNM',
    'AdaShift',
    'AdaSmooth',
    'AdaTAM',
    'Adai',
    'Adalite',
    'Adam',
    'AdamC',
    'AdamG',
    'AdamMini',
    'AdamP',
    'AdamS',
    'AdamW',
    'AdamWSN',
    'Adan',
    'AggMo',
    'Aida',
    'AliG',
    'Alice',
    'Amos',
    'Ano',
    'ApolloDQN',
    'AvaGrad',
    'Conda',
    'DAdaptAdaGrad',
    'DAdaptAdam',
    'DAdaptAdan',
    'DAdaptLion',
    'DAdaptSGD',
    'DeMo',
    'DiffGrad',
    'DistributedMuon',
    'DualAdam',
    'DynamicLossScaler',
    'EXAdam',
    'EmoFact',
    'EmoLynx',
    'EmoNavi',
    'FAdam',
    'Fira',
    'FlashAdamW',
    'FriendlySAM',
    'Fromage',
    'GaLore',
    'Grams',
    'Gravity',
    'GrokFastAdamW',
    'Kate',
    'Kron',
    'LaProp',
    'Lamb',
    'Lion',
    'LoRARite',
    'LookSAM',
    'Lookahead',
    'Magma',
    'Muon',
    'NAdam',
    'Nero',
    'NorMuon',
    'NovoGrad',
    'OrthoGrad',
    'PAdam',
    'PCGrad',
    'Prodigy',
    'QHAdam',
    'RAdam',
    'RMSprop',
    'Ranger',
    'Ranger21',
    'Ranger25',
    'RotoGrad',
    'SCIONLight',
    'SGDSaI',
    'SPlus',
    'SaRA',
    'SafeFP16Optimizer',
    'ScalableShampoo',
    'ScheduleFreeAdamW',
    'ScheduleFreeRAdam',
    'ScheduleFreeSGD',
    'ScheduleFreeWrapper',
    'Shampoo',
    'SignSGD',
    'SimplifiedAdEMAMix',
    'SophiaH',
    'SpectralSphere',
    'StableAdamW',
    'StableSPAM',
    'Tiger',
    'Yogi',
    'agc',
    'centralize_gradient',
    'create_optimizer',
    'get_optimizer_parameters',
    'get_supported_optimizers',
    'load_ao_optimizer',
    'load_bnb_optimizer',
    'load_optimizer',
    'load_q_galore_optimizer',
]

HAS_BNB: bool = find_spec('bitsandbytes') is not None
HAS_Q_GALORE: bool = find_spec('q_galore_torch') is not None
HAS_TORCHAO: bool = find_spec('torchao') is not None

OPTIMIZER_LIST: list[OptimizerType] = [
    LBFGS,
    SGD,
    Adam,
    AdamW,
    NAdam,
    RMSprop,
    A2Grad,
    ADOPT,
    APOLLO,
    ASGD,
    AccSGD,
    AdEMAMix,
    AdaBelief,
    AdaBound,
    AdaDelta,
    AdaFactor,
    AdaGC,
    AdaGO,
    AdaHessian,
    AdaLOMO,
    AdaMax,
    AdaMod,
    AdaMuon,
    AdaNorm,
    AdaPNM,
    AdaShift,
    AdaSmooth,
    AdaTAM,
    Adai,
    Adalite,
    AdamC,
    AdamG,
    AdamMini,
    AdamP,
    AdamS,
    AdamWSN,
    Adan,
    AggMo,
    Aida,
    AliG,
    Alice,
    BCOS,
    Amos,
    Ano,
    ApolloDQN,
    AvaGrad,
    BSAM,
    CAME,
    Conda,
    DAdaptAdaGrad,
    DAdaptAdam,
    DAdaptAdan,
    DAdaptLion,
    DAdaptSGD,
    DeMo,
    DiffGrad,
    DistributedMuon,
    DualAdam,
    EXAdam,
    EmoFact,
    EmoLynx,
    EmoNavi,
    FAdam,
    FOCUS,
    FTRL,
    Fira,
    FlashAdamW,
    Fromage,
    GaLore,
    Grams,
    Gravity,
    GrokFastAdamW,
    Kate,
    Kron,
    LARS,
    LOMO,
    LoRARite,
    LaProp,
    Lamb,
    Lion,
    MADGRAD,
    Magma,
    MARS,
    MSVAG,
    Muon,
    Nero,
    NorMuon,
    NovoGrad,
    PAdam,
    PID,
    PNM,
    Prodigy,
    QHAdam,
    QHM,
    RACS,
    RAdam,
    Ranger,
    Ranger21,
    Ranger25,
    ROSE,
    SCION,
    SCIONLight,
    SGDP,
    SGDSaI,
    SGDW,
    SM3,
    SOAP,
    SPAM,
    SPlus,
    SRMM,
    SWATS,
    SaRA,
    ScalableShampoo,
    ScheduleFreeAdamW,
    ScheduleFreeRAdam,
    ScheduleFreeSGD,
    Shampoo,
    SignSGD,
    SimplifiedAdEMAMix,
    SophiaH,
    StableAdamW,
    StableSPAM,
    TAM,
    Tiger,
    VSGD,
    Yogi,
    SpectralSphere,
]
OPTIMIZERS: dict[str, OptimizerType] = {str(optimizer.__name__).lower(): optimizer for optimizer in OPTIMIZER_LIST}

BNB_OPTIMIZERS = (
    ('paged_ademamix8bit', 'PagedAdEMAMix8bit'),
    ('paged_ademamix32bit', 'PagedAdEMAMix32bit'),
    ('paged_adam8bit', 'PagedAdam8bit'),
    ('paged_adam32bit', 'PagedAdam32bit'),
    ('paged_adamw8bit', 'PagedAdamW8bit'),
    ('paged_lion32bit', 'PagedLion32bit'),
    ('ademamix8bit', 'AdEMAMix8bit'),
    ('ademamix32bit', 'AdEMAMix32bit'),
    ('adam8bit', 'Adam8bit'),
    ('adam32bit', 'Adam32bit'),
    ('adamw8bit', 'AdamW8bit'),
    ('adamw32bit', 'AdamW32bit'),
    ('adagrad8bit', 'Adagrad8bit'),
    ('adagrad32bit', 'Adagrad32bit'),
    ('lamb8bit', 'LAMB8bit'),
    ('lamb32bit', 'LAMB32bit'),
    ('lars8bit', 'LARS8bit'),
    ('lars32bit', 'LARS32bit'),
    ('lion8bit', 'Lion8bit'),
    ('lion32bit', 'Lion32bit'),
    ('rmsprop8bit', 'RMSprop8bit'),
    ('rmsprop32bit', 'RMSprop32bit'),
    ('sgd8bit', 'SGD8bit'),
    ('sgd32bit', 'SGD32bit'),
)


def load_bnb_optimizer(optimizer: str) -> OptimizerType:  # pragma: no cover
    """Return an optimizer class from bitsandbytes.

    Args:
        optimizer: Lowercase optimizer name, including the integration prefix.

    Returns:
        OptimizerType: Optimizer class from the optional integration.

    Raises:
        NotImplementedError: The optimizer name is unsupported.

    """
    from bitsandbytes import optim  # noqa: PLC0415

    for name, cls_name in BNB_OPTIMIZERS:
        if name in optimizer:
            return getattr(optim, cls_name)

    raise NotImplementedError(f'not implemented optimizer {optimizer}')


def load_q_galore_optimizer(optimizer: str) -> OptimizerType:  # pragma: no cover
    """Return an optimizer class from Q-GaLore.

    Args:
        optimizer: Lowercase optimizer name, including the integration prefix.

    Returns:
        OptimizerType: Optimizer class from the optional integration.

    Raises:
        NotImplementedError: The optimizer name is unsupported.

    """
    import q_galore_torch  # noqa: PLC0415

    if 'adamw8bit' in optimizer:
        return q_galore_torch.QGaLoreAdamW8bit

    raise NotImplementedError(f'not implemented optimizer {optimizer}')


def load_ao_optimizer(optimizer: str) -> OptimizerType:  # pragma: no cover
    """Return an optimizer class from TorchAO.

    Args:
        optimizer: Lowercase optimizer name, including the integration prefix.

    Returns:
        OptimizerType: Optimizer class from the optional integration.

    Raises:
        NotImplementedError: The optimizer name is unsupported.

    """
    from torchao.prototype import low_bit_optim  # noqa: PLC0415

    if 'adamw8bit' in optimizer:
        return low_bit_optim.AdamW8bit
    if 'adamw4bit' in optimizer:
        return low_bit_optim.AdamW4bit
    if 'adamwfp8' in optimizer:
        return low_bit_optim.AdamWFp8

    raise NotImplementedError(f'not implemented optimizer {optimizer}')


def load_optimizer(optimizer: str) -> OptimizerType:
    """Return an optimizer class by name.

    Names are case insensitive. Use the `bnb`, `q_galore`, or `torchao` prefix for
    optional integrations, which require their dependencies and CUDA.

    Args:
        optimizer: Registered optimizer name.

    Returns:
        OptimizerType: Optimizer class to instantiate with parameters and options.

    Raises:
        ImportError: An optional integration or CUDA is unavailable.
        NotImplementedError: The optimizer name is unsupported.

    """
    optimizer_name: str = optimizer.lower()

    if optimizer_name.startswith('bnb'):
        if HAS_BNB and torch.cuda.is_available():
            return load_bnb_optimizer(optimizer_name)  # pragma: no cover
        raise ImportError(f'bitsandbytes and CUDA required for the optimizer {optimizer_name}')
    if optimizer_name.startswith('q_galore'):
        if HAS_Q_GALORE and torch.cuda.is_available():
            return load_q_galore_optimizer(optimizer_name)  # pragma: no cover
        raise ImportError(f'bitsandbytes, q-galore-torch, and CUDA required for the optimizer {optimizer_name}')
    if optimizer_name.startswith('torchao'):
        if HAS_TORCHAO and torch.cuda.is_available():
            return load_ao_optimizer(optimizer_name)  # pragma: no cover
        raise ImportError(
            f'torchao required for the optimizer {optimizer_name}. '
            'usage: https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim#usage'
        )
    if optimizer_name not in OPTIMIZERS:
        raise NotImplementedError(f'not implemented optimizer : {optimizer_name}')

    return OPTIMIZERS[optimizer_name]


def create_optimizer(
    model: nn.Module,
    optimizer_name: str,
    lr: float | torch.Tensor = 1e-3,
    weight_decay: float = 0.0,
    wd_ban_list: Sequence[str] = ('bias', 'LayerNorm.bias', 'LayerNorm.weight'),
    use_lookahead: bool = False,
    use_orthograd: bool = False,
    compile: bool = False,  # noqa: A002
    compile_kwargs: dict | None = None,
    **kwargs,
) -> Optimizer:
    """Create an optimizer for a model with optional wrappers and compilation.

    Args:
        model: Model whose parameters to optimize.
        optimizer_name: Case insensitive name accepted by `load_optimizer()`.
        lr: Learning rate. Compilation converts a float to a tensor on the model's device.
        weight_decay: Weight decay coefficient.
        wd_ban_list: Name patterns to exclude from weight decay. Matches parameter names and module class names.
        use_lookahead: Wrap the optimizer with Lookahead, unless it already includes Lookahead.
        use_orthograd: Project gradients with OrthoGrad before each update.
        compile: Compile the optimizer step with `torch.compile`.
        compile_kwargs: Options for `torch.compile`. Dynamic tracing defaults to `True`.
        **kwargs (dict): Optimizer and wrapper options.

    Returns:
        Optimizer: Configured optimizer instance.

    """
    optimizer_name = optimizer_name.lower()

    if compile and not isinstance(lr, torch.Tensor):
        lr = torch.tensor(lr, device=next(model.parameters()).device)

    if optimizer_name != 'lbfgs':
        kwargs['weight_decay'] = weight_decay

    parameters = (
        get_optimizer_parameters(model, weight_decay, wd_ban_list)
        if weight_decay > 0.0
        else [{'params': model.parameters(), 'weight_decay': weight_decay}]
    )

    optimizer_class = cast(Callable[..., Optimizer], load_optimizer(optimizer_name))

    if optimizer_name == 'alig':
        optimizer = optimizer_class(parameters, max_lr=lr, **kwargs)
    elif optimizer_name in ('lomo', 'adalomo', 'adammini'):
        optimizer = optimizer_class(model, lr=lr, **kwargs)
    elif optimizer_name in ('muon', 'adamuon', 'adago', 'normuon'):
        warn(f'highly recommend you to manually create the {optimizer_name} manually.', UserWarning, stacklevel=1)

        optimizer = prepare_muon_parameters(model, optimizer_name, lr=lr, **kwargs)
    else:
        optimizer = optimizer_class(parameters, lr=lr, **kwargs)

    if use_orthograd:
        optimizer = OrthoGrad(optimizer, **kwargs)

    if use_lookahead:
        if optimizer_name in ('ranger', 'ranger21', 'ranger25'):
            warn(f'{optimizer} already has a Lookahead variant.', UserWarning, stacklevel=1)
        else:
            optimizer = Lookahead(
                optimizer,
                k=kwargs.get('k', 5),
                alpha=kwargs.get('alpha', 0.5),
                pullback_momentum=kwargs.get('pullback_momentum', 'none'),
            )

    if compile:
        optimizer.step = MethodType(  # ty: ignore[invalid-assignment]
            torch.compile(optimizer.step.__func__, **{'dynamic': True, **(compile_kwargs or {})}),
            optimizer,
        )

    return optimizer


def get_optimizer_parameters(
    model_or_parameter: nn.Module | list,
    weight_decay: float,
    wd_ban_list: Sequence[str] = ('bias', 'LayerNorm.bias', 'LayerNorm.weight'),
) -> ParamsT:
    """Group trainable parameters by whether to apply weight decay.

    With a model, patterns match parameter names and module class names. For example,
    `LayerNorm` excludes all parameters of LayerNorm modules. With named parameters,
    patterns match parameter names only.

    Args:
        model_or_parameter: Model or list of `(name, parameter)` pairs.
        weight_decay: Weight decay coefficient for parameters outside the ban list.
        wd_ban_list: Substrings identifying parameters to exclude from weight decay.

    Returns:
        ParamsT: Nonempty parameter groups with the requested weight decay or zero weight decay.

    """
    banned_parameter_patterns: set[str] = set()

    if isinstance(model_or_parameter, nn.Module):
        for module_name, module in model_or_parameter.named_modules():
            for param_name, _ in module.named_parameters(recurse=False):
                full_param_name: str = f'{module_name}.{param_name}' if module_name else param_name
                if any(
                    banned in pattern
                    for banned in wd_ban_list
                    for pattern in (full_param_name, module._get_name(), f'{module._get_name()}.{param_name}')
                ):
                    banned_parameter_patterns.add(full_param_name)

        model_or_parameter = list(model_or_parameter.named_parameters())
    else:
        banned_parameter_patterns.update(wd_ban_list)

    groups = [
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
    return [group for group in groups if group['params']]


def get_supported_optimizers(filters: str | list[str] | None = None) -> list[str]:
    """List registered optimizer names in alphabetical order.

    Args:
        filters: Wildcard pattern or list of patterns, such as `'*adam*'`. `None` selects all names.

    Returns:
        list[str]: Matching names in lowercase, without duplicates.

    """
    if filters is None:
        return sorted(OPTIMIZERS.keys())

    include_filters: Sequence[str] = filters if isinstance(filters, (tuple, list)) else [filters]

    filtered_list: set[str] = set()
    for include_filter in include_filters:
        filtered_list.update(fnmatch.filter(OPTIMIZERS.keys(), include_filter))

    return sorted(filtered_list)
