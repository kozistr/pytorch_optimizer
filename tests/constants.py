from typing import Any, Dict, List, Tuple, Union

from pytorch_optimizer import (
    ASGD,
    LARS,
    MADGRAD,
    MSVAG,
    OPTIMIZERS,
    PID,
    PNM,
    QHM,
    SGDP,
    SGDW,
    SM3,
    SRMM,
    SWATS,
    A2Grad,
    AccSGD,
    AdaBelief,
    AdaBound,
    AdaDelta,
    AdaFactor,
    AdaHessian,
    Adai,
    AdaMax,
    AdaMod,
    AdamP,
    AdamS,
    Adan,
    AdaNorm,
    AdaPNM,
    AdaShift,
    AdaSmooth,
    AggMo,
    AliG,
    Amos,
    Apollo,
    AvaGrad,
    DAdaptAdaGrad,
    DAdaptAdam,
    DAdaptAdan,
    DAdaptSGD,
    DiffGrad,
    Fromage,
    Gravity,
    Lamb,
    Lion,
    Nero,
    NovoGrad,
    QHAdam,
    RAdam,
    Ranger,
    Ranger21,
    ScalableShampoo,
    Shampoo,
    SignSGD,
    SophiaH,
    Yogi,
)
from tests.utils import build_lookahead

ADAPTIVE_FLAGS: List[bool] = [True, False]
PULLBACK_MOMENTUM: List[str] = ['none', 'reset', 'pullback']

VALID_OPTIMIZER_NAMES: List[str] = list(OPTIMIZERS.keys())
INVALID_OPTIMIZER_NAMES: List[str] = [
    'asam',
    'sam',
    'gsam',
    'pcgrad',
    'adamd',
    'lookahead',
]

SPARSE_OPTIMIZERS: List[str] = ['madgrad', 'dadaptadagrad', 'sm3']
NO_SPARSE_OPTIMIZERS: List[str] = [
    optimizer for optimizer in VALID_OPTIMIZER_NAMES if optimizer not in SPARSE_OPTIMIZERS
]

BETA_OPTIMIZER_NAMES: List[str] = [
    'adabelief',
    'adabound',
    'adamp',
    'diffgrad',
    'lamb',
    'radam',
    'ranger',
    'ranger21',
    'pnm',
    'adapnm',
    'adan',
    'adai',
    'scalableshampoo',
    'dadaptadam',
    'dadaptadan',
    'adams',
    'adafactor',
    'novograd',
    'lion',
    'adanorm',
    'yogi',
    'swats',
    'adamod',
    'aggmo',
    'qhadam',
    'adamax',
    'adasmooth',
    'adashift',
    'sophiah',
]

VALID_LR_SCHEDULER_NAMES: List[str] = [
    'CosineAnnealingWarmupRestarts',
    'ConstantLR',
    'CosineAnnealingLR',
    'CosineAnnealingWarmRestarts',
    'CyclicLR',
    'OneCycleLR',
]
INVALID_LR_SCHEDULER_NAMES: List[str] = ['dummy']

OPTIMIZERS: List[Tuple[Any, Dict[str, Union[float, bool, int]], int]] = [
    (build_lookahead, {'lr': 5e-1, 'weight_decay': 1e-3}, 5),
    (AdaBelief, {'lr': 5e-1, 'weight_decay': 1e-3}, 10),
    (AdaBelief, {'lr': 5e-1, 'weight_decay': 1e-3, 'ams_bound': True}, 10),
    (AdaBelief, {'lr': 5e-1, 'weight_decay': 1e-3, 'weight_decouple': False}, 10),
    (AdaBelief, {'lr': 5e-1, 'weight_decay': 1e-3, 'fixed_decay': True}, 10),
    (AdaBelief, {'lr': 5e-1, 'weight_decay': 1e-3, 'rectify': False}, 10),
    (AdaBound, {'lr': 1e0, 'gamma': 0.1, 'weight_decay': 1e-3}, 20),
    (AdaBound, {'lr': 1e0, 'gamma': 0.1, 'weight_decay': 1e-3, 'fixed_decay': True}, 20),
    (AdaBound, {'lr': 1e0, 'gamma': 0.1, 'weight_decay': 1e-3, 'weight_decouple': False}, 20),
    (AdaBound, {'lr': 1e0, 'gamma': 0.1, 'weight_decay': 1e-3, 'ams_bound': True}, 20),
    (Adai, {'lr': 5e-1, 'weight_decay': 0.0}, 5),
    (Adai, {'lr': 1e0, 'weight_decay': 0.0, 'use_gc': True}, 20),
    (Adai, {'lr': 5e-1, 'weight_decay': 0.0, 'dampening': 0.9}, 5),
    (Adai, {'lr': 5e-1, 'weight_decay': 1e-4, 'weight_decouple': False}, 5),
    (Adai, {'lr': 5e-1, 'weight_decay': 1e-4, 'weight_decouple': True}, 5),
    (Adai, {'lr': 5e-1, 'weight_decay': 1e-4, 'weight_decouple': False, 'stable_weight_decay': True}, 5),
    (Adai, {'lr': 5e-1, 'weight_decay': 1e-4, 'weight_decouple': True, 'stable_weight_decay': True}, 5),
    (AdamP, {'lr': 5e-1, 'weight_decay': 1e-3}, 5),
    (AdamP, {'lr': 5e-1, 'weight_decay': 1e-3, 'use_gc': True}, 10),
    (AdamP, {'lr': 5e-1, 'weight_decay': 1e-3, 'nesterov': True}, 5),
    (DiffGrad, {'lr': 5e-2, 'weight_decay': 1e-3}, 5),
    (DiffGrad, {'lr': 5e-2, 'weight_decay': 1e-3, 'ams_bound': True}, 5),
    (DiffGrad, {'lr': 5e-1, 'weight_decay': 1e-3, 'rectify': True}, 5),
    (Lamb, {'lr': 1e-1, 'weight_decay': 1e-3}, 5),
    (Lamb, {'lr': 1e-1, 'weight_decay': 1e-3, 'pre_norm': True, 'max_grad_norm': 0.0}, 5),
    (Lamb, {'lr': 1e-1, 'weight_decay': 1e-3, 'grad_averaging': False}, 5),
    (Lamb, {'lr': 1e-1, 'weight_decay': 1e-3, 'adam': True, 'eps': 1e-8}, 5),
    (Lamb, {'lr': 1e-1, 'weight_decay': 1e-3, 'pre_norm': True, 'eps': 1e-8}, 5),
    (Lamb, {'lr': 5e-1, 'weight_decay': 1e-3, 'rectify': True, 'degenerated_to_sgd': True}, 5),
    (LARS, {'lr': 5e-1, 'weight_decay': 1e-3}, 20),
    (LARS, {'lr': 5e-1, 'nesterov': True}, 20),
    (MADGRAD, {'lr': 5e-1, 'weight_decay': 1e-3}, 10),
    (MADGRAD, {'lr': 5e-1, 'weight_decay': 1e-3, 'eps': 0.0}, 10),
    (MADGRAD, {'lr': 1e-1, 'weight_decay': 1e-3, 'momentum': 0.0}, 10),
    (MADGRAD, {'lr': 5e-1, 'weight_decay': 1e-3, 'weight_decouple': True}, 10),
    (RAdam, {'lr': 5e-0, 'weight_decay': 1e-3}, 10),
    (RAdam, {'lr': 5e-1, 'weight_decay': 1e-3, 'degenerated_to_sgd': True}, 5),
    (SGDP, {'lr': 5e-1, 'weight_decay': 1e-4}, 10),
    (SGDP, {'lr': 5e-1, 'weight_decay': 1e-4, 'nesterov': True}, 10),
    (Ranger, {'lr': 1e0, 'weight_decay': 1e-3}, 75),
    (Ranger, {'lr': 5e0, 'weight_decay': 1e-3, 'degenerated_to_sgd': True}, 5),
    (Ranger21, {'lr': 7.5e-1, 'weight_decay': 1e-3, 'num_iterations': 75}, 75),
    (Shampoo, {'lr': 5e-1, 'weight_decay': 1e-3, 'momentum': 0.1}, 10),
    (
        ScalableShampoo,
        {
            'lr': 1e-1,
            'weight_decay': 1e-3,
            'start_preconditioning_step': 9,
            'preconditioning_compute_steps': 9,
            'graft_type': 0,
        },
        10,
    ),
    (
        ScalableShampoo,
        {
            'lr': 1e-1,
            'weight_decay': 1e-3,
            'start_preconditioning_step': 9,
            'preconditioning_compute_steps': 10,
            'graft_type': 1,
        },
        10,
    ),
    (
        ScalableShampoo,
        {
            'lr': 1e-1,
            'weight_decay': 1e-3,
            'start_preconditioning_step': 9,
            'preconditioning_compute_steps': 10,
            'graft_type': 2,
        },
        10,
    ),
    (
        ScalableShampoo,
        {
            'lr': 1e-2,
            'weight_decay': 1e-3,
            'start_preconditioning_step': 9,
            'preconditioning_compute_steps': 10,
            'graft_type': 3,
        },
        10,
    ),
    (
        ScalableShampoo,
        {
            'lr': 1e-1,
            'weight_decay': 1e-3,
            'start_preconditioning_step': 9,
            'preconditioning_compute_steps': 10,
            'graft_type': 4,
        },
        10,
    ),
    (
        ScalableShampoo,
        {
            'lr': 1e-1,
            'weight_decay': 1e-3,
            'start_preconditioning_step': 9,
            'preconditioning_compute_steps': 10,
            'pre_conditioner_type': 0,
        },
        10,
    ),
    (
        ScalableShampoo,
        {
            'lr': 1e-1,
            'weight_decay': 1e-3,
            'start_preconditioning_step': 9,
            'preconditioning_compute_steps': 10,
            'pre_conditioner_type': 1,
        },
        10,
    ),
    (
        ScalableShampoo,
        {
            'lr': 1e-1,
            'weight_decay': 1e-3,
            'start_preconditioning_step': 9,
            'preconditioning_compute_steps': 10,
            'pre_conditioner_type': 2,
        },
        10,
    ),
    (
        ScalableShampoo,
        {
            'lr': 1e-1,
            'weight_decay': 1e-3,
            'start_preconditioning_step': 9,
            'preconditioning_compute_steps': 10,
            'inverse_exponent_override': 1,
        },
        10,
    ),
    (
        ScalableShampoo,
        {
            'lr': 1e-1,
            'weight_decay': 1e-3,
            'start_preconditioning_step': 9,
            'preconditioning_compute_steps': 10,
            'nesterov': False,
        },
        10,
    ),
    (
        ScalableShampoo,
        {
            'lr': 1e-1,
            'weight_decay': 1e-3,
            'start_preconditioning_step': 9,
            'preconditioning_compute_steps': 10,
            'decoupled_weight_decay': True,
        },
        10,
    ),
    (
        ScalableShampoo,
        {
            'lr': 1e-0,
            'weight_decay': 1e-3,
            'start_preconditioning_step': 9,
            'preconditioning_compute_steps': 10,
            'decoupled_learning_rate': False,
        },
        10,
    ),
    (
        ScalableShampoo,
        {
            'lr': 1e-1,
            'weight_decay': 1e-3,
            'start_preconditioning_step': 9,
            'preconditioning_compute_steps': 10,
            'moving_average_for_momentum': True,
        },
        10,
    ),
    (PNM, {'lr': 5e-1}, 25),
    (PNM, {'lr': 5e-1, 'weight_decouple': False}, 25),
    (AdaPNM, {'lr': 5e-1, 'weight_decay': 1e-3}, 10),
    (AdaPNM, {'lr': 5e-1, 'weight_decay': 1e-3, 'weight_decouple': False}, 10),
    (AdaPNM, {'lr': 5e-1, 'weight_decay': 1e-3, 'ams_bound': False}, 10),
    (Nero, {'lr': 5e-1}, 25),
    (Nero, {'lr': 5e0, 'constraints': False}, 5),
    (Adan, {'lr': 5e-1}, 5),
    (Adan, {'lr': 5e-1, 'max_grad_norm': 1.0}, 5),
    (Adan, {'lr': 5e-1, 'weight_decay': 1e-3, 'use_gc': True}, 5),
    (Adan, {'lr': 5e-1, 'weight_decay': 1e-3, 'weight_decouple': True}, 5),
    (DAdaptAdaGrad, {'lr': 3e0, 'weight_decay': 1e-3}, 30),
    (DAdaptAdaGrad, {'lr': 5e0, 'weight_decay': 1e-3, 'momentum': 0.1}, 20),
    (DAdaptAdam, {'lr': 5e4, 'weight_decay': 1e-3}, 5),
    (DAdaptSGD, {'lr': 2e0, 'weight_decay': 1e-3}, 25),
    (DAdaptAdan, {'lr': 2e0, 'weight_decay': 1e-3}, 20),
    (DAdaptAdan, {'lr': 2e0, 'weight_decay': 1e-3, 'weight_decouple': True}, 20),
    (AdamS, {'lr': 1e0, 'weight_decay': 1e-3}, 10),
    (AdamS, {'lr': 1e0, 'weight_decay': 1e-3, 'ams_bound': True}, 20),
    (AdaFactor, {'lr': 7.5e-1, 'weight_decay': 1e-3, 'scale_parameter': False}, 100),
    (AdaFactor, {'lr': 7.5e-1, 'weight_decay': 1e-3, 'ams_bound': True}, 125),
    (Apollo, {'lr': 5e-1, 'weight_decay': 1e-3}, 10),
    (Apollo, {'lr': 5e-1, 'weight_decay': 1e-3, 'rebound': 'belief'}, 10),
    (Apollo, {'lr': 5e-1, 'weight_decay': 1e-3, 'weight_decay_type': 'stable', 'warmup_steps': 0}, 50),
    (NovoGrad, {'lr': 5e-1, 'weight_decay': 1e-3, 'grad_averaging': True}, 5),
    (Lion, {'lr': 5e-1, 'weight_decay': 1e-3}, 5),
    (Lion, {'lr': 5e-1, 'weight_decay': 1e-3, 'weight_decouple': False}, 5),
    (Lion, {'lr': 5e-1, 'weight_decay': 1e-3, 'use_gc': True}, 10),
    (AliG, {'max_lr': 5e-1, 'momentum': 0.9}, 5),
    (AliG, {'max_lr': 5e-1, 'momentum': 0.9, 'adjusted_momentum': True}, 5),
    (SM3, {'lr': 5e-1, 'momentum': 0.9, 'beta': 0.9}, 5),
    (AdaNorm, {'lr': 5e-1, 'weight_decay': 1e-3}, 5),
    (AdaNorm, {'lr': 5e-1, 'weight_decay': 1e-3, 'fixed_decay': True}, 5),
    (AdaNorm, {'lr': 5e-1, 'weight_decay': 1e-3, 'weight_decouple': False}, 5),
    (AdaNorm, {'lr': 5e-1, 'weight_decay': 1e-3, 'ams_bound': True}, 5),
    (A2Grad, {'variant': 'uni', 'beta': 1.0, 'lips': 1.0}, 5),
    (A2Grad, {'variant': 'inc', 'beta': 1.0, 'lips': 1.0}, 5),
    (A2Grad, {'variant': 'exp', 'beta': 1.0, 'lips': 1.0, 'rho': 0.9}, 5),
    (AccSGD, {'lr': 1e-0, 'weight_decay': 1e-3}, 5),
    (SGDW, {'lr': 5e-1, 'momentum': 0.9, 'weight_decay': 1e-3}, 5),
    (SGDW, {'lr': 5e-1, 'momentum': 0.9, 'weight_decay': 1e-3, 'weight_decouple': False}, 5),
    (SGDW, {'lr': 5e-1, 'momentum': 0.9, 'weight_decay': 1e-3, 'nesterov': True}, 5),
    (ASGD, {'lr': 5e-1, 'weight_decay': 1e-3}, 5),
    (ASGD, {'lr': 5e-1, 'weight_decay': 1e-3, 'weight_decouple': False}, 5),
    (Yogi, {'lr': 5e-1, 'weight_decay': 1e-3}, 5),
    (Fromage, {'lr': 5e-1, 'p_bound': 2.0}, 5),
    (MSVAG, {'lr': 5e-1}, 10),
    (AdaMod, {'lr': 5e1, 'weight_decay': 1e-3}, 10),
    (AdaMod, {'lr': 5e1, 'weight_decay': 1e-3, 'weight_decouple': False}, 10),
    (AggMo, {'lr': 5e0, 'weight_decay': 1e-3}, 5),
    (AggMo, {'lr': 5e0, 'weight_decay': 1e-3, 'weight_decouple': True}, 5),
    (QHAdam, {'lr': 1e0, 'nus': (0.9, 0.9), 'weight_decay': 1e-3}, 5),
    (QHAdam, {'lr': 1e0, 'weight_decay': 1e-3, 'weight_decouple': True}, 5),
    (QHM, {'lr': 1e0, 'weight_decay': 1e-3}, 5),
    (QHM, {'lr': 1e0, 'weight_decay': 1e-3, 'weight_decouple': True}, 5),
    (PID, {'lr': 1e0, 'momentum': 0.9, 'dampening': 1.0, 'weight_decay': 1e-3}, 5),
    (PID, {'lr': 1e0, 'momentum': 0.9, 'dampening': 1.0, 'weight_decay': 1e-3, 'weight_decouple': True}, 5),
    (AdaMax, {'lr': 1e0, 'weight_decay': 1e-3}, 5),
    (AdaMax, {'lr': 1e0, 'weight_decay': 1e-3, 'weight_decouple': True}, 5),
    (Gravity, {'lr': 1e0}, 5),
    (AdaSmooth, {'lr': 5e-1}, 5),
    (SRMM, {'lr': 5e-1}, 5),
    (AvaGrad, {'lr': 1e1}, 5),
    (AdaShift, {'lr': 1e0, 'keep_num': 1}, 5),
    (AdaDelta, {'lr': 5e1}, 5),
    (Amos, {'lr': 1e0, 'momentum': 0.9}, 5),
    (SignSGD, {'lr': 1e0, 'momentum': 0.0}, 5),
    (SignSGD, {'lr': 1e0, 'momentum': 0.9}, 5),
    (SophiaH, {'lr': 1e1, 'weight_decay': 1e-3, 'update_period': 2, 'hessian_distribution': 'gaussian'}, 5),
    (AdaHessian, {'lr': 1e0, 'weight_decay': 1e-3, 'hessian_distribution': 'rademacher'}, 5),
    (AdaHessian, {'lr': 1e0, 'weight_decay': 1e-3, 'hessian_distribution': 'gaussian'}, 5),
    (SWATS, {'lr': 5e-1, 'weight_decay': 1e-3}, 5),
    (SWATS, {'lr': 5e-1, 'weight_decay': 1e-3, 'ams_bound': True}, 5),
]
ADANORM_SUPPORTED_OPTIMIZERS: List[Tuple[Any, Dict[str, Union[float, bool, int]], int]] = [
    (AdaBelief, {'lr': 5e-1, 'weight_decay': 1e-3, 'adanorm': True}, 10),
    (AdamP, {'lr': 5e-1, 'weight_decay': 1e-3, 'adanorm': True}, 5),
    (AdamS, {'lr': 7.5e-1, 'weight_decay': 1e-3, 'adanorm': True}, 5),
    (AdaPNM, {'lr': 5e-1, 'weight_decay': 1e-3, 'adanorm': True}, 5),
    (DiffGrad, {'lr': 5e-2, 'weight_decay': 1e-3, 'adanorm': True}, 5),
    (Lamb, {'lr': 5e-2, 'adanorm': True}, 10),
    (RAdam, {'lr': 5e0, 'weight_decay': 1e-3, 'adanorm': True}, 10),
    (Ranger, {'lr': 2.5e0, 'weight_decay': 1e-3, 'adanorm': True}, 75),
    (Adan, {'lr': 5e-1, 'weight_decay': 1e-3, 'adanorm': True}, 5),
    (Lion, {'lr': 5e-1, 'weight_decay': 1e-3, 'adanorm': True}, 5),
    (Yogi, {'lr': 5e-1, 'weight_decay': 1e-3, 'adanorm': True}, 5),
    (AdaMax, {'lr': 5e-1, 'weight_decay': 1e-3, 'adanorm': True}, 5),
    (SWATS, {'lr': 5e-1, 'weight_decay': 1e-3, 'adanorm': True}, 5),
]
ADAMD_SUPPORTED_OPTIMIZERS: List[Tuple[Any, Dict[str, Union[float, bool, int]], int]] = [
    (AdaBelief, {'lr': 1e1, 'weight_decay': 1e-3, 'adam_debias': True}, 5),
    (AdaBound, {'lr': 1e0, 'gamma': 0.1, 'weight_decay': 1e-3, 'adam_debias': True}, 35),
    (AdamP, {'lr': 1e0, 'weight_decay': 1e-3, 'adam_debias': True}, 5),
    (AdamS, {'lr': 2e1, 'weight_decay': 1e-3, 'adam_debias': True}, 5),
    (DiffGrad, {'lr': 2e0, 'weight_decay': 1e-3, 'rectify': True, 'adam_debias': True}, 15),
    (Lamb, {'lr': 1e0, 'weight_decay': 1e-3, 'rectify': True, 'adam_debias': True}, 30),
    (RAdam, {'lr': 1e0, 'weight_decay': 1e-3, 'adam_debias': True}, 25),
    (Ranger, {'lr': 5e0, 'weight_decay': 1e-3, 'adam_debias': True}, 50),
    (Ranger21, {'lr': 1e0, 'weight_decay': 1e-3, 'adam_debias': True, 'num_iterations': 125}, 125),
    (AdaPNM, {'lr': 1e0, 'weight_decay': 1e-3, 'adam_debias': True}, 10),
    (NovoGrad, {'lr': 1e0, 'weight_decay': 1e-3, 'adam_debias': True}, 5),
    (AdaNorm, {'lr': 1e0, 'weight_decay': 1e-3, 'adam_debias': True}, 5),
    (Yogi, {'lr': 1e0, 'weight_decay': 1e-3, 'adam_debias': True}, 5),
    (AdaMod, {'lr': 1e2, 'weight_decay': 1e-3, 'adam_debias': True}, 20),
    (AdaMax, {'lr': 1e0, 'weight_decay': 1e-3, 'adam_debias': True}, 5),
    (AvaGrad, {'lr': 1e1, 'weight_decay': 1e-3, 'adam_debias': True}, 5),
    (AdaHessian, {'lr': 5e0, 'weight_decay': 1e-3, 'adam_debias': True}, 5),
]
