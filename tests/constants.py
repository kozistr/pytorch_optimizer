from typing import Any, Dict, List, Tuple, Union

from pytorch_optimizer.optimizer import (
    ADOPT,
    APOLLO,
    ASGD,
    CAME,
    FOCUS,
    FTRL,
    LARS,
    MADGRAD,
    MARS,
    MSVAG,
    OPTIMIZERS,
    PID,
    PNM,
    QHM,
    SGDP,
    SGDW,
    SM3,
    SOAP,
    SPAM,
    SRMM,
    SWATS,
    TAM,
    A2Grad,
    AccSGD,
    AdaBelief,
    AdaBound,
    AdaDelta,
    AdaFactor,
    AdaHessian,
    Adai,
    Adalite,
    AdaMax,
    AdamG,
    AdaMod,
    AdamP,
    AdamS,
    Adan,
    AdaNorm,
    AdaPNM,
    AdaShift,
    AdaSmooth,
    AdaTAM,
    AdEMAMix,
    AggMo,
    Aida,
    AliG,
    Amos,
    ApolloDQN,
    AvaGrad,
    DAdaptAdaGrad,
    DAdaptAdam,
    DAdaptAdan,
    DAdaptLion,
    DAdaptSGD,
    DiffGrad,
    EXAdam,
    FAdam,
    Fromage,
    GaLore,
    Grams,
    Gravity,
    GrokFastAdamW,
    Kate,
    Kron,
    Lamb,
    LaProp,
    Lion,
    Muon,
    Nero,
    NovoGrad,
    PAdam,
    Prodigy,
    QHAdam,
    RAdam,
    Ranger,
    Ranger21,
    Ranger25,
    ScalableShampoo,
    ScheduleFreeAdamW,
    ScheduleFreeRAdam,
    ScheduleFreeSGD,
    SGDSaI,
    Shampoo,
    SignSGD,
    SophiaH,
    StableAdamW,
    Tiger,
    Yogi,
)
from tests.utils import build_lookahead, build_orthograd

DECOUPLE_FLAGS: List[bool] = [True, False]
ADAPTIVE_FLAGS: List[bool] = [True, False]
PULLBACK_MOMENTUM: List[str] = ['none', 'reset', 'pullback']

VALID_OPTIMIZER_NAMES: List[str] = list(OPTIMIZERS.keys())
INVALID_OPTIMIZER_NAMES: List[str] = [
    'asam',
    'sam',
    'gsam',
    'wsam',
    'pcgrad',
    'lookahead',
    'trac',
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
    'ranger25',
    'pnm',
    'adapnm',
    'adan',
    'adai',
    'scalableshampoo',
    'dadaptadam',
    'dadaptadan',
    'dadaptlion',
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
    'prodigy',
    'padam',
    'came',
    'aida',
    'galore',
    'adalite',
    'bsam',
    'schedulefreeadamw',
    'fadam',
    'grokfastadamw',
    'stableadamw',
    'adammini',
    'adamg',
    'ademamix',
    'soap',
    'muon',
    'laprop',
    'apollo',
    'mars',
    'adatam',
    'focus',
    'exadam',
]

VALID_LR_SCHEDULER_NAMES: List[str] = [
    'constant',
    'linear',
    'proportion',
    'step',
    'multi_step',
    'multiplicative',
    'cyclic',
    'one_cycle',
    'cosine',
    'poly',
    'cosine_annealing',
    'cosine_annealing_with_warm_restart',
    'cosine_annealing_with_warmup',
    'chebyshev',
    'rex',
    'warmup_stable_decay',
]
INVALID_LR_SCHEDULER_NAMES: List[str] = ['dummy']

OPTIMIZERS: List[Tuple[Any, Dict[str, Union[float, bool, int]], int]] = [
    (build_lookahead, {'lr': 5e-1, 'weight_decay': 1e-3}, 5),
    (build_orthograd, {'lr': 5e-1, 'weight_decay': 1e-3}, 10),
    (AdaBelief, {'lr': 5e-1, 'weight_decay': 1e-3}, 5),
    (AdaBelief, {'lr': 5e-1, 'weight_decay': 1e-3, 'ams_bound': True}, 5),
    (AdaBelief, {'lr': 5e-1, 'weight_decay': 1e-3, 'weight_decouple': False}, 5),
    (AdaBelief, {'lr': 5e-1, 'weight_decay': 1e-3, 'fixed_decay': True}, 5),
    (AdaBelief, {'lr': 5e-1, 'weight_decay': 1e-3, 'rectify': True}, 10),
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
    (Lamb, {'lr': 5e-2, 'weight_decay': 1e-3, 'grad_averaging': False}, 5),
    (Lamb, {'lr': 1e-1, 'weight_decay': 1e-3, 'adam': True, 'eps': 1e-8}, 5),
    (Lamb, {'lr': 1e-1, 'weight_decay': 1e-3, 'pre_norm': True, 'eps': 1e-8}, 5),
    (Lamb, {'lr': 5e-1, 'weight_decay': 1e-3, 'rectify': True, 'degenerated_to_sgd': True}, 5),
    (LARS, {'lr': 5e-1, 'weight_decay': 1e-3}, 20),
    (LARS, {'lr': 5e-1, 'nesterov': True}, 20),
    (MADGRAD, {'lr': 5e-1, 'weight_decay': 1e-3}, 10),
    (MADGRAD, {'lr': 5e-1, 'weight_decay': 1e-3, 'eps': 0.0}, 10),
    (MADGRAD, {'lr': 1e-1, 'weight_decay': 1e-3, 'momentum': 0.0}, 10),
    (MADGRAD, {'lr': 5e-1, 'weight_decay': 1e-3, 'weight_decouple': True}, 10),
    (RAdam, {'lr': 5e0, 'weight_decay': 1e-3}, 10),
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
    (DAdaptLion, {'lr': 3e0, 'weight_decay': 1e-3}, 20),
    (AdamS, {'lr': 1e0, 'weight_decay': 1e-3}, 10),
    (AdamS, {'lr': 1e0, 'weight_decay': 1e-3, 'ams_bound': True}, 20),
    (AdaFactor, {'lr': 1e1, 'weight_decay': 1e-3, 'scale_parameter': False}, 100),
    (AdaFactor, {'lr': 1e1, 'weight_decay': 1e-3, 'ams_bound': True}, 120),
    (AdaFactor, {'lr': 1e1, 'betas': (None, 0.999), 'weight_decay': 1e-3}, 40),
    (ApolloDQN, {'lr': 5e-1, 'weight_decay': 1e-3}, 10),
    (ApolloDQN, {'lr': 5e-1, 'weight_decay': 1e-3, 'rebound': 'belief'}, 10),
    (ApolloDQN, {'lr': 5e-1, 'weight_decay': 1e-3, 'weight_decay_type': 'stable', 'warmup_steps': 0}, 50),
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
    (Prodigy, {'lr': 5e1, 'beta3': None, 'weight_decay': 1e-3}, 10),
    (Prodigy, {'lr': 5e0, 'beta3': None, 'weight_decay': 1e-3, 'eps': None}, 15),
    (Prodigy, {'lr': 5e1, 'beta3': 0.999, 'weight_decay': 1e-3}, 10),
    (Prodigy, {'lr': 1e1, 'beta3': 0.999, 'weight_decay': 1e-3, 'bias_correction': True}, 15),
    (Prodigy, {'lr': 1e0, 'beta3': 0.999, 'weight_decay': 1e-3, 'safeguard_warmup': True}, 15),
    (PAdam, {'lr': 1e0, 'weight_decay': 1e-3}, 5),
    (Tiger, {'lr': 1e0, 'weight_decay': 1e-3}, 5),
    (CAME, {'lr': 7.5e-1, 'weight_decay': 1e-3}, 70),
    (CAME, {'lr': 7.5e-1, 'weight_decay': 1e-3, 'ams_bound': True}, 70),
    (Aida, {'lr': 1e0, 'weight_decay': 1e-3, 'ams_bound': True}, 5),
    (
        GaLore,
        {'lr': 1e0, 'weight_decay': 1e-3, 'rank': 2, 'scale': 1.0, 'update_proj_gap': 1, 'projection_type': 'std'},
        5,
    ),
    (
        GaLore,
        {
            'lr': 1e0,
            'weight_decay': 1e-3,
            'rank': 2,
            'scale': 1.0,
            'update_proj_gap': 1,
            'projection_type': 'reverse_std',
        },
        5,
    ),
    (
        GaLore,
        {'lr': 5e-1, 'weight_decay': 1e-3, 'rank': 2, 'scale': 1.0, 'update_proj_gap': 2, 'projection_type': 'left'},
        5,
    ),
    (
        GaLore,
        {'lr': 1e0, 'weight_decay': 1e-3, 'rank': 2, 'scale': 1.0, 'update_proj_gap': 1, 'projection_type': 'right'},
        5,
    ),
    (
        GaLore,
        {'lr': 5e-1, 'weight_decay': 1e-3, 'rank': 2, 'scale': 1.0, 'update_proj_gap': 2, 'projection_type': 'full'},
        5,
    ),
    (
        GaLore,
        {'lr': 1e0, 'weight_decay': 1e-3, 'rank': 2, 'scale': 1.0, 'update_proj_gap': 1, 'projection_type': 'random'},
        5,
    ),
    (Adalite, {'lr': 1e0, 'weight_decay': 1e-3}, 5),
    (ScheduleFreeSGD, {'lr': 1e0, 'weight_decay': 1e-3}, 5),
    (ScheduleFreeAdamW, {'lr': 1e0, 'weight_decay': 1e-3}, 5),
    (ScheduleFreeAdamW, {'lr': 1e-2, 'weight_decay': 1e-3, 'use_palm': True}, 5),
    (ScheduleFreeRAdam, {'lr': 1e0, 'weight_decay': 1e-3, 'degenerated_to_sgd': True}, 5),
    (ScheduleFreeRAdam, {'lr': 1e0, 'weight_decay': 1e-3, 'use_palm': True, 'degenerated_to_sgd': True}, 5),
    (FAdam, {'lr': 1e0, 'weight_decay': 1e-3}, 5),
    (GrokFastAdamW, {'lr': 5e0, 'weight_decay': 1e-3, 'grokfast_after_step': 1}, 5),
    (Kate, {'lr': 5e-2}, 10),
    (StableAdamW, {'lr': 1e0}, 5),
    (AdamG, {'lr': 1e0, 'p': 0.5}, 60),
    (AdEMAMix, {'lr': 1e0}, 3),
    (AdEMAMix, {'lr': 1e0, 't_alpha_beta3': 5}, 3),
    (
        SOAP,
        {'lr': 1e0, 'shampoo_beta': 0.95, 'precondition_frequency': 1, 'merge_dims': False, 'precondition_1d': True},
        5,
    ),
    (ADOPT, {'lr': 1e0}, 5),
    (FTRL, {'lr': 1e0, 'beta': 0.0, 'lambda_1': 0.0, 'lambda_2': 0.0}, 5),
    (Muon, {'lr': 1e0, 'ns_steps': 6, 'adam_lr': 1e0, 'adamw_wd': 1e-2}, 5),
    (Muon, {'lr': 1e0, 'ns_steps': 6, 'adam_lr': 1e0, 'adamw_wd': 1e-2, 'nesterov': False}, 5),
    (LaProp, {'lr': 1e0, 'weight_decay': 1e-3}, 5),
    (LaProp, {'lr': 1e0, 'centered': True, 'weight_decay': 1e-3}, 11),
    (LaProp, {'lr': 1e0, 'ams_bound': True, 'weight_decay': 1e-3}, 5),
    (
        APOLLO,
        {
            'lr': 1e-1,
            'weight_decay': 1e-3,
            'rank': 2,
            'update_proj_gap': 1,
            'scale': 1.0,
            'scale_type': 'tensor',
            'projection_type': 'right',
        },
        15,
    ),
    (
        APOLLO,
        {
            'lr': 1e-1,
            'weight_decay': 1e-3,
            'rank': 2,
            'update_proj_gap': 1,
            'scale': 1.0,
            'scale_type': 'channel',
            'projection_type': 'right',
        },
        15,
    ),
    (MARS, {'lr': 5e-1, 'lr_1d': 5e-1, 'weight_decay': 1e-3, 'mars_type': 'adamw'}, 5),
    (MARS, {'lr': 1e-1, 'weight_decay': 1e-3, 'mars_type': 'lion', 'optimize_1d': True}, 5),
    (MARS, {'lr': 5e-1, 'lr_1d': 5e-1, 'weight_decay': 1e-3, 'mars_type': 'shampoo'}, 5),
    (MARS, {'lr': 5e-1, 'lr_1d': 5e-1, 'weight_decay': 1e-3, 'mars_type': 'adamw', 'ams_bound': True}, 5),
    (SGDSaI, {'lr': 1e0}, 15),
    (SGDSaI, {'lr': 1e0, 'momentum': 0.0}, 15),
    (Grams, {'lr': 1e-1, 'weight_decay': 1e-3}, 5),
    (SPAM, {'lr': 1e0, 'weight_decay': 1e-3, 'warmup_epoch': 1, 'grad_accu_steps': 1, 'update_proj_gap': 1}, 5),
    (TAM, {'lr': 1e0, 'weight_decay': 1e-3}, 5),
    (AdaTAM, {'lr': 1e-1, 'weight_decay': 1e-3}, 5),
    (FOCUS, {'lr': 1e-1, 'weight_decay': 1e-3}, 5),
    (Kron, {'lr': 1e0, 'weight_decay': 1e-3}, 3),
    (EXAdam, {'lr': 1e-1, 'weight_decay': 1e-3}, 5),
    (Ranger25, {'lr': 5e0}, 2),
    (Ranger25, {'lr': 5e0, 't_alpha_beta3': 5}, 2),
    (Ranger25, {'lr': 2e-1, 'stable_adamw': False, 'orthograd': False, 'eps': None}, 3),
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
    (Aida, {'lr': 1e0, 'weight_decay': 1e-3, 'adanorm': True}, 5),
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
    (
        Ranger21,
        {'lr': 5e-1, 'weight_decay': 1e-3, 'adam_debias': True, 'num_iterations': 125, 'disable_lr_scheduler': True},
        125,
    ),
    (AdaPNM, {'lr': 1e0, 'weight_decay': 1e-3, 'adam_debias': True}, 10),
    (NovoGrad, {'lr': 1e0, 'weight_decay': 1e-3, 'adam_debias': True}, 5),
    (AdaNorm, {'lr': 1e0, 'weight_decay': 1e-3, 'adam_debias': True}, 5),
    (Yogi, {'lr': 1e0, 'weight_decay': 1e-3, 'adam_debias': True}, 5),
    (AdaMod, {'lr': 1e2, 'weight_decay': 1e-3, 'adam_debias': True}, 20),
    (AdaMax, {'lr': 1e0, 'weight_decay': 1e-3, 'adam_debias': True}, 5),
    (AvaGrad, {'lr': 1e1, 'weight_decay': 1e-3, 'adam_debias': True}, 5),
    (AdaHessian, {'lr': 5e0, 'weight_decay': 1e-3, 'adam_debias': True}, 5),
    (Aida, {'lr': 1e1, 'weight_decay': 1e-3, 'rectify': True, 'adam_debias': True}, 10),
]
COPT_SUPPORTED_OPTIMIZERS: List[Tuple[Any, Dict[str, Union[float, bool, int]], int]] = [
    (AdaFactor, {'lr': 1e1, 'weight_decay': 1e-3, 'cautious': True}, 70),
    (Lion, {'lr': 5e-1, 'weight_decay': 1e-3, 'cautious': True}, 5),
    (AdEMAMix, {'lr': 1e0, 'cautious': True}, 2),
    (LaProp, {'lr': 1e0, 'cautious': True}, 2),
    (AdamP, {'lr': 1e0, 'cautious': True}, 2),
    (ADOPT, {'lr': 1e1, 'cautious': True}, 3),
    (AdaShift, {'lr': 1e1, 'keep_num': 1, 'cautious': True}, 3),
    (MARS, {'lr': 5e-1, 'lr_1d': 5e-1, 'weight_decay': 1e-3, 'cautious': True}, 3),
    (MARS, {'lr': 5e-1, 'lr_1d': 5e-1, 'weight_decay': 1e-3, 'optimize_1d': True, 'cautious': True}, 3),
]
STABLE_ADAMW_SUPPORTED_OPTIMIZERS: List[Tuple[Any, Dict[str, Union[float, bool, int]], int]] = [
    (ADOPT, {'lr': 1e0, 'weight_decay': 1e-3, 'stable_adamw': True}, 5),
    (AdEMAMix, {'lr': 1e0, 'weight_decay': 1e-3, 'stable_adamw': True}, 10),
]
