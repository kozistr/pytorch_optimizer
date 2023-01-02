from typing import List

ADAPTIVE_FLAGS: List[bool] = [True, False]
PULLBACK_MOMENTUM: List[str] = ['none', 'reset', 'pullback']

SPARSE_OPTIMIZERS: List[str] = [
    'madgrad',
]
NO_SPARSE_OPTIMIZERS: List[str] = [
    'adamp',
    'sgdp',
    'madgrad',
    'ranger',
    'ranger21',
    'radam',
    'adabound',
    'adabelief',
    'diffgrad',
    'diffrgrad',
    'lamb',
    'ralamb',
    'lars',
    'shampoo',
    'nero',
    'adan',
    'adai',
    'adapnm',
    'pnm',
]
VALID_OPTIMIZER_NAMES: List[str] = [
    'adamp',
    'adan',
    'sgdp',
    'madgrad',
    'ranger',
    'ranger21',
    'radam',
    'adabound',
    'adabelief',
    'diffgrad',
    'diffrgrad',
    'lamb',
    'ralamb',
    'lars',
    'shampoo',
    'pnm',
    'adapnm',
    'nero',
    'adai',
]
INVALID_OPTIMIZER_NAMES: List[str] = [
    'asam',
    'sam',
    'pcgrad',
    'adamd',
    'lookahead',
    'chebyshev_schedule',
]
BETA_OPTIMIZER_NAMES: List[str] = [
    'adabelief',
    'adabound',
    'adamp',
    'diffgrad',
    'diffrgrad',
    'lamb',
    'radam',
    'ranger',
    'ranger21',
    'ralamb',
    'pnm',
    'adapnm',
    'adan',
    'adai',
]

VALID_LR_SCHEDULER_NAMES: List[str] = [
    'CosineAnnealingWarmupRestarts',
]
INVALID_LR_SCHEDULER_NAMES: List[str] = [
    'dummy',
]
