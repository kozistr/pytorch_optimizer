from typing import List

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
ADAPTIVE_FLAGS: List[bool] = [True, False]
PULLBACK_MOMENTUM: List[str] = ['none', 'reset', 'pullback']
VALID_LR_SCHEDULER_NAMES: List[str] = [
    'CosineAnnealingWarmupRestarts',
]
INVALID_LR_SCHEDULER_NAMES: List[str] = [
    'dummy',
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
