from functools import cache
from inspect import signature

import numpy as np
import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.optimizer import OPTIMIZERS
from tests.utils import build_optimizer

VALID_OPTIMIZER_NAMES = tuple(OPTIMIZERS)


def optimizers_with_argument(argument: str) -> frozenset[str]:
    return frozenset(name for name, cls in OPTIMIZERS.items() if argument in signature(cls).parameters)


FOREACH_OPTIMIZERS = optimizers_with_argument('foreach')
MAXIMIZE_OPTIMIZERS = optimizers_with_argument('maximize')
BETA_OPTIMIZER_NAMES = optimizers_with_argument('betas')
MODEL_OPTIMIZERS = optimizers_with_argument('model')


GRADIENT_OPTIONS = {
    'lookahead': {'k': 1},
    'ranger21': {'lookahead_merge_time': 1},
    'lamb': {'pre_norm': True},
    'alice': {'rank': 2, 'leading_basis': 1},
    'adahessian': {'update_period': 2},
}


# Model-based, distributed, and closure-driven optimizers cannot use a plain parameter probe.
SKIP_CAPABILITY_PROBE = MODEL_OPTIMIZERS | {'lbfgs', 'demo', 'distributedmuon', 'bsam'}


@cache
def supports_gradient(optimizer_name: str, kind: str) -> bool:
    if kind not in ('complex', 'sparse'):
        raise ValueError(f'Unknown gradient kind: {kind}')

    if optimizer_name in SKIP_CAPABILITY_PROBE:
        return False

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)

        dtype = torch.complex64 if kind == 'complex' else torch.float32

        param = torch.ones(2, 2, dtype=dtype, requires_grad=True)
        param.grad = torch.ones_like(param)
        if kind == 'sparse':
            param.grad = torch.eye(2).to_sparse()

        options = {**GRADIENT_OPTIONS.get(optimizer_name, {})}
        if optimizer_name == 'madgrad':
            options['momentum'] = 0.0
        if optimizer_name in ('muon', 'adamuon', 'adago', 'normuon'):
            options['use_muon'] = True

        optimizer = build_optimizer(optimizer_name, [param], **options)
        unsupported_error = NoComplexParameterError if kind == 'complex' else NoSparseGradientError

        rng_state = np.random.get_state()

        try:
            optimizer.step(lambda: 0.1)
        except unsupported_error:
            return False
        except RuntimeError as error:
            if kind != 'sparse' or 'does not support sparse gradients' not in str(error):
                raise
            return False
        finally:
            np.random.set_state(rng_state)

        assert torch.isfinite(param).all(), f'{optimizer_name} produced nonfinite {kind} updates'

        return True


COMPLEX_OPTIMIZERS = frozenset(name for name in VALID_OPTIMIZER_NAMES if supports_gradient(name, 'complex'))
SPARSE_OPTIMIZERS = frozenset(name for name in VALID_OPTIMIZER_NAMES if supports_gradient(name, 'sparse'))


NATIVE_OPTIMIZERS = frozenset(name for name, cls in OPTIMIZERS.items() if cls.__module__.startswith('torch.optim'))
SKIP_LEARNING_RATE = NATIVE_OPTIMIZERS | (frozenset(OPTIMIZERS) - optimizers_with_argument('lr')) | {'a2grad'}
SKIP_EPSILON = frozenset(OPTIMIZERS) - optimizers_with_argument('eps') - {'distributedmuon'}

# SCION variants expose weight decay but do not validate it in their constructors.
SKIP_WEIGHT_DECAY = (frozenset(OPTIMIZERS) - optimizers_with_argument('weight_decay')) | {'scion', 'scionlight'}
SKIP_CREATE_OPTIMIZER = NATIVE_OPTIMIZERS | {'demo', 'distributedmuon'}


SKIP_NO_GRADIENT_TEST = SKIP_CAPABILITY_PROBE - {'bsam'}
SKIP_BF16_OPTIMIZERS = frozenset({'adai', 'prodigy', 'nero', 'lorarite'})
