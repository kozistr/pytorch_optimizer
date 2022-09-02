"""
PyTorch Hub models
Usage:
    import torch
    optimizer = torch.hub.load('kozistr/pytorch_optimizer', 'adamp')
"""
from functools import partial as _partial
from functools import update_wrapper as _update_wrapper

from pytorch_optimizer import get_supported_optimizers as _get_supported_optimizers
from pytorch_optimizer import load_optimizer as _load_optimizer

dependencies = ['torch']

for optimizer in _get_supported_optimizers():
    name: str = optimizer.__name__
    for n in (name, name.lower(), name.upper()):
        func = _partial(_load_optimizer, optimizer=n)
        _update_wrapper(func, optimizer)
        globals()[n] = func
