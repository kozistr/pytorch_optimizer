"""
PyTorch Hub models
Usage:
    import torch
    optimizer = torch.hub.load('kozistr/pytorch_optimizer', 'adamp')
"""
from functools import partial as _partial
from functools import update_wrapper as _update_wrapper

from pytorch_optimizer import get_supported_lr_schedulers as _get_supported_lr_schedulers
from pytorch_optimizer import get_supported_optimizers as _get_supported_optimizers
from pytorch_optimizer import load_lr_scheduler as _load_lr_scheduler
from pytorch_optimizer import load_optimizer as _load_optimizer

dependencies = ['torch']

for _optimizer in _get_supported_optimizers():
    name: str = _optimizer.__name__
    _func = _partial(_load_optimizer, optimizer=name)
    _update_wrapper(_func, _optimizer.__init__)
    for n in (name, name.lower(), name.upper()):
        globals()[n] = _func

for _scheduler in _get_supported_lr_schedulers():
    name: str = _scheduler.__name__
    _func = _partial(_load_lr_scheduler, lr_scheduler=name)
    _update_wrapper(_func, _scheduler.__init__)
    for n in (name, name.lower(), name.upper()):
        globals()[n] = _func
