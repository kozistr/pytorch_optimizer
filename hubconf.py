dependencies = ['torch']

from functools import partial as _partial, update_wrapper as _update_wrapper

from pytorch_optimizer import (
    get_supported_optimizers as _get_supported_optimizers,
    load_optimizer as _load_optimizer,
)

for optimizer in _get_supported_optimizers():
    name = optimizer.__name__
    for n in (name, name.lower()):
        func = _partial(_load_optimizer, optimizer=n)
        _update_wrapper(func, optimizer)
        globals()[n] = func
