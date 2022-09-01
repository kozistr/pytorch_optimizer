dependencies = ["torch"]

from functools import partial

from pytorch_optimizer import get_supported_optimizers, load_optimizer

for optimizer in get_supported_optimizers():
    name = optimizer.__name__
    for n in (name, name.lower()):
        globals()[n] = partial(load_optimizer, optimizer=n)
