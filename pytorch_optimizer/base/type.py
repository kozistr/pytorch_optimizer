from typing import Any, Callable, Dict, Iterable, Literal, Optional, Tuple, Type, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

OPTIMIZER = Type[Optimizer]
OPTIMIZER_INSTANCE_OR_CLASS = Union[OPTIMIZER, Optimizer]
SCHEDULER = Type[LRScheduler]

Defaults = Dict[str, Any]
ParamGroup = Dict[str, Any]
State = Dict
Parameters = Optional[Union[Iterable[torch.Tensor], Iterable[ParamGroup]]]

Closure = Optional[Callable[[], float]]
Loss = Optional[float]
Betas = Union[
    Tuple[float, float],
    Tuple[float, float, float],
    Tuple[None, float],
]

HUTCHINSON_G = Literal['gaussian', 'rademacher']
CLASS_MODE = Literal['binary', 'multiclass', 'multilabel']

DATA_FORMAT = Literal['channels_first', 'channels_last']
