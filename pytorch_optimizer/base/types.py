from typing import Callable, Dict, Iterable, Literal, Optional, Tuple, Type, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

CLOSURE = Optional[Callable[[], float]]
LOSS = Optional[float]
BETAS = Union[Tuple[float, float], Tuple[float, float, float], Tuple[None, float]]
DEFAULTS = Dict
PARAMETERS = Optional[Union[Iterable[Dict], Iterable[torch.Tensor]]]
STATE = Dict
OPTIMIZER = Type[Optimizer]
SCHEDULER = Type[LRScheduler]

HUTCHINSON_G = Literal['gaussian', 'rademacher']
CLASS_MODE = Literal['binary', 'multiclass', 'multilabel']

DATA_FORMAT = Literal['channels_first', 'channels_last']
