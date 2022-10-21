from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

CLOSURE = Optional[Callable[[], float]]
LOSS = Optional[float]
BETAS = Union[Tuple[float, float], Tuple[float, float, float]]
DEFAULTS = Dict[str, Any]
PARAMETERS = Optional[Union[Iterable[Dict[str, Any]], Iterable[torch.Tensor]]]
STATE = Dict[str, Any]
OPTIMIZER = Type[Optimizer]
SCHEDULER = Type[_LRScheduler]
