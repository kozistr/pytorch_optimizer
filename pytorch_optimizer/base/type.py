from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Literal, Optional, Tuple, Type, Union

if TYPE_CHECKING:  # pragma: no cover
    from typing import TypeAlias
else:  # pragma: no cover
    TypeAlias = object

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

OptimizerType: TypeAlias = Type[Optimizer]
OptimizerInstanceOrClass: TypeAlias = Union[OptimizerType, Optimizer]
SchedulerClass: TypeAlias = Type[LRScheduler]

Defaults: TypeAlias = Dict[str, Any]
ParamGroup: TypeAlias = Dict[str, Any]
State: TypeAlias = Dict
ParamsT: TypeAlias = Union[
    Iterable[torch.Tensor],
    Iterable[Dict[str, Any]],
    Iterable[Tuple[str, torch.Tensor]],
]

Closure: TypeAlias = Optional[Callable[[], float]]
Loss: TypeAlias = Optional[float]
Betas: TypeAlias = Union[
    Tuple[float, float],
    Tuple[float, float, float],
]

HutchinsonG: TypeAlias = Literal['gaussian', 'rademacher']
ClassMode: TypeAlias = Literal['binary', 'multiclass', 'multilabel']

DataFormat: TypeAlias = Literal['channels_first', 'channels_last']
