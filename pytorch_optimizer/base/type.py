from collections.abc import Callable, Iterable
from typing import Any, Literal, TypeAlias

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

OptimizerType: TypeAlias = type[Optimizer]
OptimizerInstanceOrClass: TypeAlias = OptimizerType | Optimizer
SchedulerClass: TypeAlias = type[LRScheduler]

Defaults: TypeAlias = dict[str, Any]
ParamGroup: TypeAlias = dict[str, Any]
State: TypeAlias = dict
ParamsT: TypeAlias = Iterable[torch.Tensor] | Iterable[dict[str, Any]] | Iterable[tuple[str, torch.Tensor]]

Closure: TypeAlias = Callable[[], float] | None
Loss: TypeAlias = float | None
Betas: TypeAlias = tuple[float, float] | tuple[float, float, float]

HutchinsonG: TypeAlias = Literal['gaussian', 'rademacher']
ClassMode: TypeAlias = Literal['binary', 'multiclass', 'multilabel']

DataFormat: TypeAlias = Literal['channels_first', 'channels_last']
