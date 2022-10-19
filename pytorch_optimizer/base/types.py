from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch

CLOSURE = Optional[Callable[[], float]]
LOSS = Optional[float]
BETAS = Union[Tuple[float, float], Tuple[float, float, float]]
DEFAULTS = Dict[str, Any]
PARAMETERS = Optional[Union[Iterable[Dict[str, Any]], Iterable[torch.Tensor]]]
STATE = Dict[str, Any]
