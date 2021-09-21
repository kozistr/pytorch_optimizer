from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch

CLOSURE = Optional[Callable[[], float]]
LOSS = Optional[float]
BETAS = Tuple[float, float]
DEFAULT_PARAMETERS = Dict[str, Any]
PARAMS = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]
STATE = Dict[str, Any]
