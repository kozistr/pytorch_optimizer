from typing import Any, Callable, Dict, Optional, Tuple

CLOSURE = Optional[Callable[[], float]]
LOSS = Optional[float]
BETAS = Tuple[float, float]
DEFAULT_PARAMETERS = Dict[str, Any]
