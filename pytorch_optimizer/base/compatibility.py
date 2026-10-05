import re
import warnings
from importlib.util import find_spec

import torch


def parse_pytorch_version(version_string: str) -> list[int]:
    """Parse the major, minor, and patch numbers of a PyTorch version string."""
    match = re.match(r'(\d+\.\d+\.\d+)', version_string)
    if not match:
        raise ValueError(f'invalid version string format: {version_string}')

    return [int(x) for x in match.group(1).split('.')]


def compare_versions(v1: str, v2: str) -> bool:
    """Return whether PyTorch version `v1` is at least `v2`."""
    return parse_pytorch_version(v1) >= parse_pytorch_version(v2)


HAS_TRANSFORMERS: bool = find_spec('transformers') is not None
TORCH_VERSION_AT_LEAST_2_4: bool = compare_versions(torch.__version__, '2.4.0')
TORCH_VERSION_AT_LEAST_2_8: bool = compare_versions(torch.__version__, '2.8.0')

if HAS_TRANSFORMERS:  # pragma: no cover
    try:
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
    except ImportError:
        from transformers.deepspeed import is_deepspeed_zero3_enabled
else:

    def is_deepspeed_zero3_enabled() -> bool:
        """Check if DeepSpeed zero3 is enabled."""
        if HAS_TRANSFORMERS:
            return is_deepspeed_zero3_enabled()  # pragma: no cover

        warnings.warn(
            'you need to install `transformers` to use `is_deepspeed_zero3_enabled` function. it will return False.',
            category=ImportWarning,
            stacklevel=2,
        )

        return False
