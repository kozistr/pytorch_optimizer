from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytorch_optimizer.optimizer.shampoo.kl import KLSOAP, KLShampoo
    from pytorch_optimizer.optimizer.shampoo.shampoo import ScalableShampoo, Shampoo

__all__ = ['KLSOAP', 'KLShampoo', 'ScalableShampoo', 'Shampoo']


def __getattr__(name: str):
    """Load optimizer classes after the shared utility imports finish."""
    module = 'kl' if name in ('KLSOAP', 'KLShampoo') else 'shampoo'
    optimizer = getattr(import_module(f'{__name__}.{module}'), name)
    globals()[name] = optimizer

    return optimizer
