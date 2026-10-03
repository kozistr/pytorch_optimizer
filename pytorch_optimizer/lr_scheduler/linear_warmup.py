import math

import numpy as np

from pytorch_optimizer.base.scheduler import BaseLinearWarmupScheduler


class LinearScheduler(BaseLinearWarmupScheduler):
    """Linear learning rate decay after linear warmup.

    Args:
        optimizer: Optimizer whose learning rate to update.
        t_max: Total scheduler steps, including warmup.
        max_lr: Learning rate at the end of warmup.
        min_lr: Final learning rate.
        init_lr: Learning rate at the first warmup step.
        warmup_steps: Number of linear warmup steps.

    """

    def _step(self) -> float:
        return self.max_lr + (self.min_lr - self.max_lr) * (self.step_t - self.warmup_steps) / (
            self.total_steps - self.warmup_steps
        )


class CosineScheduler(BaseLinearWarmupScheduler):
    """Cosine learning rate decay after linear warmup.

    Args:
        optimizer: Optimizer whose learning rate to update.
        t_max: Total scheduler steps, including warmup.
        max_lr: Learning rate at the end of warmup.
        min_lr: Final learning rate.
        init_lr: Learning rate at the first warmup step.
        warmup_steps: Number of linear warmup steps.

    """

    def _step(self) -> float:
        phase: float = (self.step_t - self.warmup_steps) / (self.total_steps - self.warmup_steps) * math.pi
        return self.min_lr + (self.max_lr - self.min_lr) * (np.cos(phase) + 1.0) / 2.0


class PolyScheduler(BaseLinearWarmupScheduler):
    """Polynomial learning rate schedule after linear warmup.

    After warmup, compute `min_lr + (max_lr - min_lr) * elapsed_steps ** poly_order`.

    Args:
        optimizer (Optimizer): Optimizer whose learning rate to update.
        poly_order: Positive exponent of the polynomial schedule.
        **kwargs (dict): Options for `BaseLinearWarmupScheduler`, including `t_max` and `max_lr`.

    """

    def __init__(self, optimizer, poly_order: float = 0.5, **kwargs):
        self.poly_order = poly_order

        if poly_order <= 0:
            raise ValueError(f'poly_order must be positive. {poly_order}')

        super().__init__(optimizer, **kwargs)

    def _step(self) -> float:
        return self.min_lr + (self.max_lr - self.min_lr) * (self.step_t - self.warmup_steps) ** self.poly_order
