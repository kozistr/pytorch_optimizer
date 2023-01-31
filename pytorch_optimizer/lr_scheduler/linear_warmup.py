import math

import numpy as np

from pytorch_optimizer.base.scheduler import BaseLinearWarmupScheduler


class LinearScheduler(BaseLinearWarmupScheduler):
    r"""Linear LR Scheduler w/ linear warmup."""

    def _step(self) -> float:
        return self.max_lr + (self.min_lr - self.max_lr) * (self.step_t - self.warmup_steps) / (
            self.total_steps - self.warmup_steps
        )


class CosineScheduler(BaseLinearWarmupScheduler):
    r"""Cosine LR Scheduler w/ linear warmup."""

    def _step(self) -> float:
        phase: float = (self.step_t - self.warmup_steps) / (self.total_steps - self.warmup_steps) * math.pi
        return self.min_lr + (self.max_lr - self.min_lr) * (np.cos(phase) + 1.0) / 2.0


class PolyScheduler(BaseLinearWarmupScheduler):
    r"""Poly LR Scheduler.

    :param: poly_order: float. lr scheduler decreases with steps.
    """

    def __init__(self, poly_order: float = 0.5, **kwargs):
        self.poly_order = poly_order

        if poly_order <= 0:
            raise ValueError(f'[-] poly_order must be positive. {poly_order}')

        super().__init__(**kwargs)

    def _step(self) -> float:
        return self.min_lr + (self.max_lr - self.min_lr) * (self.step_t - self.warmup_steps) ** self.poly_order
