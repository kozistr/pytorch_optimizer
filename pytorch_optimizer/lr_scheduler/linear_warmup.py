import math

import numpy as np

from pytorch_optimizer.base.scheduler import BaseLinearWarmupScheduler


class LinearScheduler(BaseLinearWarmupScheduler):
    def _step(self) -> float:
        return self.max_lr + (self.min_lr - self.max_lr) * (self.t - self.warmup_steps) / (
            self.total_steps - self.warmup_steps
        )


class CosineScheduler(BaseLinearWarmupScheduler):
    def _step(self) -> float:
        phase: float = (self.t - self.warmup_steps) / (self.total_steps - self.warmup_steps) * math.pi
        return self.min_lr + (self.max_lr - self.min_lr) * (np.cos(phase) + 1.0) / 2.0


class PolyScheduler(BaseLinearWarmupScheduler):
    r"""Poly LR Scheduler

    :param: poly_order: float. lr scheduler decreases with steps.
    """

    def __init__(self, poly_order: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.poly_order = poly_order

        if poly_order <= 0:
            raise ValueError(f'[-] poly_order must be positive. {poly_order}')

    def _step(self) -> float:
        return self.min_lr + (self.max_lr - self.min_lr) * (self.t - self.warmup_steps) ** self.poly_order
