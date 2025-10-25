from typing import List


class ProportionScheduler:
    r"""ProportionScheduler (Rho Scheduler of GSAM).

    This scheduler outputs a value that evolves proportionally to a given learning rate scheduler.

    Args:
        lr_scheduler: Learning rate scheduler.
        max_lr (float): Maximum learning rate.
        min_lr (float): Minimum learning rate.
        max_value (float): Maximum value of rho.
        min_value (float): Minimum value of rho.
    """

    def __init__(
        self, lr_scheduler, max_lr: float, min_lr: float = 0.0, max_value: float = 2.0, min_value: float = 2.0
    ):
        self.lr_scheduler = lr_scheduler
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.max_value = max_value
        self.min_value = min_value

        self.step_t: int = 0
        self.last_lr: List[float] = []

        self.step()

    def get_lr(self) -> float:
        return self.last_lr[0]

    def step(self) -> float:
        self.step_t += 1

        if hasattr(self.lr_scheduler, 'last_lr'):
            lr = self.lr_scheduler.last_lr[0]
        else:
            lr = self.lr_scheduler.optimizer.param_groups[0]['lr']

        if self.max_lr > self.min_lr:
            value = self.min_value + (self.max_value - self.min_value) * (lr - self.min_lr) / (
                self.max_lr - self.min_lr
            )
        else:
            value = self.max_value

        self.last_lr = [value]

        return value
