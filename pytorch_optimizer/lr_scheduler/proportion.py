from typing import List


class ProportionScheduler:
    r"""ProportionScheduler (Rho Scheduler of GSAM)
        This scheduler outputs a value that evolves proportional to lr_scheduler.

    :param lr_scheduler: learning rate scheduler.
    :param max_lr: float. maximum lr.
    :param min_lr: float. minimum lr.
    :param max_value: float. maximum of rho.
    :param min_value: float. minimum of rho.
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
