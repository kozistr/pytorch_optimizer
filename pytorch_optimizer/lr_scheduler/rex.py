from typing import List, Optional

from torch.optim.lr_scheduler import LRScheduler

from pytorch_optimizer.base.types import OPTIMIZER


class REXScheduler(LRScheduler):
    r"""Revisiting Budgeted Training with an Improved Schedule.

    :param optimizer: Optimizer. wrapped optimizer instance.
    :param total_steps: int. number of steps to optimize.
    :param max_lr: float. max lr.
    :param min_lr: float. min lr.
    """

    def __init__(
        self,
        optimizer: OPTIMIZER,
        total_steps: int,
        max_lr: float = 1.0,
        min_lr: float = 0.0,
    ):
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr

        self.step_t: int = 0
        self.base_lrs: List[float] = []

        # record current value in self._last_lr to match API from torch.optim.lr_scheduler
        self.last_lr: List[float] = [self.max_lr]

        super().__init__(optimizer)

        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self) -> float:
        return self.last_lr[0]

    def get_linear_lr(self) -> float:
        if self.step_t >= self.total_steps:
            return self.min_lr

        progress: float = self.step_t / self.total_steps

        return self.min_lr + (self.max_lr - self.min_lr) * ((1.0 - progress) / (1.0 - progress / 2.0))

    def step(self, epoch: Optional[int] = None) -> float:
        value: float = self.get_linear_lr()

        self.step_t += 1

        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = value

        self.last_lr = [value]

        return value
