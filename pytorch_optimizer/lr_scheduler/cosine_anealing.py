import math
from typing import List, Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CosineAnnealingWarmupRestarts(LRScheduler):
    r"""CosineAnnealingWarmupRestarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer instance.
        first_cycle_steps (int): Number of steps in the first cycle.
        cycle_mult (float): Cycle steps magnification factor.
        max_lr (float): Maximum learning rate.
        min_lr (float): Minimum learning rate.
        warmup_steps (int): Number of warmup steps.
        gamma (float): Decrease rate of max learning rate by cycle.
        last_epoch (int): The index of the last epoch for resuming training.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 1e-4,
        min_lr: float = 1e-6,
        warmup_steps: int = 0,
        gamma: float = 0.9,
        last_epoch: int = -1,
    ):
        if warmup_steps >= first_cycle_steps:
            raise ValueError(
                f'warmup_steps must be smaller than first_cycle_steps. {warmup_steps} < {first_cycle_steps}'
            )

        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.cur_cycle_steps = first_cycle_steps
        self.step_in_cycle = last_epoch
        self.last_epoch = last_epoch

        self.cycle: int = 0
        self.base_lrs: List[float] = []

        super().__init__(optimizer, last_epoch)

        self.init_lr()

    def init_lr(self) -> None:
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self) -> List[float]:
        if self.step_in_cycle == -1:
            return self.base_lrs

        if self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs
            ]

        return [
            base_lr
            + (self.max_lr - base_lr)
            * (
                1
                + math.cos(
                    math.pi * (self.step_in_cycle - self.warmup_steps) / (self.cur_cycle_steps - self.warmup_steps)
                )
            )
            / 2.0
            for base_lr in self.base_lrs
        ]

    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
                )
        elif epoch >= self.first_cycle_steps:
            if self.cycle_mult == 1.0:
                self.step_in_cycle = epoch % self.first_cycle_steps
                self.cycle = epoch // self.first_cycle_steps
            else:
                n: int = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                self.cycle = n
                self.step_in_cycle = epoch - int(
                    self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1)
                )  # fmt: skip
                self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** n  # fmt: skip
        else:
            self.cur_cycle_steps = self.first_cycle_steps
            self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)  # fmt: skip
        self.last_epoch = math.floor(epoch)

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
