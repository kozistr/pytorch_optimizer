from abc import ABC, abstractmethod
from typing import List

from torch.optim import Optimizer

from pytorch_optimizer.base.exception import NegativeLRError, NegativeStepError


class BaseLinearWarmupScheduler(ABC):
    """BaseLinearWarmupScheduler class.

    A learning rate scheduler class that implements a linear warmup strategy.

    Args:
        optimizer (Optimizer): The optimizer whose learning rate will be scheduled.
            It will set the learning rate to all trainable parameters in the optimizer.
        t_max (int): Total number of training steps (epochs or iterations).
        max_lr (float): The maximum learning rate after warmup.
        min_lr (float): The minimum learning rate to decay to (or start from if warmup).
        init_lr (float): Initial learning rate at the start of warmup.
        warmup_steps (int): Number of steps to warm up linearly from init_lr to max_lr.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        t_max: int,
        max_lr: float,
        min_lr: float = 0.0,
        init_lr: float = 0.0,
        warmup_steps: int = 0,
    ):
        self.optimizer = optimizer
        self.total_steps = t_max
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps

        self.step_t: int = 0
        self.base_lrs: List[float] = []

        # record current value in self._last_lr to match API from torch.optim.lr_scheduler
        self.last_lr: List[float] = [init_lr]

        self.validate_parameters()

        self._init_lr()

    def validate_parameters(self):
        if self.min_lr < 0:
            raise NegativeLRError(self.min_lr, 'min_lr')

        if self.max_lr < 0:
            raise NegativeLRError(self.max_lr, 'max_lr')

        if self.init_lr < 0:
            raise NegativeLRError(self.init_lr, 'init_lr')

        if self.total_steps < 0:
            raise NegativeStepError(self.total_steps, 't_max')

        if self.warmup_steps < 0:
            raise NegativeStepError(self.warmup_steps, 'warmup_steps')

    def _init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def step(self):
        if self.step_t < self.warmup_steps:
            value = self.init_lr + (self.max_lr - self.init_lr) * self.step_t / self.warmup_steps
        elif self.step_t == self.warmup_steps:
            value = self.max_lr
        else:
            value = self._step()

        self.step_t += 1

        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = value

        self.last_lr = [value]

        return value

    @abstractmethod
    def _step(self) -> float:  # pragma: no cover
        raise NotImplementedError

    def get_lr(self) -> float:
        return self.last_lr[0]
