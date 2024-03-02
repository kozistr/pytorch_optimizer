from abc import ABC, abstractmethod
from typing import List

from pytorch_optimizer.base.exception import NegativeLRError, NegativeStepError
from pytorch_optimizer.base.types import OPTIMIZER


class BaseLinearWarmupScheduler(ABC):
    r"""BaseLinearWarmupScheduler class.

        The LR Scheduler class based on this class has linear warmup strategy.

    :param optimizer: Optimizer. OPTIMIZER. It will set learning rate to all trainable parameters in optimizer.
    :param t_max: int. total steps to train.
    :param max_lr: float. maximum lr.
    :param min_lr: float. minimum lr.
    :param init_lr: float. initial lr.
    :param warmup_steps: int. steps to warm-up.
    """

    def __init__(
        self,
        optimizer: OPTIMIZER,
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
