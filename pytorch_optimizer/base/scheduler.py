from abc import ABC, abstractmethod

from torch.optim import Optimizer

from pytorch_optimizer.base.exception import NegativeLRError, NegativeStepError


class BaseLinearWarmupScheduler(ABC):
    """Base scheduler for linear warmup followed by a learning rate schedule.

    Each call to `step()` sets the same learning rate for all parameter groups.

    Args:
        optimizer: Optimizer whose learning rate to update.
        t_max: Total number of scheduler steps, including warmup.
        max_lr: Learning rate at the end of warmup.
        min_lr: Baseline learning rate and initial rate before the first step.
        init_lr: Learning rate at the first warmup step.
        warmup_steps: Number of steps to increase the rate from `init_lr` to `max_lr`.

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
        self.base_lrs: list[float] = []

        # record current value in self._last_lr to match API from torch.optim.lr_scheduler
        self.last_lr: list[float] = [init_lr]

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
        """Advance the schedule and update all parameter groups.

        Returns:
            float: Learning rate for this step.

        """
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
        """Return the learning rate from the most recent scheduler step."""
        return self.last_lr[0]
