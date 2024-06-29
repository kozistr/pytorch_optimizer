import math
from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


def get_wsd_scheduler_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
    min_lr_ratio: float,
    num_cycles: float,
) -> float:
    r"""Get WSD learning rate.

    :param current_step: int. the number of current steps.
    :param num_warmup_steps: int. the number of warmup steps.
    :param num_stable_steps: int. the number of stable steps.
    :param num_decay_steps: int. the number of decay steps.
    :param min_lr_ratio: float. the minimum learning rate as a ratio of the initial learning rate.
    :param num_cycles: float. the number of waves in the cosine schedule (the defaults is to just decrease from the max
        value to 0 following a half-cosine)
    """
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    if current_step < num_warmup_steps + num_stable_steps:
        return 1.0
    if current_step < num_warmup_steps + num_stable_steps + num_decay_steps:
        progress = float(current_step - num_warmup_steps - num_stable_steps) / float(max(1, num_decay_steps))
        value = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        return (1.0 - min_lr_ratio) * value + min_lr_ratio
    return min_lr_ratio


def get_wsd_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LRScheduler:
    r"""Get Warmup-Stable-Decay learning rate scheduler.

    :param optimizer: Optimizer. the optimizer for which to schedule the learning rate.
    :param num_warmup_steps: int. the number of warmup steps.
    :param num_stable_steps: int. the number of stable steps.
    :param num_decay_steps: int. the number of decay steps.
    :param min_lr_ratio: float. the minimum learning rate as a ratio of the initial learning rate.
    :param num_cycles: float. the number of waves in the cosine schedule (the defaults is to just decrease from the max
        value to 0 following a half-cosine)
    :param last_epoch: int. the index of the last epoch when resuming training.
    """
    lr_scheduler = partial(
        get_wsd_scheduler_lambda,
        num_warmup_steps=num_warmup_steps,
        num_stable_steps=num_stable_steps,
        num_decay_steps=num_decay_steps,
        min_lr_ratio=min_lr_ratio,
        num_cycles=num_cycles,
    )

    return LambdaLR(optimizer, lr_scheduler, last_epoch)
