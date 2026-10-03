import math
from functools import partial
from typing import Literal

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

COOLDOWN_TYPE = Literal['cosine', '1-sqrt', 'linear', '1-square']


def get_cosine_cooldown_lr_ratio(
    current_step: int,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
    min_lr_ratio: float,
    num_cycles: float,
) -> float:
    """Return the cosine cooldown learning rate multiplier."""
    progress = float(current_step - num_warmup_steps - num_stable_steps) / float(max(1, num_decay_steps))
    value = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return (1.0 - min_lr_ratio) * value + min_lr_ratio


def get_1sqrt_cooldown_lr_ratio(
    current_step: int,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
) -> float:
    """Return the `1 - sqrt(progress)` cooldown learning rate multiplier."""
    return 1.0 - math.sqrt((current_step - num_warmup_steps - num_stable_steps) / num_decay_steps)


def get_1square_cooldown_lr_ratio(
    current_step: int,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
) -> float:
    """Return the `1 - progress ** 2` cooldown learning rate multiplier."""
    return 1.0 - math.pow((current_step - num_warmup_steps - num_stable_steps) / num_decay_steps, 2)


def get_linear_cooldown_lr_ratio(
    current_step: int,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
) -> float:
    """Return the linear cooldown learning rate multiplier."""
    return 1.0 - (current_step - num_warmup_steps - num_stable_steps) / num_decay_steps


def get_wsd_scheduler_lambda(  # noqa: PLR0911
    current_step: int,
    *,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
    min_lr_ratio: float,
    num_cycles: float,
    cooldown_type: COOLDOWN_TYPE,
) -> float:
    """Return the warmup-stable-decay learning rate multiplier.

    Args:
        current_step: Current scheduler step, starting at 0.
        num_warmup_steps: Number of warmup steps.
        num_stable_steps: Number of stable steps.
        num_decay_steps: Number of decay steps.
        min_lr_ratio: Learning rate multiplier after decay, also used as the floor for cosine cooldown.
        num_cycles: Number of cosine cycles during decay. Used only for cosine cooldown.
        cooldown_type: Decay curve: `'cosine'`, `'1-sqrt'`, `'linear'`, or `'1-square'`.

    """
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    if current_step < num_warmup_steps + num_stable_steps:
        return 1.0
    if current_step < num_warmup_steps + num_stable_steps + num_decay_steps:
        if cooldown_type == 'cosine':
            return get_cosine_cooldown_lr_ratio(
                current_step, num_warmup_steps, num_stable_steps, num_decay_steps, min_lr_ratio, num_cycles
            )
        if cooldown_type == '1-sqrt':
            return get_1sqrt_cooldown_lr_ratio(current_step, num_warmup_steps, num_stable_steps, num_decay_steps)
        if cooldown_type == '1-square':
            return get_1square_cooldown_lr_ratio(current_step, num_warmup_steps, num_stable_steps, num_decay_steps)
        if cooldown_type == 'linear':
            return get_linear_cooldown_lr_ratio(current_step, num_warmup_steps, num_stable_steps, num_decay_steps)
    return min_lr_ratio


def get_wsd_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    cooldown_type: COOLDOWN_TYPE = '1-sqrt',
    last_epoch: int = -1,
) -> LRScheduler:
    """Create a warmup-stable-decay learning rate scheduler.

    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of warmup steps.
        num_stable_steps: The number of stable steps.
        num_decay_steps: The number of decay steps.
        min_lr_ratio: Learning rate multiplier after decay, also used as the floor for cosine cooldown.
        num_cycles: Number of cosine cycles during decay. Used only for cosine cooldown.
        cooldown_type: Decay curve: `'cosine'`, `'1-sqrt'`, `'linear'`, or `'1-square'`.
        last_epoch: The index of the last epoch when resuming training.

    """
    lr_scheduler = partial(
        get_wsd_scheduler_lambda,
        num_warmup_steps=num_warmup_steps,
        num_stable_steps=num_stable_steps,
        num_decay_steps=num_decay_steps,
        min_lr_ratio=min_lr_ratio,
        num_cycles=num_cycles,
        cooldown_type=cooldown_type,
    )

    return LambdaLR(optimizer, lr_scheduler, last_epoch)
