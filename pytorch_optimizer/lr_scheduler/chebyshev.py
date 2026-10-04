from functools import partial

import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


def get_chebyshev_steps(num_epochs: int, small_m: float = 0.05, big_m: float = 1.0) -> np.ndarray:
    """Compute reciprocal Chebyshev nodes as learning rate multipliers.

    Args:
        num_epochs: Number of Chebyshev nodes.
        small_m: Lower bound of the spectral interval.
        big_m: Upper bound of the spectral interval.

    Returns:
        np.ndarray: Reciprocal nodes, with shape `(num_epochs,)`.

    """
    c, r = (big_m + small_m) / 2.0, (big_m - small_m) / 2.0
    thetas = (np.arange(num_epochs) + 0.5) * np.pi / num_epochs  # epoch starts from 0, so +0.5 instead of -0.5

    return 1.0 / (c - r * np.cos(thetas))


def get_chebyshev_permutation(num_epochs: int) -> np.ndarray:
    """Construct a fractal permutation of Chebyshev node indices.

    Args:
        num_epochs: Requested number of indices.

    Returns:
        np.ndarray: Zero based indices, with length rounded up to a power of two.

    """
    perm = np.array([0])
    while len(perm) < num_epochs:
        perm = np.vstack([perm, 2 * len(perm) - 1 - perm]).T.flatten()
    return perm


def get_chebyshev_perm_steps(num_epochs: int) -> np.ndarray:
    """Return Chebyshev learning rate multipliers in fractal order.

    Args:
        num_epochs: Number of total epochs.

    """
    if num_epochs < 1:
        raise IndexError('num_epochs must be positive')

    steps: np.ndarray = get_chebyshev_steps(num_epochs)
    perm: np.ndarray = get_chebyshev_permutation(num_epochs - 2)
    return steps[perm[perm < num_epochs]]


def get_chebyshev_lr_lambda(epoch: int, num_epochs: int, is_warmup: bool = False) -> float:
    """Return the Chebyshev learning rate multiplier for an epoch.

    Args:
        epoch: Current epoch.
        num_epochs: Total number of epochs.
        is_warmup: Whether it is the warmup stage.

    Returns:
        float: Learning rate ratio for the given epoch based on Chebyshev schedule.

    """
    if is_warmup:
        return 1.0

    epoch_power: int = np.power(2, int(np.log2(num_epochs - 1)) + 1) if num_epochs > 1 else 1
    scheduler = get_chebyshev_perm_steps(epoch_power)

    idx: int = epoch - 2
    if idx < 0:
        idx = 0
    elif idx > len(scheduler) - 1:
        idx = len(scheduler) - 1

    chebyshev_value: float = scheduler[idx]

    return chebyshev_value


def get_chebyshev_schedule(
    optimizer: Optimizer, num_epochs: int, is_warmup: bool = False, last_epoch: int = -1
) -> LRScheduler:
    """Create a fractal Chebyshev learning rate scheduler.

    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_epochs: Number of total epochs.
        is_warmup: Whether it is the warmup stage.
        last_epoch: The index of the last epoch when resuming training.

    """
    lr_scheduler = partial(get_chebyshev_lr_lambda, num_epochs=num_epochs, is_warmup=is_warmup)

    return LambdaLR(optimizer, lr_scheduler, last_epoch)
