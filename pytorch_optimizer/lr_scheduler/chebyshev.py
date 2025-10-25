from functools import partial

import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


def get_chebyshev_steps(num_epochs: int, small_m: float = 0.05, big_m: float = 1.0) -> np.ndarray:
    r"""Chebyshev steps.

    Computes Chebyshev step sizes according to the formula:
        gamma_{t} = (M + m) / 2.0 - (M - m) * cos((t - 0.5) * pi / T) / 2, where t = 1, ..., T

    Args:
        num_epochs (int): Total number of steps T.
        small_m (float): The lower bound m.
        big_m (float): The upper bound M.

    Returns:
        np.array: Array of Chebyshev step sizes of length num_epochs.
    """
    c, r = (big_m + small_m) / 2.0, (big_m - small_m) / 2.0
    thetas = (np.arange(num_epochs) + 0.5) * np.pi / num_epochs  # epoch starts from 0, so +0.5 instead of -0.5

    return 1.0 / (c - r * np.cos(thetas))


def get_chebyshev_permutation(num_epochs: int) -> np.ndarray:
    r"""Fractal Chebyshev permutation.

    This permutation is defined recursively as:
        sigma_{2T} := interlace(sigma_{T}, 2T + 1 - sigma_{T}),
    where the interlace function is defined as:
        interlace([a_1, ..., a_n], [b_1, ..., b_n]) := [a_1, b_1, a_2, b_2, ..., a_n, b_n]

    Args:
        num_epochs (int): Number of epochs (T).
    """
    perm = np.array([0])
    while len(perm) < num_epochs:
        perm = np.vstack([perm, 2 * len(perm) - 1 - perm]).T.flatten()
    return perm


def get_chebyshev_perm_steps(num_epochs: int) -> np.ndarray:
    r"""Get Chebyshev schedules.

    Args:
        num_epochs (int): Number of total epochs.
    """
    steps: np.ndarray = get_chebyshev_steps(num_epochs)
    perm: np.ndarray = get_chebyshev_permutation(num_epochs - 2)
    return steps[perm]


def get_chebyshev_lr_lambda(epoch: int, num_epochs: int, is_warmup: bool = False) -> float:
    """Get Chebyshev learning rate ratio.

    Args:
        epoch (int): Current epoch.
        num_epochs (int): Total number of epochs.
        is_warmup (bool): Whether it is the warm-up stage.

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
    """Get Chebyshev learning rate scheduler.

    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        num_epochs (int): Number of total epochs.
        is_warmup (bool): Whether it is the warm-up stage.
        last_epoch (int): The index of the last epoch when resuming training.
    """
    lr_scheduler = partial(get_chebyshev_lr_lambda, num_epochs=num_epochs, is_warmup=is_warmup)

    return LambdaLR(optimizer, lr_scheduler, last_epoch)
