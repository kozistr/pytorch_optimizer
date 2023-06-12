import numpy as np


def get_chebyshev_steps(num_epochs: int, small_m: float = 0.05, big_m: float = 1.0) -> np.ndarray:
    r"""Chebyshev steps.

        gamma_{t} = (M + m) / 2.0 - (M - m) * cos ((t - 0.5) * pi / T) / 2, where t = 1, ..., T

    :param num_epochs: int. stands for 'T' notation.
    :param small_m: float. stands for 'm' notation.
    :param big_m:  float. stands for 'M' notation.
    :return: np.array. chebyshev_steps.
    """
    c, r = (big_m + small_m) / 2.0, (big_m - small_m) / 2.0
    thetas = (np.arange(num_epochs) + 0.5) * np.pi / num_epochs  # epoch starts from 0, so +0.5 instead of -0.5

    return 1.0 / (c - r * np.cos(thetas))


def get_chebyshev_permutation(num_epochs: int) -> np.ndarray:
    r"""Fractal chebyshev permutation.

        sigma_{2T} := interlace(sigma_{T}, 2T + 1 - sigma_{T}), where
        interlace([a_{1}, ..., a_{n}], [b_{1}, ..., b_{n}]) := [a_{1}, b_{1}, ..., n_{1}, b_{n}]

    :param num_epochs: int. number of epochs.
    """
    perm = np.array([0])
    while len(perm) < num_epochs:
        perm = np.vstack([perm, 2 * len(perm) - 1 - perm]).T.flatten()
    return perm


def get_chebyshev_schedule(num_epochs: int) -> np.ndarray:
    r"""Get Chebyshev schedules.

    :param num_epochs: int. number of total epochs.
    """
    steps: np.ndarray = get_chebyshev_steps(num_epochs)
    perm: np.ndarray = get_chebyshev_permutation(num_epochs - 2)
    return steps[perm]


def get_chebyshev_lr(lr: float, epoch: int, num_epochs: int, is_warmup: bool = False) -> float:
    r"""Get chebyshev learning rate.

    :param lr: float. learning rate.
    :param epoch: int. current epochs.
    :param num_epochs: int. number of total epochs.
    :param is_warmup: bool. whether warm-up stage or not.
    """
    if is_warmup:
        return lr

    epoch_power: int = np.power(2, int(np.log2(num_epochs - 1)) + 1) if num_epochs > 1 else 1
    scheduler = get_chebyshev_schedule(epoch_power)

    idx: int = epoch - 2
    if idx < 0:
        idx = 0
    elif idx > len(scheduler) - 1:
        idx = len(scheduler) - 1

    chebyshev_value: float = scheduler[idx]

    return lr * chebyshev_value
