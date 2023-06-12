import numpy as np


def get_chebyshev_steps(num_epochs: int, small_m: float = 0.05, big_m: float = 1.0) -> np.ndarray:
    r"""Chebyshev steps.

    :param num_epochs: int. stands for 'T' notation.
    :param small_m: float. stands for 'm' notation.
    :param big_m:  float. stands for 'M' notation.
    :return: np.array. chebyshev_steps.
    """
    c, r = (big_m + small_m) / 2.0, (big_m - small_m) / 2.0
    thetas = (np.arange(num_epochs) + 0.5) / num_epochs * np.pi

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

    :param num_epochs: int. number of epochs.
    """
    if num_epochs < 1:
        raise ValueError(f'[-] num_epochs must be over 1. {num_epochs} > 1')
    elif num_epochs == 1:
        return get_chebyshev_steps(1)

    steps: np.ndarray = get_chebyshev_steps(num_epochs - 2)
    perm: np.ndarray = get_chebyshev_permutation(num_epochs - 2)

    return steps[perm]
