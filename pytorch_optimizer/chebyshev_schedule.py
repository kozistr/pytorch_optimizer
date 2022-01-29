import numpy as np


def chebyshev_steps(small_m: float, big_m: float, num_epochs: int) -> np.ndarray:
    """
    :param small_m: float. stands for 'm' notation.
    :param big_m:  float. stands for 'M' notation.
    :param num_epochs: int. stands for 'T' notation.
    :return: np.array. chebyshev_steps
    """
    c, r = (big_m + small_m) / 2.0, (big_m - small_m) / 2.0
    thetas = (np.arange(num_epochs) + 0.5) / num_epochs * np.pi
    return 1.0 / (c - r * np.cos(thetas))


def chebyshev_perm(num_epochs: int) -> np.ndarray:
    perm = np.array([0])
    while len(perm) < num_epochs:
        perm = np.vstack([perm, 2 * len(perm) - 1 - perm]).T.flatten()
    return perm


def get_chebyshev_schedule(num_epochs: int) -> np.ndarray:
    steps: np.ndarray = chebyshev_steps(0.1, 1, num_epochs - 2)
    perm: np.ndarray = chebyshev_perm(num_epochs - 2)
    chebyshev_schedule = steps[perm]

    return chebyshev_schedule
