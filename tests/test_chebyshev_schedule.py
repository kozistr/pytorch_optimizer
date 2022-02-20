import numpy as np
import pytest

from pytorch_optimizer import get_chebyshev_schedule
from pytorch_optimizer.chebyshev_schedule import chebyshev_perm


def test_get_chebyshev_schedule():
    np.testing.assert_almost_equal(get_chebyshev_schedule(3), 1.81818182, decimal=6)
    np.testing.assert_array_equal(chebyshev_perm(5), np.asarray([0, 7, 3, 4, 1, 6, 2, 5]))

    with pytest.raises(IndexError):
        get_chebyshev_schedule(2)
