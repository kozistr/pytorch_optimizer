import numpy as np
import pytest

from pytorch_optimizer import get_chebyshev_schedule


def test_get_chebyshev_schedule():
    np.testing.assert_almost_equal(get_chebyshev_schedule(3), 1.81818182, decimal=6)

    with pytest.raises(IndexError):
        get_chebyshev_schedule(2)
