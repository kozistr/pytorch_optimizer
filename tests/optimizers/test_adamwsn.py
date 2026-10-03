import pytest

from pytorch_optimizer.optimizer.snsm import closest_smaller_divisor_of_n_to_k


class TestAdamWSNUtils:
    def test_csd(self):
        assert closest_smaller_divisor_of_n_to_k(2, 2) == 2
        assert closest_smaller_divisor_of_n_to_k(5, 3) == 1

        with pytest.raises(ValueError):
            closest_smaller_divisor_of_n_to_k(1, 2)
