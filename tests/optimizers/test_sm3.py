import pytest
import torch

from pytorch_optimizer.optimizer import load_optimizer
from pytorch_optimizer.optimizer.sm3 import reduce_max_except_dim
from tests.fixtures import make_parameter, make_sparse_parameters


class TestSM3Utils:
    def test_max_reduce_except_dim(self):
        x = torch.tensor(1.0)
        assert reduce_max_except_dim(x, 0) == x

        x = torch.zeros((1, 1))
        with pytest.raises(ValueError):
            reduce_max_except_dim(x, 3)


class TestSm3:
    def test_sm3_make_sparse(self):
        _, weight_sparse = make_sparse_parameters()

        optimizer = load_optimizer('sm3')([weight_sparse])

        values = torch.tensor(1.0)
        optimizer.make_sparse(weight_sparse.grad, values)

    def test_sm3_rank0(self):
        optimizer = load_optimizer('sm3')([make_parameter(())])
        optimizer.step()

        assert str(optimizer) == 'SM3'
