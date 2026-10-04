import pytest
import torch

from pytorch_optimizer.optimizer.sm3 import reduce_max_except_dim
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


class TestSM3Utils:
    def test_max_reduce_except_dim(self):
        x = torch.tensor(1.0)
        assert reduce_max_except_dim(x, 0) == x

        x = torch.zeros((1, 1))
        with pytest.raises(ValueError):
            reduce_max_except_dim(x, 3)


class TestSm3:
    def test_sm3_make_sparse(self):
        param = make_parameter((5, 3), grad=None)
        optimizer = build_optimizer('sm3', [param], lr=0.1)
        indices = torch.tensor([[0, 2]])
        param.grad = torch.sparse_coo_tensor(indices, torch.ones(2, 3), param.shape)
        sparse = optimizer.make_sparse(param.grad, torch.tensor(1.0))
        assert sparse.shape == param.shape
        assert sparse._nnz() == 0

        for _ in range(2):
            param.grad = torch.sparse_coo_tensor(indices, torch.ones(2, 3), param.shape)
            optimizer.step()

        expected = torch.zeros_like(param)
        expected[[0, 2]] = -0.1 * (1.0 + 2.0**-0.5)
        torch.testing.assert_close(param, expected)

    def test_sm3_rank0(self):
        optimizer = build_optimizer('sm3', [make_parameter(())])
        optimizer.step()

        assert str(optimizer) == 'SM3'
