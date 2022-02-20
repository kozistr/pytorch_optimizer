from typing import List

import pytest
import torch

from pytorch_optimizer import load_optimizers

SPARSE_OPTIMIZERS: List[str] = [
    'madgrad',
]

NO_SPARSE_OPTIMIZERS: List[str] = [
    'adamp',
    'sgdp',
    'madgrad',
    'ranger',
    'ranger21',
    'radam',
    'adabound',
    'adahessian',
    'adabelief',
    'diffgrad',
    'diffrgrad',
    'lamb',
    'ralamb',
    'lars',
]


@pytest.mark.parametrize('no_sparse_optimizer', NO_SPARSE_OPTIMIZERS)
def test_sparse_not_supported(no_sparse_optimizer):
    param = torch.randn(1, 1).to_sparse(1).requires_grad_(True)
    param.grad = torch.randn(1, 1).to_sparse(1)

    optimizer = load_optimizers(optimizer=no_sparse_optimizer)
    if no_sparse_optimizer == 'ranger21':
        optimizer = optimizer([param], num_iterations=1)
    else:
        optimizer = optimizer([param])

    optimizer.zero_grad()

    with pytest.raises(RuntimeError):
        optimizer.step()


@pytest.mark.parametrize('sparse_optimizer', SPARSE_OPTIMIZERS)
def test_sparse_supported(sparse_optimizer):
    param = torch.randn(1, 1).to_sparse(1).requires_grad_(True)
    param.grad = torch.randn(1, 1).to_sparse(1)

    optimizer = load_optimizers(optimizer=sparse_optimizer)([param], momentum=0.0)
    optimizer.zero_grad()
    optimizer.step()

    optimizer = load_optimizers(optimizer=sparse_optimizer)([param], momentum=0.0, eps=0.0)
    optimizer.zero_grad()
    optimizer.step()

    with pytest.raises(RuntimeError):
        optimizer = load_optimizers(optimizer=sparse_optimizer)([param], momentum=0.0, weight_decay=1e-3)
        optimizer.zero_grad()
        optimizer.step()
