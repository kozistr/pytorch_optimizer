import numpy as np
import pytest
import torch

from pytorch_optimizer import SAM, AdamP, load_optimizer
from pytorch_optimizer.base.exception import NoSparseGradientError
from tests.constants import NO_SPARSE_OPTIMIZERS, OPTIMIZERS, SPARSE_OPTIMIZERS
from tests.utils import build_environment, ids, tensor_to_numpy


@pytest.mark.parametrize('optimizer_config', OPTIMIZERS, ids=ids)
def test_no_gradients(optimizer_config):
    (x_data, y_data), model, loss_fn = build_environment()

    model.fc1.weight.requires_grad = False
    model.fc1.bias.requires_grad = False

    optimizer_class, config, iterations = optimizer_config
    optimizer = optimizer_class(model.parameters(), **config)

    if optimizer_class.__name__ == 'Nero':
        pytest.skip(f'skip {optimizer_class.__name__}')

    init_loss, loss = np.inf, np.inf
    for _ in range(iterations):
        optimizer.zero_grad()

        y_pred = model(x_data)
        loss = loss_fn(y_pred, y_data)

        if init_loss == np.inf:
            init_loss = loss

        loss.backward()

        optimizer.step()

    assert tensor_to_numpy(init_loss) >= tensor_to_numpy(loss)


@pytest.mark.parametrize('no_sparse_optimizer', NO_SPARSE_OPTIMIZERS)
def test_sparse_not_supported(no_sparse_optimizer):
    param = torch.randn(1, 1).to_sparse(1).requires_grad_(True)
    param.grad = torch.randn(1, 1).to_sparse(1)

    optimizer = load_optimizer(optimizer=no_sparse_optimizer)
    if no_sparse_optimizer == 'ranger21':
        optimizer = optimizer([param], num_iterations=1)
    else:
        optimizer = optimizer([param])

    optimizer.zero_grad()

    with pytest.raises(NoSparseGradientError):
        optimizer.step()


@pytest.mark.parametrize('sparse_optimizer', SPARSE_OPTIMIZERS)
def test_sparse_supported(sparse_optimizer):
    param = torch.randn(1, 1).to_sparse(1).requires_grad_(True)
    param.grad = torch.randn(1, 1).to_sparse(1)

    optimizer = load_optimizer(optimizer=sparse_optimizer)([param], momentum=0.0)
    optimizer.zero_grad()
    optimizer.step()

    optimizer = load_optimizer(optimizer=sparse_optimizer)([param], momentum=0.0, eps=0.0)
    optimizer.zero_grad()
    optimizer.step()

    with pytest.raises(NoSparseGradientError):
        optimizer = load_optimizer(optimizer=sparse_optimizer)([param], momentum=0.0, weight_decay=1e-3)
        optimizer.zero_grad()
        optimizer.step()


def test_sam_no_gradient():
    (x_data, y_data), model, loss_fn = build_environment()
    model.fc1.weight.requires_grad = False
    model.fc1.weight.grad = None

    optimizer = SAM(model.parameters(), AdamP)
    optimizer.zero_grad()

    loss = loss_fn(y_data, model(x_data))
    loss.backward()
    optimizer.first_step(zero_grad=True)

    loss_fn(y_data, model(x_data)).backward()
    optimizer.second_step(zero_grad=True)
