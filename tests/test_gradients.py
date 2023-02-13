import pytest
import torch

from pytorch_optimizer import SAM, AdamP, Lookahead, load_optimizer
from pytorch_optimizer.base.exception import NoSparseGradientError
from tests.constants import NO_SPARSE_OPTIMIZERS, SPARSE_OPTIMIZERS, VALID_OPTIMIZER_NAMES
from tests.utils import build_environment, simple_parameter, simple_sparse_parameter


@pytest.mark.parametrize('optimizer_name', [*VALID_OPTIMIZER_NAMES, 'lookahead'])
def test_no_gradients(optimizer_name):
    p1 = simple_parameter(require_grad=True)
    p2 = simple_parameter(require_grad=False)
    p3 = simple_parameter(require_grad=True)
    p4 = simple_parameter(require_grad=False)
    params = [{'params': [p1, p2]}] + [{'params': [p3]}] + [{'params': [p4]}]

    if optimizer_name == 'ranger21':
        optimizer = load_optimizer(optimizer_name)(params, num_iterations=1, lookahead_merge_time=1)
    elif optimizer_name in ('lamb', 'ralamb'):
        optimizer = load_optimizer(optimizer_name)(params, pre_norm=True)
    elif optimizer_name == 'lookahead':
        optimizer = Lookahead(load_optimizer('adamp')(params), k=1)
    else:
        optimizer = load_optimizer(optimizer_name)(params)

    optimizer.zero_grad()
    p1.grad = torch.zeros(1, 1)
    p2.grad = None
    p3.grad = torch.zeros(1, 1)
    p4.grad = None
    optimizer.step()


@pytest.mark.parametrize('no_sparse_optimizer', NO_SPARSE_OPTIMIZERS)
def test_sparse_not_supported(no_sparse_optimizer):
    param = simple_sparse_parameter()

    opt = load_optimizer(optimizer=no_sparse_optimizer)
    optimizer = opt([param], num_iterations=1) if no_sparse_optimizer == 'ranger21' else opt([param])

    optimizer.zero_grad()

    with pytest.raises(NoSparseGradientError):
        optimizer.step()


@pytest.mark.parametrize('sparse_optimizer', SPARSE_OPTIMIZERS)
def test_sparse_supported(sparse_optimizer):
    param = simple_sparse_parameter()
    opt = load_optimizer(optimizer=sparse_optimizer)

    optimizer = opt([param], momentum=0.0)
    optimizer.zero_grad()
    optimizer.step()

    optimizer = opt([param], momentum=0.0, eps=0.0)
    optimizer.reset()
    optimizer.zero_grad()
    optimizer.step()

    optimizer = opt([param], momentum=0.9, weight_decay=1e-3)
    optimizer.reset()
    optimizer.zero_grad()

    if sparse_optimizer == 'madgrad':
        optimizer = opt([param], weight_decay=1e-3, decouple_decay=True)
        optimizer.reset()
        optimizer.zero_grad()

        with pytest.raises(NoSparseGradientError):
            optimizer.step()

    if sparse_optimizer == 'madgrad':
        with pytest.raises(NoSparseGradientError):
            optimizer.step()
    else:
        optimizer.step()


@pytest.mark.parametrize('optimizer_name', VALID_OPTIMIZER_NAMES)
def test_bf16_gradient(optimizer_name):
    # torch.eye does not support bf16
    if optimizer_name == 'shampoo':
        pytest.skip(f'skip {optimizer_name}')

    param = torch.randn(1, 1).bfloat16().requires_grad_(True)
    param.grad = torch.randn(1, 1).bfloat16()

    opt = load_optimizer(optimizer=optimizer_name)
    optimizer = opt([param], num_iterations=1) if optimizer_name == 'ranger21' else opt([param])

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
