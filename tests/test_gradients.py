import pytest
import torch

from pytorch_optimizer import SAM, AdamP, Lookahead, load_optimizer
from pytorch_optimizer.base.exception import NoSparseGradientError
from tests.constants import NO_SPARSE_OPTIMIZERS, SPARSE_OPTIMIZERS, VALID_OPTIMIZER_NAMES
from tests.utils import build_environment, simple_parameter


@pytest.mark.parametrize('optimizer_name', VALID_OPTIMIZER_NAMES + ['lookahead'])
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


@pytest.mark.parametrize('optimizer_name', VALID_OPTIMIZER_NAMES)
def test_bf16_gradient(optimizer_name):
    # "addcmul_cpu_out" & "eye" not implemented for fp16, bfp16 but gpu op only
    if optimizer_name in ('shampoo', 'adabelief'):
        pytest.skip(optimizer_name)

    param = torch.randn(1, 1).bfloat16().requires_grad_(True)
    param.grad = torch.randn(1, 1).bfloat16()

    optimizer = load_optimizer(optimizer_name)
    if optimizer_name == 'ranger21':
        optimizer = optimizer([param], num_iterations=1)
    else:
        optimizer = optimizer([param])

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
