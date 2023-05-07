import pytest
import torch

from pytorch_optimizer import SAM, AdamP, Lookahead, load_optimizer
from pytorch_optimizer.base.exception import NoSparseGradientError
from tests.constants import (
    ADANORM_SUPPORTED_OPTIMIZERS,
    NO_SPARSE_OPTIMIZERS,
    SPARSE_OPTIMIZERS,
    VALID_OPTIMIZER_NAMES,
)
from tests.utils import build_environment, ids, simple_parameter, simple_sparse_parameter


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
    optimizer.step(lambda: 0.1)  # for AliG optimizer


@pytest.mark.parametrize('no_sparse_optimizer', NO_SPARSE_OPTIMIZERS)
def test_sparse_not_supported(no_sparse_optimizer):
    param = simple_sparse_parameter()[1]

    opt = load_optimizer(optimizer=no_sparse_optimizer)
    optimizer = opt([param], num_iterations=1) if no_sparse_optimizer == 'ranger21' else opt([param])

    with pytest.raises(NoSparseGradientError):
        optimizer.step(lambda: 0.1)


@pytest.mark.parametrize('sparse_optimizer', SPARSE_OPTIMIZERS)
def test_sparse_gradient(sparse_optimizer):
    opt = load_optimizer(optimizer=sparse_optimizer)

    weight, weight_sparse = simple_sparse_parameter()

    params = {'lr': 1e-3, 'momentum': 0.0}
    if sparse_optimizer == 'sm3':
        params.update({'beta': 0.9})

    opt_dense = opt([weight], **params)
    opt_sparse = opt([weight_sparse], **params)

    opt_dense.step()
    opt_sparse.step()
    assert torch.allclose(weight, weight_sparse)

    weight.grad = torch.rand_like(weight)
    weight.grad[1] = 0.0
    weight_sparse.grad = weight.grad.to_sparse()

    opt_dense.step()
    opt_sparse.step()
    assert torch.allclose(weight, weight_sparse)

    weight.grad = torch.rand_like(weight)
    weight.grad[0] = 0.0
    weight_sparse.grad = weight.grad.to_sparse()

    opt_dense.step()
    opt_sparse.step()
    assert torch.allclose(weight, weight_sparse)


@pytest.mark.parametrize('sparse_optimizer', SPARSE_OPTIMIZERS)
def test_sparse_supported(sparse_optimizer):
    opt = load_optimizer(optimizer=sparse_optimizer)

    optimizer = opt([simple_sparse_parameter()[1]], momentum=0.0)
    optimizer.zero_grad()
    optimizer.step()

    optimizer = opt([simple_sparse_parameter()[1]], momentum=0.0, eps=0.0)
    optimizer.step()

    if sparse_optimizer == 'madgrad':
        optimizer = opt([simple_sparse_parameter()[1]], momentum=0.0, weight_decay=1e-3, weight_decouple=False)
        with pytest.raises(NoSparseGradientError):
            optimizer.step()

    if sparse_optimizer in ('madgrad', 'dadapt'):
        optimizer = opt([simple_sparse_parameter()[1]], momentum=0.9, weight_decay=1e-3)
        optimizer.reset()
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
    optimizer.step(lambda: 0.1)


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


@pytest.mark.parametrize('optimizer_name', ['DAdaptAdaGrad', 'DAdaptAdam', 'DAdaptSGD', 'DAdaptAdan'])
def test_d_adapt_no_progress(optimizer_name):
    param = simple_parameter(True)
    param.grad = None

    optimizer = load_optimizer(optimizer_name)([param])
    optimizer.zero_grad()
    optimizer.step()


@pytest.mark.parametrize('optimizer_name', ['DAdaptAdaGrad', 'DAdaptAdam', 'DAdaptSGD', 'DAdaptAdan'])
def test_d_adapt_2nd_stage_gradient(optimizer_name):
    p1 = simple_parameter(require_grad=False)
    p2 = simple_parameter(require_grad=True)
    p3 = simple_parameter(require_grad=True)
    params = [{'params': [p1]}] + [{'params': [p2]}] + [{'params': [p3]}]

    optimizer = load_optimizer(optimizer_name)(params)
    optimizer.zero_grad()

    p1.grad = None
    p2.grad = torch.randn(1, 1)
    p3.grad = torch.randn(1, 1)

    optimizer.step()


def test_fromage_zero_norm():
    param = simple_parameter(require_grad=True)

    optimizer = load_optimizer('fromage')([param])
    optimizer.step()


@pytest.mark.parametrize('optimizer_config', ADANORM_SUPPORTED_OPTIMIZERS, ids=ids)
def test_ada_norm_gradient(optimizer_config):
    param = simple_parameter(True)
    param.grad = torch.ones(1, 1)

    optimizer_class, config = optimizer_config[:2]

    optimizer = optimizer_class([param], adanorm=True)
    optimizer.step()

    param.grad = torch.zeros(1, 1)
    optimizer.step()


def test_nero_zero_scale():
    param = simple_parameter()

    optimizer = load_optimizer('nero')([param], constraints=False)
    optimizer.zero_grad()

    param.grad = torch.zeros(1, 1)
    optimizer.step()


@pytest.mark.parametrize('optimizer_name', ['adabelief', 'radam', 'lamb', 'diffgrad', 'ranger'])
def test_rectified_optimizer_gradient(optimizer_name):
    param = simple_parameter()

    parameters = {'n_sma_threshold': 1000, 'degenerated_to_sgd': False}
    if optimizer_name not in ('adabelief', 'radam', 'ranger'):
        parameters.update({'rectify': True})

    optimizer = load_optimizer(optimizer_name)([param], **parameters)
    optimizer.zero_grad()

    param.grad = torch.zeros(1, 1)
    optimizer.step()
