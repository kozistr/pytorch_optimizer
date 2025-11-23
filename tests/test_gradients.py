from typing import List

import pytest
import torch
from torch import nn

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.optimizer import (
    SAM,
    TRAC,
    WSAM,
    AdamP,
    FriendlySAM,
    Lookahead,
    LookSAM,
    OrthoGrad,
    load_optimizer,
)
from tests.constants import NO_COMPLEX_OPTIMIZERS, NO_SPARSE_OPTIMIZERS, SPARSE_OPTIMIZERS, VALID_OPTIMIZER_NAMES
from tests.utils import (
    build_model,
    build_schedulefree,
    simple_complex_parameter,
    simple_parameter,
    simple_sparse_parameter,
    sphere_loss,
)


class OptimizerBuilder:

    @staticmethod
    def with_muon(params, use_muon: bool):
        def with_flag(group):
            if isinstance(group, dict):
                return group if 'use_muon' in group else {**group, 'use_muon': use_muon}
            return {'params': group, 'use_muon': use_muon}

        return [with_flag(group) for group in params] if isinstance(params, list) else [with_flag(params)]

    @classmethod
    def create(cls, name: str, params: List, **overrides):
        optimizer_name: str = name.lower()

        if optimizer_name == 'lookahead':
            return Lookahead(load_optimizer('adamw')(params), k=1)
        if optimizer_name == 'trac':
            return TRAC(load_optimizer('adamw')(params))
        if optimizer_name == 'orthograd':
            return OrthoGrad(load_optimizer('adamw')(params))

        if optimizer_name == 'ranger21':
            overrides.update({'num_iterations': 1, 'lookahead_merge_time': 1})
        elif optimizer_name == 'bsam':
            overrides.update({'num_data': 1})
        elif optimizer_name in ('lamb', 'ralamb'):
            overrides.update({'pre_norm': True})
        elif optimizer_name == 'alice':
            overrides.update({'rank': 2, 'leading_basis': 1})
        elif optimizer_name == 'adahessian':
            overrides.update({'update_period': 2})

        if optimizer_name in ('muon', 'adamuon', 'adago'):
            params = cls.with_muon(params, use_muon=overrides.pop('use_muon', False))

        return load_optimizer(optimizer_name)(params, **overrides)


@pytest.mark.parametrize('optimizer_name', [*VALID_OPTIMIZER_NAMES, 'lookahead', 'trac', 'orthograd'])
def test_no_gradients(optimizer_name):
    if optimizer_name in {'lbfgs', 'lomo', 'adalomo', 'adammini', 'demo', 'distributedmuon'}:
        pytest.skip(f'skip {optimizer_name} optimizer.')

    p1 = simple_parameter(require_grad=True)
    p2 = simple_parameter(require_grad=False)
    p3 = simple_parameter(require_grad=True)
    p4 = simple_parameter(require_grad=False)
    params = [{'params': [p1, p2]}, {'params': [p3]}, {'params': [p4]}]

    optimizer = OptimizerBuilder.create(optimizer_name, params)
    optimizer.zero_grad()

    loss = sphere_loss(p1 + p3)
    p1.grad, p3.grad = torch.autograd.grad(loss, [p1, p3], create_graph=True)

    optimizer.step(lambda: 0.1)
    optimizer.zero_grad(set_to_none=True)


@pytest.mark.parametrize('no_sparse_optimizer', NO_SPARSE_OPTIMIZERS)
def test_sparse_not_supported(no_sparse_optimizer):
    if no_sparse_optimizer in {'lbfgs', 'sgd', 'lomo', 'adalomo', 'bsam', 'adammini', 'demo', 'distributedmuon'}:
        pytest.skip(f'skip {no_sparse_optimizer} optimizer.')

    param = simple_sparse_parameter()[1]

    optimizer = OptimizerBuilder.create(no_sparse_optimizer, [param])

    with pytest.raises((RuntimeError, NoSparseGradientError)):
        optimizer.step(lambda: 0.1)


@pytest.mark.parametrize('sparse_optimizer', SPARSE_OPTIMIZERS)
def test_sparse(sparse_optimizer):
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

        if sparse_optimizer == 'madgrad':
            with pytest.raises(NoSparseGradientError):
                optimizer.step()
        else:
            optimizer.step()


@pytest.mark.parametrize('optimizer', [SAM, LookSAM, FriendlySAM])
def test_sam_no_gradient(optimizer, environment):
    x_data, y_data = environment
    model, loss_fn = build_model()

    model.fc1.weight.requires_grad = False
    model.fc1.weight.grad = None

    optimizer = optimizer(model.parameters(), AdamP)
    optimizer.zero_grad()

    loss = loss_fn(y_data, model(x_data))
    loss.backward()
    optimizer.first_step(zero_grad=True)

    loss_fn(y_data, model(x_data)).backward()
    optimizer.second_step(zero_grad=True)


def test_wsam_no_gradient(environment):
    x_data, y_data = environment
    model, loss_fn = build_model()

    model.fc1.weight.requires_grad = False
    model.fc1.weight.grad = None

    optimizer = WSAM(model, model.parameters(), AdamP)
    optimizer.zero_grad()

    loss = loss_fn(y_data, model(x_data))
    loss.backward()
    optimizer.first_step(zero_grad=True)

    loss_fn(y_data, model(x_data)).backward()
    optimizer.second_step(zero_grad=True)


@pytest.mark.parametrize('optimizer_name', ['DAdaptAdaGrad', 'DAdaptAdam', 'DAdaptSGD', 'DAdaptAdan', 'Prodigy'])
def test_no_progression(optimizer_name):
    param = simple_parameter(True)
    param.grad = None

    optimizer = load_optimizer(optimizer_name)([param])
    optimizer.zero_grad()
    optimizer.step()


@pytest.mark.parametrize(
    'optimizer_name', ['DAdaptAdaGrad', 'DAdaptAdam', 'DAdaptSGD', 'DAdaptAdan', 'DAdaptLion', 'Prodigy']
)
def test_2nd_stage_gradient(optimizer_name):
    p1 = simple_parameter(require_grad=False)
    p2 = simple_parameter(require_grad=True)
    p3 = simple_parameter(require_grad=True)
    params = [{'params': [p1]}, {'params': [p2]}, {'params': [p3]}]

    optimizer = load_optimizer(optimizer_name)(params)
    optimizer.zero_grad()

    p1.grad = None
    p2.grad = torch.randn(1, 1)
    p3.grad = torch.randn(1, 1)

    optimizer.step()


def test_fromage_zero_norm():
    optimizer = load_optimizer('fromage')([simple_parameter(require_grad=True)])
    optimizer.step()


def test_schedulefree_sparse_gradient():
    param = simple_sparse_parameter()[1]

    optimizer = build_schedulefree([param])
    optimizer.train()

    with pytest.raises(NoSparseGradientError):
        optimizer.step(lambda: 0.1)


@pytest.mark.parametrize('optimizer', ['muon', 'adamuon', 'adago'])
def test_muon_no_gradient(optimizer):
    model = nn.Sequential(nn.Linear(1, 1))
    model[0].weight.grad = None
    model[0].bias.grad = None

    params = [
        {'params': [p for p in model.parameters() if p.ndim >= 2], 'use_muon': True},
        {'params': [p for p in model.parameters() if p.ndim < 2], 'use_muon': False},
    ]

    optimizer = load_optimizer(optimizer)(params)
    optimizer.step()


@pytest.mark.parametrize('no_complex_optimizer', NO_COMPLEX_OPTIMIZERS)
def test_complex_not_supported(no_complex_optimizer):
    if no_complex_optimizer in (
        'adam',
        'adamw',
        'sgd',
        'nadam',
        'lbfgs',
        'rmsprop',
        'lomo',
        'bsam',
        'adammini',
        'adalomo',
        'demo',
        'distributedmuon',
    ):
        pytest.skip(f'skip {no_complex_optimizer}.')

    param = simple_complex_parameter()

    use_muon: bool = no_complex_optimizer in ('muon', 'adamuon', 'adago')
    optimizer = OptimizerBuilder.create(no_complex_optimizer, [param], use_muon=use_muon)

    with pytest.raises(NoComplexParameterError):
        optimizer.step(lambda: 0.1)


def test_orthograd_skip_conditions():
    param = simple_parameter(require_grad=True)
    param.grad = None

    optimizer = OrthoGrad(load_optimizer('adamw')([param]))
    optimizer.apply_orthogonal_gradients([param])
