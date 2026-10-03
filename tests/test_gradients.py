import pytest
import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.optimizer import load_optimizer
from tests.fixtures import make_parameter, make_sparse_parameters
from tests.optimizer_cases import (
    COMPLEX_OPTIMIZERS,
    GRADIENT_OPTIONS,
    SKIP_COMPLEX_NOT_SUPPORTED,
    SKIP_NO_GRADIENT_TEST,
    SKIP_SPARSE_NOT_SUPPORTED,
    SPARSE_OPTIMIZERS,
    VALID_OPTIMIZER_NAMES,
    optimizers_with_argument,
)
from tests.utils import build_optimizer, sphere_loss

NO_SPARSE_OPTIMIZERS = [opt for opt in VALID_OPTIMIZER_NAMES if opt not in SPARSE_OPTIMIZERS]


NO_COMPLEX_OPTIMIZERS = [opt for opt in VALID_OPTIMIZER_NAMES if opt not in COMPLEX_OPTIMIZERS]


class TestGradientAvailability:
    @pytest.mark.parametrize('optimizer_name', [*VALID_OPTIMIZER_NAMES, 'lookahead', 'trac', 'orthograd'])
    def test_no_gradients(self, optimizer_name):
        if optimizer_name in SKIP_NO_GRADIENT_TEST:
            pytest.skip(f'skip {optimizer_name} optimizer.')

        p1 = make_parameter(requires_grad=True)
        p2 = make_parameter(requires_grad=False)
        p3 = make_parameter(requires_grad=True)
        p4 = make_parameter(requires_grad=False)
        params = [{'params': [p1, p2]}, {'params': [p3]}, {'params': [p4]}]

        optimizer = build_optimizer(optimizer_name, params, **GRADIENT_OPTIONS.get(optimizer_name, {}))
        optimizer.zero_grad()

        loss = sphere_loss(p1 + p3)
        p1.grad, p3.grad = torch.autograd.grad(loss, [p1, p3], create_graph=True)

        optimizer.step(lambda: 0.1)
        optimizer.zero_grad(set_to_none=True)


class TestSparseGradients:
    @pytest.mark.parametrize('no_sparse_optimizer', NO_SPARSE_OPTIMIZERS)
    def test_sparse_not_supported(self, no_sparse_optimizer):
        if no_sparse_optimizer in SKIP_SPARSE_NOT_SUPPORTED:
            pytest.skip(f'skip {no_sparse_optimizer} optimizer.')

        param = make_sparse_parameters()[1]

        optimizer = build_optimizer(no_sparse_optimizer, [param])

        with pytest.raises((RuntimeError, NoSparseGradientError)):
            optimizer.step(lambda: 0.1)

    @pytest.mark.parametrize('sparse_optimizer', sorted(SPARSE_OPTIMIZERS))
    def test_sparse(self, sparse_optimizer):
        opt = load_optimizer(optimizer=sparse_optimizer)

        weight, weight_sparse = make_sparse_parameters()

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

    @pytest.mark.parametrize('sparse_optimizer', sorted(SPARSE_OPTIMIZERS))
    def test_sparse_supported(self, sparse_optimizer):
        opt = load_optimizer(optimizer=sparse_optimizer)

        optimizer = opt([make_sparse_parameters()[1]], momentum=0.0)
        optimizer.zero_grad()
        optimizer.step()

        options = {'eps': 0.0} if sparse_optimizer in optimizers_with_argument('eps') else {}
        optimizer = opt([make_sparse_parameters()[1]], momentum=0.0, **options)
        optimizer.step()

        if sparse_optimizer == 'madgrad':
            optimizer = opt([make_sparse_parameters()[1]], momentum=0.0, weight_decay=1e-3, weight_decouple=False)
            with pytest.raises(NoSparseGradientError):
                optimizer.step()

        if sparse_optimizer in ('madgrad', 'dadapt'):
            optimizer = opt([make_sparse_parameters()[1]], momentum=0.9, weight_decay=1e-3)

            if sparse_optimizer == 'madgrad':
                with pytest.raises(NoSparseGradientError):
                    optimizer.step()
            else:
                optimizer.step()


class TestComplexParameters:
    @pytest.mark.parametrize('no_complex_optimizer', NO_COMPLEX_OPTIMIZERS)
    def test_complex_not_supported(self, no_complex_optimizer):
        if no_complex_optimizer in SKIP_COMPLEX_NOT_SUPPORTED:
            pytest.skip(f'skip {no_complex_optimizer}.')

        param = make_parameter(dtype=torch.complex64, grad=1.0)

        use_muon: bool = no_complex_optimizer in ('muon', 'adamuon', 'adago', 'normuon')
        optimizer = build_optimizer(
            no_complex_optimizer, [param], use_muon=use_muon, **GRADIENT_OPTIONS.get(no_complex_optimizer, {})
        )

        with pytest.raises(NoComplexParameterError):
            optimizer.step(lambda: 0.1)
