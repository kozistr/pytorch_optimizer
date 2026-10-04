import pytest
import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from tests.fixtures import make_parameter, make_sparse_parameters
from tests.optimizer_cases import (
    COMPLEX_OPTIMIZERS,
    GRADIENT_OPTIONS,
    SKIP_CAPABILITY_PROBE,
    SKIP_NO_GRADIENT_TEST,
    SPARSE_OPTIMIZERS,
    VALID_OPTIMIZER_NAMES,
    optimizers_with_argument,
)
from tests.utils import build_optimizer, sphere_loss

NO_SPARSE_OPTIMIZERS = [opt for opt in VALID_OPTIMIZER_NAMES if opt not in SPARSE_OPTIMIZERS | SKIP_CAPABILITY_PROBE]
NO_COMPLEX_OPTIMIZERS = [opt for opt in VALID_OPTIMIZER_NAMES if opt not in COMPLEX_OPTIMIZERS | SKIP_CAPABILITY_PROBE]


class TestGradientAvailability:
    @pytest.mark.parametrize(
        'optimizer_name',
        [
            name
            for name in (*VALID_OPTIMIZER_NAMES, 'lookahead', 'trac', 'orthograd')
            if name not in SKIP_NO_GRADIENT_TEST
        ],
    )
    def test_no_gradients(self, optimizer_name):
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
        param = make_sparse_parameters()[1]

        optimizer = build_optimizer(no_sparse_optimizer, [param])

        with pytest.raises((RuntimeError, NoSparseGradientError)):
            optimizer.step(lambda: 0.1)

    @pytest.mark.parametrize('sparse_optimizer', sorted(SPARSE_OPTIMIZERS))
    def test_sparse(self, sparse_optimizer):
        weight, weight_sparse = make_sparse_parameters()

        params = {'lr': 1e-3, 'momentum': 0.0}
        if sparse_optimizer == 'sm3':
            params.update({'beta': 0.9})

        opt_dense = build_optimizer(sparse_optimizer, [weight], **params)
        opt_sparse = build_optimizer(sparse_optimizer, [weight_sparse], **params)

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

    @pytest.mark.parametrize(
        'sparse_optimizer',
        sorted(SPARSE_OPTIMIZERS & optimizers_with_argument('eps')),
    )
    def test_zero_epsilon(self, sparse_optimizer):
        param = make_sparse_parameters()[1]
        optimizer = build_optimizer(sparse_optimizer, [param], momentum=0.0, eps=0.0)
        optimizer.step()

        assert torch.isfinite(param).all()

    @pytest.mark.parametrize('options', [{'momentum': 0.9}, {'weight_decay': 1e-3, 'weight_decouple': False}])
    def test_madgrad_unsupported_sparse_options(self, options):
        optimizer = build_optimizer('madgrad', [make_sparse_parameters()[1]], **{'momentum': 0.0, **options})

        with pytest.raises(NoSparseGradientError):
            optimizer.step()


class TestComplexParameters:
    @pytest.mark.parametrize('no_complex_optimizer', NO_COMPLEX_OPTIMIZERS)
    def test_complex_not_supported(self, no_complex_optimizer):
        param = make_parameter(dtype=torch.complex64, grad=1.0)

        use_muon: bool = no_complex_optimizer in ('muon', 'adamuon', 'adago', 'normuon')
        optimizer = build_optimizer(
            no_complex_optimizer, [param], use_muon=use_muon, **GRADIENT_OPTIONS.get(no_complex_optimizer, {})
        )

        with pytest.raises(NoComplexParameterError):
            optimizer.step(lambda: 0.1)
