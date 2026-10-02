import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.optimizer.sara import SaRA

REFERENCE = Path('/Users/shanyu/Foliation-Engine/_local/hpc_feed_20260930/sara/adamw2_ref.py')


def load_reference():
    spec = importlib.util.spec_from_file_location('sara_adamw2_ref', REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AdamW


def test_sara_matches_reference_and_keeps_large_entries():
    if not REFERENCE.is_file():
        pytest.skip('reference SaRA optimizer is not on this machine')
    reference_cls = load_reference()
    torch.manual_seed(0)
    start = torch.tensor([0.1, -0.2, 3.0, -4.0])
    left = nn.Parameter(start.clone())
    right = nn.Parameter(start.clone())
    config = {
        'lr': 1e-2,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 1e-2,
        'threshold': 0.5,
        'progressive_iter': 100,
        'lambda_rank': 0.0,
    }
    reference = reference_cls([left], **config)
    port = SaRA([right], **config)

    for _ in range(5):
        grad = torch.randn_like(start)
        left.grad = grad.clone()
        right.grad = grad.clone()
        reference.step()
        port.step()

    torch.testing.assert_close(right, left)
    torch.testing.assert_close(right[2:], start[2:])
    assert not torch.equal(right[:2], start[:2])


def test_progressive_mask_is_the_intersection():
    param = nn.Parameter(torch.tensor([0.1, 0.2]))
    optimizer = SaRA([param], lr=0.0, weight_decay=0.0, threshold=0.15, progressive_iter=1)
    assert optimizer.state[param]['mask'].tolist() == [True, False]

    param.grad = torch.ones_like(param)
    optimizer.step()
    with torch.no_grad():
        param.copy_(torch.tensor([0.2, 0.2]))
    param.grad = torch.ones_like(param)
    optimizer.step()

    assert optimizer.state[param]['mask'].tolist() == [False, False]
    torch.testing.assert_close(param, torch.tensor([0.2, 0.2]))


def test_closure_and_parameters_without_gradients():
    left = nn.Parameter(torch.tensor([0.1]))
    right = nn.Parameter(torch.tensor([0.2]))
    optimizer = SaRA([left, right], lr=0.0, weight_decay=0.0, threshold=1.0)
    optimizer.init_group({'params': [right]})

    def closure():
        left.grad = torch.ones_like(left)
        return left.sum()

    assert optimizer.step(closure).item() == pytest.approx(0.1)
    torch.testing.assert_close(right, torch.tensor([0.2]))


def test_init_group_builds_a_mask_for_a_new_parameter():
    held = nn.Parameter(torch.tensor([0.1]))
    optimizer = SaRA([held], threshold=1.0)
    extra = nn.Parameter(torch.tensor([0.2, 3.0]))
    extra.grad = torch.ones_like(extra)
    optimizer.init_group({'params': [extra], 'amsgrad': False, 'threshold': 1.0})
    assert optimizer.state[extra]['mask'].tolist() == [True, False]
    assert optimizer.state[extra]['step'] == 0


def test_negative_progressive_iter_keeps_the_initial_mask():
    param = nn.Parameter(torch.tensor([0.1, 2.0]))
    optimizer = SaRA([param], lr=1e-3, weight_decay=0.0, threshold=0.5)
    before = optimizer.state[param]['mask'].clone()
    param.grad = torch.ones_like(param)
    optimizer.step()
    torch.testing.assert_close(optimizer.state[param]['mask'], before)


def test_rank_penalty_and_amsgrad_stay_finite():
    torch.manual_seed(0)
    matrix = nn.Parameter(torch.randn(65, 65) * 1e-4)
    vector = nn.Parameter(torch.randn(4) * 1e-4)
    optimizer = SaRA([matrix], lr=1e-3, weight_decay=0.0, threshold=1.0, lambda_rank=1e-8, amsgrad=True)
    matrix.grad = torch.randn_like(matrix)
    optimizer.step()
    assert torch.isfinite(matrix).all()

    skipped = SaRA([vector], lr=1e-3, weight_decay=0.0, threshold=1.0, lambda_rank=1.0)
    vector.grad = torch.ones_like(vector)
    skipped.step()
    assert torch.isfinite(vector).all()


def test_maximize_flips_the_update_sign():
    start = torch.tensor([0.1, -0.1])
    plain = nn.Parameter(start.clone())
    flipped = nn.Parameter(start.clone())
    grad = torch.tensor([1.0, -1.0])
    config = {'lr': 1e-2, 'weight_decay': 0.0, 'threshold': 1.0, 'betas': (0.0, 0.0)}
    up = SaRA([plain], maximize=False, **config)
    down = SaRA([flipped], maximize=True, **config)
    plain.grad = grad.clone()
    flipped.grad = grad.clone()
    up.step()
    down.step()
    assert plain[0] < start[0]
    assert flipped[0] > start[0]


def test_rejects_bad_arguments_sparse_and_complex():
    param = nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError):
        SaRA([param], threshold=-1.0)
    with pytest.raises(ValueError):
        SaRA([param], lambda_rank=-0.1)
    with pytest.raises(ValueError):
        SaRA([param], progressive_iter=-2)

    sparse = nn.Parameter(torch.zeros(4))
    sparse.grad = torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([1.0]), (4,))
    with pytest.raises(NoSparseGradientError):
        SaRA([sparse], threshold=1.0).step()

    comp = nn.Parameter(torch.zeros(2, dtype=torch.complex64))
    comp.grad = torch.ones(2, dtype=torch.complex64)
    with pytest.raises(NoComplexParameterError):
        SaRA([comp], threshold=1.0).step()
