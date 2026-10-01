import pytest
import torch
from torch import nn

from pytorch_optimizer import AdamP, Lion, MuAdam, MuAdamW, MuSGD, get_mup_param_groups


class Dim:
    def __init__(self, base_dim, dim):
        self.base_dim = base_dim
        self.dim = dim

    def isinf(self):
        return self.base_dim is not None

    def width_mult(self):
        return self.dim / self.base_dim if self.isinf() else 1


class InfShape(tuple):
    """The three methods of `mup.infshape.InfShape` that the optimizers use."""

    __slots__ = ()

    def ninf(self):
        return sum(d.isinf() for d in self)

    def width_mult(self):
        for d in reversed(self):
            if d.isinf():
                return d.width_mult()
        return 1

    def fanin_fanout_mult_ratio(self):
        return self[1].width_mult() / self[0].width_mult()


def make_param(*dims):
    """Each dim is `(base_dim, dim)` and `base_dim=None` is a finite dimension."""
    p = nn.Parameter(torch.randn(*(d for _, d in dims)))
    p.infshape = InfShape(Dim(b, d) for b, d in dims)
    return p


@pytest.fixture
def params():
    return {
        'matrix': make_param((16, 64), (16, 32)),  # 4 x 2, fan-in 2, fan-out 4
        'wide_matrix': make_param((16, 64), (16, 64)),  # 4 x 4
        'input_layer': make_param((16, 64), (None, 7)),  # width x finite
        'output_layer': make_param((None, 3), (16, 64)),  # finite x width
        'bias': make_param((16, 64)),
        'fixed': make_param((None, 5)),
    }


def by_names(optimizer, params):
    names = {id(p): n for n, p in params.items()}
    return sorted(
        (
            (tuple(sorted(names[id(p)] for p in g['params'])), g['lr'], g['weight_decay'])
            for g in optimizer.param_groups
        )
    )


def test_adam_groups(params):
    optimizer = MuAdam(params.values(), lr=0.1, weight_decay=0.02)

    assert by_names(optimizer, params) == sorted(
        [
            (('matrix',), 0.1 / 2.0, 0.02 * 2.0),
            (('wide_matrix',), 0.1 / 4.0, 0.02 * 4.0),
            (('bias', 'fixed', 'input_layer', 'output_layer'), 0.1, 0.02),
        ]
    )


def test_adam_groups_decoupled_wd(params):
    optimizer = MuAdam(params.values(), lr=0.1, weight_decay=0.02, decoupled_wd=True)

    assert {(g['lr'], g['weight_decay']) for g in optimizer.param_groups} == {(0.05, 0.02), (0.025, 0.02), (0.1, 0.02)}


def test_sgd_groups(params):
    optimizer = MuSGD(params.values(), lr=0.1, weight_decay=0.02)

    # a matrix uses fan-in multiplier / fan-out multiplier (2 / 4 and 4 / 4), a vector-like weight its width multiplier
    assert by_names(optimizer, params) == sorted(
        [
            (('matrix',), 0.1 / 0.5, 0.02 * 0.5),
            (('wide_matrix',), 0.1, 0.02),
            (('bias', 'input_layer', 'output_layer'), 0.1 * 4.0, 0.02 / 4.0),
            (('fixed',), 0.1, 0.02),
        ]
    )


def test_sgd_groups_decoupled_wd(params):
    optimizer = MuSGD(params.values(), lr=0.1, weight_decay=0.02, decoupled_wd=True)

    assert {g['weight_decay'] for g in optimizer.param_groups} == {0.02}
    assert {g['lr'] for g in optimizer.param_groups} == {0.2, 0.1, 0.4}


def test_user_groups_keep_their_options():
    matrix, bias = make_param((16, 64), (16, 64)), make_param((16, 64))
    groups = [{'params': [matrix], 'lr': 0.2, 'betas': (0.5, 0.9)}, {'params': [bias], 'weight_decay': 0.3}]

    optimizer = MuAdamW(groups, lr=0.1, weight_decay=0.01, betas=(0.8, 0.9))

    first, second = optimizer.param_groups
    assert (first['lr'], first['weight_decay'], first['betas']) == (0.2 / 4.0, 0.01 * 4.0, (0.5, 0.9))
    assert (second['lr'], second['weight_decay'], second['betas']) == (0.1, 0.3, (0.8, 0.9))
    assert 'lr' not in groups[0] or groups[0]['lr'] == 0.2  # the caller's dicts are not modified
    assert 'weight_decay' not in groups[0]


def test_empty_groups_are_dropped():
    groups = get_mup_param_groups([make_param((16, 64), (16, 64))], lr=0.1)

    assert len(groups) == 1


def test_matrix_step_is_scaled():
    matrix = make_param((16, 64), (16, 64))
    reference = nn.Parameter(matrix.detach().clone())

    optimizer = MuAdam([matrix], lr=0.4)
    reference_optimizer = torch.optim.Adam([reference], lr=0.1)

    for p in (matrix, reference):
        p.grad = torch.ones_like(p)
    optimizer.step()
    reference_optimizer.step()

    torch.testing.assert_close(matrix, reference)


@pytest.mark.parametrize('impl', [AdamP, Lion])
def test_library_optimizers_as_impl(impl, params):
    optimizer = MuAdam(params.values(), impl=impl, lr=1e-2)

    for p in params.values():
        p.grad = torch.randn_like(p)
    before = {n: p.detach().clone() for n, p in params.items()}
    optimizer.step()

    assert all(not torch.equal(before[n], p) for n, p in params.items())


def test_library_optimizer_as_sgd_like(params):
    optimizer = MuSGD(params.values(), impl=torch.optim.SGD, lr=1e-2, momentum=0.9)

    assert len(optimizer.param_groups) == 4


def test_invalid_inputs(params):
    plain = nn.Parameter(torch.zeros(2, 2))

    with pytest.raises(ValueError, match='infshape'):
        MuAdam([plain], lr=0.1)

    with pytest.raises(NotImplementedError):
        MuAdam([make_param((2, 4), (2, 4), (2, 4))], lr=0.1)

    with pytest.raises(NotImplementedError):
        MuSGD([make_param((2, 4), (2, 4), (2, 4))], lr=0.1)

    with pytest.raises(ValueError, match='lr'):
        get_mup_param_groups(params.values())

    with pytest.raises(ValueError, match='mode'):
        get_mup_param_groups(params.values(), mode='lion', lr=0.1)

    with pytest.raises(ValueError, match='empty'):
        get_mup_param_groups([], lr=0.1)
