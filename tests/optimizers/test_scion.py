import pytest
import torch

from pytorch_optimizer.optimizer.scion import build_lmo_norm
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


@pytest.mark.parametrize('optimizer_name', ['scion', 'scionlight'])
def test_parameter_initialization(optimizer_name):
    matrix, bias = make_parameter(), make_parameter((1,))
    build_optimizer(optimizer_name, [matrix, bias]).init()

    torch.testing.assert_close(matrix.abs(), torch.ones_like(matrix))
    torch.testing.assert_close(bias, torch.zeros_like(bias))


@pytest.mark.parametrize(
    ('norm_type', 'options', 'shape', 'initial'),
    [
        (0, {}, (1, 1), 1.0),
        (1, {}, (1,), 0.0),
        (1, {}, (1, 1), 1.0),
        (1, {}, (1, 1, 1, 1), 1.0),
        (2, {'max_scale': True}, (1, 1), 1.0),
        (3, {}, (1, 1, 1, 1), 1.0),
        (4, {'zero_init': True}, (1, 1), 0.0),
        (4, {}, (1, 1), 1.0),
        (5, {}, (1,), 0.0),
        (6, {'normalized': True, 'transpose': True}, (1, 1), 1.0),
        (7, {'normalized': True, 'transpose': True}, (1, 1), 1.0),
    ],
)
def test_lmo_norms(norm_type, options, shape, initial):
    norm = build_lmo_norm(norm_type, **options)
    gradient = torch.ones(shape)

    torch.testing.assert_close(norm.init(gradient.clone()).abs(), torch.full_like(gradient, initial))
    assert torch.isfinite(norm.lmo(gradient)).all()


def test_auto_rejects_unsupported_dimensions():
    norm = build_lmo_norm(1)
    gradient = torch.ones(1, 1, 1, 1, 1)
    for method in (norm.init, norm.lmo):
        with pytest.raises(NotImplementedError):
            method(gradient)
