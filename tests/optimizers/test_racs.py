import pytest
import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


@pytest.mark.parametrize('optimizer_name', ['racs', 'alice'])
@pytest.mark.parametrize('weight_decouple', [False, True])
def test_non_linear_parameters(optimizer_name, weight_decouple):
    params = [make_parameter(shape, grad=1.0) for shape in ((4, 8, 1), (4, 8, 2, 2))]
    reference = [make_parameter((p.size(0), p.numel() // p.size(0)), grad=1.0) for p in params]
    options = {'rank': 4, 'leading_basis': 2, 'weight_decay': 0.1, 'weight_decouple': weight_decouple}
    optimizer = build_optimizer(optimizer_name, params, **options)
    expected = build_optimizer(optimizer_name, reference, **options)
    optimizer.step()
    expected.step()

    for p, matrix in zip(params, reference):
        torch.testing.assert_close(p.reshape_as(matrix), matrix)


@pytest.mark.parametrize(
    ('rank', 'diagonals'),
    [
        (2, [[-0.05, -0.025, -0.05], [-0.1, -0.075, -0.075]]),
        (3, [[-0.05, -0.05, -0.05], [-0.1, -0.1, -0.1]]),
    ],
)
def test_alice_switch_preserves_descent(rank, diagonals):
    parameter = make_parameter((3, 3))
    optimizer = build_optimizer(
        'alice', [parameter], lr=0.1, betas=(0.0, 0.0, 0.0), alpha=0.5, alpha_c=0.5,
        rank=rank, leading_basis=1, update_interval=1,
    )

    for diagonal in diagonals:
        parameter.grad = torch.diag(torch.tensor([3.0, 2.0, 1.0]))
        optimizer.step()

        torch.testing.assert_close(parameter, torch.diag(torch.tensor(diagonal)))
