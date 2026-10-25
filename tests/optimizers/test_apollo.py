import pytest
import torch

from pytorch_optimizer.optimizer.utils.galore import GaLoreProjector
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


@pytest.mark.parametrize('foreach', [False, True])
@pytest.mark.parametrize('weight_decay_type', ['l2', 'decoupled', 'stable'])
def test_apollodqn_parameter_independence(foreach, weight_decay_type):
    params = [make_parameter((2, 3)) for _ in range(2)]
    optimizer = build_optimizer(
        'apollodqn', params, weight_decay=0.1, weight_decay_type=weight_decay_type, foreach=foreach
    )
    for step in range(3):
        for p in params:
            p.grad = torch.arange(1.0, 7.0).reshape_as(p) / (step + 1)
        optimizer.step()

        torch.testing.assert_close(params[0], params[1])


@pytest.mark.parametrize('foreach', [False, True])
def test_apollo_reduced_rank_and_convolution(foreach):
    matrix, conv = [make_parameter(shape) for shape in ((2, 3), (2, 2, 2))]
    for p in (matrix, conv):
        p.grad = torch.arange(1.0, p.numel() + 1).reshape_as(p) / 10.0

    projector = GaLoreProjector(rank=1, scale=1.0, projection_type='right')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        projected = projector.project(matrix.grad, 1, from_random_matrix=True)
    adaptive = projected / (projected.abs() + 1e-6)
    scaled = projected * (adaptive.norm() / (projected.norm() + 1e-8))
    expected_matrix = -0.1 * projector.project_back(scaled)
    expected_conv = -0.1 * conv.grad / (conv.grad.abs() + 1e-6)

    optimizer = build_optimizer(
        'apollo', [matrix, conv], lr=0.1, betas=(0.0, 0.0), rank=1, scale=1.0,
        update_proj_gap=2, projection_type='right', foreach=foreach,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        optimizer.step()

    torch.testing.assert_close(matrix, expected_matrix)
    torch.testing.assert_close(conv, expected_conv)
