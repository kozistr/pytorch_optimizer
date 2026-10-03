import pytest
import torch

from pytorch_optimizer.optimizer.sso import solve_lambda_with_bisection
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def test_spectral_sphere_methods():
    opt = build_optimizer('spectralsphere', [make_parameter(())])
    with pytest.raises(ValueError):
        opt.step()

    x = torch.full((2, 2), 10.0)
    theta = torch.full((2, 2), 0.01)
    _ = solve_lambda_with_bisection(x, theta)

    x = torch.tensor([[5.0, 1.0], [1.0, 5.0]])
    theta = torch.tensor([[1.5, 0.0], [0.0, -2.8]])
    _ = solve_lambda_with_bisection(x, theta, initial_guess=0.18, initial_step=0.012, msign_steps=0)
