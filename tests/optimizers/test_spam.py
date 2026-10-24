import pytest
import torch

from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def test_spam_optimizer():
    param = make_parameter(grad=1.0)
    active = make_parameter((1,), grad=1.0)
    optimizer = build_optimizer(
        'spam', [param, active], density=0.0, grad_accu_steps=0, update_proj_gap=1, foreach=True
    )

    optimizer.step()

    torch.testing.assert_close(param, torch.zeros_like(param))
    torch.testing.assert_close(active, torch.zeros_like(active))
    torch.testing.assert_close(active.grad, torch.zeros_like(active))


@pytest.mark.parametrize('foreach', [False, True])
def test_stablespam_zero_gradient_preserves_moments(foreach):
    parameter = make_parameter(grad=1.0)
    optimizer = build_optimizer('stablespam', [parameter], foreach=foreach)
    optimizer.step()

    expected = parameter.detach().clone()
    names = ('exp_avg', 'exp_avg_sq', 'm_norm_t', 'v_norm_t')
    moments = {name: optimizer.state[parameter][name].clone() for name in names}

    parameter.grad.zero_()
    optimizer.step()

    torch.testing.assert_close(parameter, expected)
    torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))
    torch.testing.assert_close({name: optimizer.state[parameter][name] for name in names}, moments)
