"""The AdaNorm norm average must accumulate, and zero grads must stay finite."""
import torch

from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.optimizer.adanorm import AdaNorm


def test_ema_accumulates_and_scales_small_gradients():
    exp = torch.tensor([1.0])
    out = BaseOptimizer.get_adanorm_gradient(
        grad=torch.ones(4), adanorm=True, exp_grad_norm=exp, r=0.95
    )
    # grad norm is 2, so exp moves to 1 * 0.95 + 2 * 0.05 = 1.05, below 2:
    # no scaling.
    assert abs(exp.item() - 1.05) < 1e-6
    assert torch.equal(out, torch.ones(4))

    small = torch.full((4,), 0.1)
    out = BaseOptimizer.get_adanorm_gradient(
        grad=small, adanorm=True, exp_grad_norm=exp, r=0.95
    )
    # exp moves to 1.05 * 0.95 + 0.2 * 0.05 = 1.0075, above 0.2: scaled up.
    assert abs(exp.item() - 1.0075) < 1e-6
    assert torch.allclose(out, small * (1.0075 / 0.2))


def test_zero_gradient_after_history_stays_finite():
    exp = torch.tensor([1.0])
    out = BaseOptimizer.get_adanorm_gradient(
        grad=torch.zeros(4), adanorm=True, exp_grad_norm=exp, r=0.95
    )
    assert torch.equal(out, torch.zeros(4))
    assert torch.isfinite(exp).all()


def test_optimizer_runs_zero_grad_after_history():
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    opt = AdaNorm([param], lr=1e-3)
    param.grad = torch.tensor([0.5, -0.5, 0.25, -0.25])
    opt.step()
    assert opt.state[param]['exp_grad_norm'].item() > 0
    param.grad = torch.zeros(4)
    opt.step()
    assert torch.isfinite(param.detach()).all()
    assert torch.isfinite(opt.state[param]['exp_avg']).all()
