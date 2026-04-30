import pytest
import torch

from pytorch_optimizer.base.exception import NegativeLRError, NoSparseGradientError
from pytorch_optimizer.optimizer import DualAdam
from tests.utils import simple_parameter, simple_sparse_parameter


def _get_expected_dual_adam_update(
    param,
    grad,
    exp_avg,
    exp_avg_sq,
    *,
    lr,
    betas,
    switch_rate,
    step,
    weight_decay=0.0,
    weight_decouple=False,
    fixed_decay=False,
    eps=1e-8,
    maximize=False,
):
    beta1, beta2 = betas
    param = param.clone()
    grad = grad.clone()

    if maximize:
        grad.neg_()

    if weight_decouple:
        param.mul_(1.0 - weight_decay * (1.0 if fixed_decay else lr))
    elif weight_decay > 0.0:
        grad.add_(param, alpha=weight_decay)

    exp_avg = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1)
    exp_avg_sq = exp_avg_sq.mul(beta2).addcmul(grad, grad, value=1.0 - beta2)

    exp_avg_hat = exp_avg / (1.0 - beta1**step)
    de_nom = (exp_avg_sq / (1.0 - beta2**step)).sqrt().add(eps)

    inverse_adam_rate = max(0.0, 1.0 - step * switch_rate)
    if inverse_adam_rate >= switch_rate:
        update = (1.0 - inverse_adam_rate) * exp_avg_hat.div(de_nom) + inverse_adam_rate * exp_avg_hat.mul(de_nom)
        param.add_(update, alpha=-lr)
    else:
        param.addcdiv_(exp_avg_hat, de_nom, value=-lr)

    return param, exp_avg, exp_avg_sq


def test_dual_adam_inverse_blend_update():
    param = torch.tensor([1.0, -2.0], requires_grad=True)
    grad = torch.tensor([0.2, -0.4])
    param.grad = grad.clone()

    optimizer = DualAdam([param], lr=0.1, betas=(0.5, 0.25), switch_rate=0.25, weight_decay=0.1)

    expected_param, expected_exp_avg, expected_exp_avg_sq = _get_expected_dual_adam_update(
        param.detach(),
        grad,
        torch.zeros_like(param),
        torch.zeros_like(param),
        lr=0.1,
        betas=(0.5, 0.25),
        switch_rate=0.25,
        step=1,
        weight_decay=0.1,
    )

    optimizer.step()

    assert str(optimizer) == 'DualAdam'
    assert optimizer.param_groups[0]['step'] == 1
    assert torch.allclose(param, expected_param)
    assert torch.allclose(optimizer.state[param]['exp_avg'], expected_exp_avg)
    assert torch.allclose(optimizer.state[param]['exp_avg_sq'], expected_exp_avg_sq)


def test_dual_adam_adam_update_with_decoupled_decay_and_maximize():
    param = torch.tensor([1.5, -0.5], requires_grad=True)
    grad = torch.tensor([0.3, -0.2])
    param.grad = grad.clone()

    optimizer = DualAdam(
        [param],
        lr=0.05,
        betas=(0.5, 0.5),
        switch_rate=1.0,
        weight_decay=0.2,
        weight_decouple=True,
        fixed_decay=True,
        eps=1e-6,
        maximize=True,
    )

    expected_param, expected_exp_avg, expected_exp_avg_sq = _get_expected_dual_adam_update(
        param.detach(),
        grad,
        torch.zeros_like(param),
        torch.zeros_like(param),
        lr=0.05,
        betas=(0.5, 0.5),
        switch_rate=1.0,
        step=1,
        weight_decay=0.2,
        weight_decouple=True,
        fixed_decay=True,
        eps=1e-6,
        maximize=True,
    )

    optimizer.step()

    assert torch.allclose(param, expected_param)
    assert torch.allclose(optimizer.state[param]['exp_avg'], expected_exp_avg)
    assert torch.allclose(optimizer.state[param]['exp_avg_sq'], expected_exp_avg_sq)


def test_dual_adam_closure_without_gradient():
    param = simple_parameter(require_grad=True)
    param.grad = None

    optimizer = DualAdam([param])

    assert optimizer.step(lambda: 1.0) == 1.0
    assert optimizer.param_groups[0]['step'] == 1
    assert len(optimizer.state[param]) == 0


def test_dual_adam_sparse_gradient():
    param = simple_sparse_parameter()[1]

    optimizer = DualAdam([param])

    with pytest.raises(NoSparseGradientError):
        optimizer.step()


@pytest.mark.parametrize(
    ('kwargs', 'error'),
    [
        ({'lr': -1e-3}, NegativeLRError),
        ({'betas': (-0.1, 0.999)}, ValueError),
        ({'betas': (0.9, 1.0)}, ValueError),
        ({'switch_rate': -1e-2}, ValueError),
        ({'switch_rate': 1.1}, ValueError),
        ({'weight_decay': -1e-3}, ValueError),
        ({'eps': -1e-8}, ValueError),
    ],
)
def test_dual_adam_invalid_parameters(kwargs, error):
    with pytest.raises(error):
        DualAdam(None, **kwargs)
