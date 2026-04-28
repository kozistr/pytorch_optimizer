import pytest
import torch

from pytorch_optimizer.base.exception import NegativeLRError, NoSparseGradientError
from pytorch_optimizer.optimizer import FlashAdamW, load_optimizer
from tests.utils import simple_parameter, simple_sparse_parameter


def _run_matching_steps(flash_optimizer, torch_optimizer, flash_param, torch_param, gradients):
    for grad in gradients:
        flash_param.grad = grad.clone()
        torch_param.grad = grad.clone()
        flash_optimizer.step()
        torch_optimizer.step()
        flash_optimizer.zero_grad()
        torch_optimizer.zero_grad()


def test_flash_adamw_matches_torch_adamw():
    flash_param = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
    torch_param = torch.nn.Parameter(flash_param.detach().clone())
    gradients = [torch.tensor([0.2, -0.3, 0.4]), torch.tensor([-0.1, 0.5, -0.2])]

    flash_optimizer = FlashAdamW([flash_param], lr=1e-2, betas=(0.5, 0.9), eps=1e-6, weight_decay=0.1)
    torch_optimizer = torch.optim.AdamW([torch_param], lr=1e-2, betas=(0.5, 0.9), eps=1e-6, weight_decay=0.1)

    _run_matching_steps(flash_optimizer, torch_optimizer, flash_param, torch_param, gradients)

    assert str(flash_optimizer) == 'FlashAdamW'
    assert torch.allclose(flash_param, torch_param)
    assert flash_optimizer.param_groups[0]['step'] == 2


def test_flash_adamw_decouple_lr_weight_decay():
    param = torch.nn.Parameter(torch.tensor([2.0]))
    param.grad = torch.zeros_like(param)

    optimizer = FlashAdamW([param], lr=1e-1, betas=(0.0, 0.0), weight_decay=0.2, decouple_lr=True)
    optimizer.param_groups[0]['lr'] = 5e-2
    optimizer.step()

    expected = torch.tensor([2.0 * (1.0 - 0.2 * 0.5)])
    assert torch.allclose(param, expected)
    assert FlashAdamW.get_weight_decay_factor(1e-1, 0.0, 0.1, decouple_lr=True) == 0.0


def test_flash_adamw_maximize_and_closure_without_gradient():
    param = simple_parameter(require_grad=True)
    param.grad = torch.ones_like(param)

    optimizer = FlashAdamW([param], lr=1e-1, betas=(0.0, 0.0), weight_decay=0.0, maximize=True)
    optimizer.step()

    assert torch.allclose(param, torch.tensor([[0.1]]))

    param.grad = None
    assert optimizer.step(lambda: 1.0) == 1.0


def test_flash_adamw_param_group_initial_lr():
    param = simple_parameter(require_grad=True)
    param.grad = torch.zeros_like(param)
    optimizer = FlashAdamW([{'params': [param], 'lr': 0.2}], lr=0.1, weight_decay=0.0)

    assert optimizer.param_groups[0]['initial_lr'] == 0.2

    group = {'params': []}
    optimizer.init_group(group)
    assert group['step'] == 0
    assert group['initial_lr'] is None


def test_flash_adamw_sparse_gradient():
    param = simple_sparse_parameter()[1]

    optimizer = FlashAdamW([param])
    with pytest.raises(NoSparseGradientError):
        optimizer.step()


@pytest.mark.parametrize(
    ('kwargs', 'error'),
    [
        ({'lr': -1e-3}, NegativeLRError),
        ({'betas': (-0.1, 0.999)}, ValueError),
        ({'betas': (0.9, 1.0)}, ValueError),
        ({'eps': -1e-8}, ValueError),
        ({'weight_decay': -1e-3}, ValueError),
        ({'decouple_lr': 'yes'}, ValueError),
    ],
)
def test_flash_adamw_invalid_parameters(kwargs, error):
    with pytest.raises(error):
        FlashAdamW(None, **kwargs)


def test_load_flash_adamw():
    assert load_optimizer('flashadamw') is FlashAdamW
