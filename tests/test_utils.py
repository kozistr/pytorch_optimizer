from typing import List

import numpy as np
import torch
from torch import nn

from pytorch_optimizer.utils import (
    clip_grad_norm,
    get_optimizer_parameters,
    has_overflow,
    is_valid_parameters,
    normalize_gradient,
    unit_norm,
)
from tests.utils import Example


def test_has_overflow():
    assert has_overflow(np.inf)
    assert has_overflow(np.nan)
    assert not has_overflow(torch.Tensor([1]))


def test_normalized_gradient():
    x = torch.arange(0, 10, dtype=torch.float32)

    np.testing.assert_allclose(
        normalize_gradient(x).numpy(),
        np.asarray([0.0000, 0.3303, 0.6606, 0.9909, 1.3212, 1.6514, 1.9817, 2.3120, 2.6423, 2.9726]),
        rtol=1e-4,
        atol=1e-4,
    )

    np.testing.assert_allclose(
        normalize_gradient(x.view(1, 10), use_channels=True).numpy(),
        np.asarray([[0.0000, 0.3303, 0.6606, 0.9909, 1.3212, 1.6514, 1.9817, 2.3120, 2.6423, 2.9726]]),
        rtol=1e-4,
        atol=1e-4,
    )


def test_clip_grad_norm():
    x = torch.arange(0, 10, dtype=torch.float32, requires_grad=True)
    x.grad = torch.arange(0, 10, dtype=torch.float32)

    np.testing.assert_approx_equal(clip_grad_norm(x), 16.881943016134134, significant=4)
    np.testing.assert_approx_equal(clip_grad_norm(x, max_norm=2), 16.881943016134134, significant=4)


def test_unit_norm():
    x = torch.arange(0, 10, dtype=torch.float32)

    np.testing.assert_approx_equal(unit_norm(x).numpy(), 16.8819, significant=4)
    np.testing.assert_approx_equal(unit_norm(x.view(1, 10)).numpy(), 16.8819, significant=4)
    np.testing.assert_approx_equal(unit_norm(x.view(1, 10, 1, 1)).numpy(), 16.8819, significant=4)
    np.testing.assert_approx_equal(unit_norm(x.view(1, 10, 1, 1, 1, 1)).numpy(), 16.8819, significant=4)


def test_get_optimizer_parameters():
    model: nn.Module = Example()
    wd_ban_list: List[str] = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    before_parameters = list(model.named_parameters())
    after_parameters = get_optimizer_parameters(model, weight_decay=1e-3, wd_ban_list=wd_ban_list)

    for before, after in zip(before_parameters, after_parameters):
        layer_name: str = before[0]
        if layer_name.find('bias') != -1 or layer_name in wd_ban_list:
            assert after['weight_decay'] == 0.0


def test_is_valid_parameters():
    model: nn.Module = Example()
    wd_ban_list: List[str] = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    after_parameters = get_optimizer_parameters(model, weight_decay=1e-3, wd_ban_list=wd_ban_list)

    assert is_valid_parameters(after_parameters)
