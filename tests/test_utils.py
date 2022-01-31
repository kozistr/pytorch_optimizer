import numpy as np
import torch

from pytorch_optimizer.utils import clip_grad_norm, has_overflow, normalize_gradient, unit_norm


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
