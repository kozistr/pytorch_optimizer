import numpy as np
import pytest
import torch

from pytorch_optimizer import BCEFocalLoss, BCELoss, DiceLoss, FocalLoss, LDAMLoss, SoftF1Loss


@pytest.mark.parametrize('recipe', [('train', 0.37069410), ('eval', 0.30851572)])
def test_bce_loss(recipe):
    mode, expected_loss = recipe

    criterion = BCELoss(label_smooth=0.1, eps=1e-6)
    criterion.train(mode == 'train')

    y_pred = torch.arange(0.0, 1.0, 0.1)
    y_true = torch.FloatTensor([0.0] * 5 + [1.0] * 5)
    loss = criterion(y_pred, y_true)

    assert loss == expected_loss


@pytest.mark.parametrize(
    'recipe',
    [
        ('train', 'mean', 0.16992925),
        ('eval', 'mean', 0.14931047),
        ('train', 'sum', 1.699292540),
        ('eval', 'sum', 1.49310469),
    ],
)
def test_bce_focal_loss(recipe):
    mode, reduction, expected_loss = recipe

    criterion = BCEFocalLoss(alpha=1.0, gamma=2.0, label_smooth=0.1, eps=1e-6, reduction=reduction)
    criterion.train(mode == 'train')

    y_pred = torch.arange(0.0, 1.0, 0.1)
    y_true = torch.FloatTensor([0.0] * 5 + [1.0] * 5)
    loss = criterion(y_pred, y_true)

    assert loss == expected_loss


def test_focal_loss():
    criterion = FocalLoss(alpha=1.0, gamma=2.0)

    y_pred = torch.arange(-1.0, 1.0, 0.2)
    y_true = torch.FloatTensor([0.0] * 5 + [1.0] * 5)
    loss = criterion(y_pred, y_true)

    np.testing.assert_almost_equal(loss.item(), 0.07848126)
