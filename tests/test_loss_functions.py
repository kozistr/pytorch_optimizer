import numpy as np
import pytest
import torch

from pytorch_optimizer import BCEFocalLoss, BCELoss, DiceLoss, FocalLoss, LDAMLoss, SoftF1Loss


@torch.no_grad()
@pytest.mark.parametrize('recipe', [('train', 0.37069410), ('eval', 0.30851572)])
def test_bce_loss(recipe):
    mode, expected_loss = recipe

    criterion = BCELoss(label_smooth=0.1, eps=1e-6)
    criterion.train(mode == 'train')

    y_pred = torch.arange(0.0, 1.0, 0.1)
    y_true = torch.FloatTensor([0.0] * 5 + [1.0] * 5)
    loss = criterion(y_pred, y_true)

    assert loss == expected_loss


@torch.no_grad()
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


@torch.no_grad()
def test_focal_loss():
    criterion = FocalLoss(alpha=1.0, gamma=2.0)

    y_pred = torch.arange(-1.0, 1.0, 0.2)
    y_true = torch.FloatTensor([0.0] * 5 + [1.0] * 5)
    loss = criterion(y_pred, y_true)

    np.testing.assert_almost_equal(loss.item(), 0.07848126)


@torch.no_grad()
def test_soft_f1_loss():
    criterion = SoftF1Loss()

    y_pred = torch.sigmoid(torch.arange(-1.0, 1.0, 0.2))
    y_true = torch.FloatTensor([0.0] * 5 + [1.0] * 5)
    loss = criterion(y_pred, y_true)

    np.testing.assert_almost_equal(loss.item(), 0.38905364)


@torch.no_grad()
def test_dice_loss():
    eps: float = 1e-6
    criterion = DiceLoss(mode='binary', from_logits=False, label_smooth=0.01)

    # Ideal case
    y_pred = torch.tensor([1.0, 1.0, 1.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([1, 1, 1])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    y_pred = torch.tensor([1.0, 0.0, 1.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([1, 0, 1])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    y_pred = torch.tensor([0.0, 0.0, 0.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([0, 0, 0])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    # Worst case
    y_pred = torch.tensor([1.0, 1.0, 1.0]).view(1, 1, -1)
    y_true = torch.tensor([0, 0, 0]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    y_pred = torch.tensor([1.0, 0.0, 1.0]).view(1, 1, -1)
    y_true = torch.tensor([0, 1, 0]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.996677, abs=eps)

    y_pred = torch.tensor([0.0, 0.0, 0.0]).view(1, 1, -1)
    y_true = torch.tensor([1, 1, 1]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.996677, abs=eps)


@torch.no_grad()
def test_ldam_loss():
    criterion = LDAMLoss()

    y_pred = torch.sigmoid(torch.arange(-1.0, 1.0, 0.2))
    y_true = torch.FloatTensor([0.0] * 5 + [1.0] * 5)
    loss = criterion(y_pred, y_true)

    np.testing.assert_almost_equal(loss.item(), 0.38905364)
