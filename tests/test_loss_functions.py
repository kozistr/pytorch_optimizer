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

    assert float(loss) == pytest.approx(expected_loss, abs=1e-6)


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

    assert float(loss) == pytest.approx(expected_loss, abs=1e-6)


@torch.no_grad()
def test_focal_loss():
    criterion = FocalLoss(alpha=1.0, gamma=2.0)

    y_pred = torch.arange(-1.0, 1.0, 0.2)
    y_true = torch.FloatTensor([0.0] * 5 + [1.0] * 5)
    loss = criterion(y_pred, y_true)

    assert float(loss) == pytest.approx(0.07848126, abs=1e-6)


@torch.no_grad()
def test_soft_f1_loss():
    criterion = SoftF1Loss()

    y_pred = torch.sigmoid(torch.arange(-1.0, 1.0, 0.2))
    y_true = torch.FloatTensor([0.0] * 5 + [1.0] * 5)
    loss = criterion(y_pred, y_true)

    np.testing.assert_almost_equal(loss.item(), 0.38905364)


@torch.no_grad()
def test_binary_dice_loss():
    # brought from https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/tests/test_losses.py#L84

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

    criterion = DiceLoss(mode='binary', ignore_index=1)

    y_pred = torch.tensor([0.0, 0.0, 0.0]).view(1, 1, -1)
    y_true = torch.tensor([1, 1, 1]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    with pytest.raises(ValueError):
        DiceLoss(mode='binary', classes=[0])


@torch.no_grad()
def test_multiclass_dice_loss():
    eps: float = 1e-6
    criterion = DiceLoss(mode='multiclass', classes=[0])

    y_pred = torch.tensor([[0.0, 0.1, 0.4], [0.8, 0.3, 0.5], [0.7, 0.9, 0.8]]).view(3, 3, -1)
    y_true = torch.tensor([[1], [0], [2]]).view(3, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.5749718, abs=eps)

    criterion = DiceLoss(mode='multiclass', classes=[0], ignore_index=1)
    y_pred = torch.tensor([[0.0, 0.1, 0.4], [0.8, 0.3, 0.5], [0.7, 0.9, 0.8]]).view(3, 3, -1)
    y_true = torch.tensor([[1], [0], [2]]).view(3, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.506536, abs=eps)


@torch.no_grad()
def test_ldam_loss():
    criterion = LDAMLoss(num_class_list=[1, 2, 3, 4])

    y_pred = torch.FloatTensor([[-0.5, -0.25, 0.25, 0.5], [0.8, -0.25, 0.25, 0.5]])
    y_true = torch.LongTensor([3, 0])
    loss = criterion(y_pred, y_true)

    np.testing.assert_almost_equal(loss.item(), 4.5767049)
