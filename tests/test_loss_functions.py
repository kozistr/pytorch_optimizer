import pytest
import torch

from pytorch_optimizer.loss import (
    BCEFocalLoss,
    BCELoss,
    BinaryBiTemperedLogisticLoss,
    BiTemperedLogisticLoss,
    DiceLoss,
    FocalCosineLoss,
    FocalLoss,
    FocalTverskyLoss,
    JaccardLoss,
    LDAMLoss,
    LovaszHingeLoss,
    SoftF1Loss,
    TverskyLoss,
    soft_dice_score,
    soft_jaccard_score,
)
from pytorch_optimizer.loss.bi_tempered import bi_tempered_logistic_loss
from tests.constants import BINARY_DICE_RECIPES
from tests.utils import MultiClassExample


class TestBinaryCE:

    @torch.no_grad()
    @pytest.mark.parametrize('recipe', [('train', 0.37069410), ('eval', 0.30851572)])
    def test_bce_loss(self, recipe, binary_predictions):
        mode, expected_loss = recipe

        criterion = BCELoss(label_smooth=0.1, eps=1e-6)
        criterion.train(mode == 'train')

        y_pred, y_true = binary_predictions
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
    def test_bce_focal_loss(self, recipe, binary_predictions):
        mode, reduction, expected_loss = recipe

        criterion = BCEFocalLoss(alpha=1.0, gamma=2.0, label_smooth=0.1, eps=1e-6, reduction=reduction)
        criterion.train(mode == 'train')

        y_pred, y_true = binary_predictions
        loss = criterion(y_pred, y_true)

        assert float(loss) == pytest.approx(expected_loss, abs=1e-6)

    @torch.no_grad()
    def test_focal_loss(self, binary_predictions):
        criterion = FocalLoss(alpha=1.0, gamma=2.0)

        y_pred = torch.arange(-1.0, 1.0, 0.2)
        _, y_true = binary_predictions
        loss = criterion(y_pred, y_true)

        assert float(loss) == pytest.approx(0.07848126, abs=1e-6)

    @torch.no_grad()
    def test_focal_cosine_loss(self):
        criterion = FocalCosineLoss(alpha=1.0, gamma=2.0, focal_weight=0.1)

        y_pred = torch.FloatTensor([[0.9, 0.1, 0.1], [0.2, 0.9, 0.1], [0.2, 0.1, 0.1]])
        y_true = torch.LongTensor([0, 1, 2])
        loss = criterion(y_pred, y_true)

        assert float(loss) == pytest.approx(0.2413520, abs=1e-6)

    @torch.no_grad()
    def test_soft_f1_loss(self, binary_predictions):
        criterion = SoftF1Loss()

        y_pred = torch.sigmoid(torch.arange(-1.0, 1.0, 0.2))
        _, y_true = binary_predictions
        loss = criterion(y_pred, y_true)

        assert float(loss) == pytest.approx(0.38905364, abs=1e-6)


class TestDiceAndJaccard:
    eps: float = 1e-6

    @torch.no_grad()
    def test_soft_dice_score(self):
        y_pred = torch.tensor([1.0, 1.0, 1.0]).view(1, 1, 1, -1)
        y_true = torch.tensor(([1, 0, 1])).view(1, 1, 1, -1)

        dice_score = soft_dice_score(y_pred, y_true, dims=None)
        assert float(dice_score) == pytest.approx(0.8, abs=self.eps)

        dice_score = soft_dice_score(y_pred, y_true, dims=(1, 2)).mean()
        assert float(dice_score) == pytest.approx(0.666666, abs=self.eps)

    @torch.no_grad()
    def test_soft_jaccard_score(self):
        y_pred = torch.tensor([1.0, 1.0, 1.0]).view(1, 1, 1, -1)
        y_true = torch.tensor(([1, 0, 1])).view(1, 1, 1, -1)

        jaccard_score = soft_jaccard_score(y_pred, y_true, dims=None)
        assert float(jaccard_score) == pytest.approx(0.666666, abs=self.eps)

        jaccard_score = soft_jaccard_score(y_pred, y_true, dims=(1, 2)).mean()
        assert float(jaccard_score) == pytest.approx(0.666666, abs=self.eps)

    @torch.no_grad()
    @pytest.mark.parametrize('y_pred,y_true,pred_view,expected_loss', BINARY_DICE_RECIPES)
    @pytest.mark.parametrize(
        'criterion',
        [
            DiceLoss(mode='binary', from_logits=False, label_smooth=0.01, ignore_index=-100),
            JaccardLoss(mode='binary', from_logits=False, label_smooth=0.01),
        ],
    )
    def test_binary_dice_loss(self, y_pred, y_true, pred_view, expected_loss, criterion):
        # brought from https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/tests/test_losses.py#L84
        y_pred = torch.tensor(y_pred).view(*pred_view)
        y_true = torch.tensor(y_true).view(1, 1, 1, -1)
        loss = criterion(y_pred, y_true)

        assert float(loss) == pytest.approx(expected_loss, abs=self.eps)

    @torch.no_grad()
    def test_multiclass_dice_loss(self):
        y_pred = torch.tensor([[0.0, 0.1, 0.4], [0.8, 0.3, 0.5], [0.7, 0.9, 0.8]]).view(3, 3, -1)
        y_true = torch.tensor([[1], [0], [2]]).view(3, -1)

        criterion = DiceLoss(mode='multiclass', classes=[0])
        loss = criterion(y_pred, y_true)
        assert float(loss) == pytest.approx(0.5749718, abs=self.eps)

        criterion = DiceLoss(mode='multiclass', classes=[0], ignore_index=1)
        loss = criterion(y_pred, y_true)
        assert float(loss) == pytest.approx(0.506536, abs=self.eps)

    @torch.no_grad()
    def test_multilabel_dice_loss(self):
        y_pred = torch.tensor([[0.6, 0.6, 0.6], [0.1, 0.1, 0.1], [0.1, 0.9, 0.1]]).view(3, 3, -1)
        y_true = torch.tensor([[1, 1, 1], [0, 0, 0], [0, 0, 1]]).view(3, 3, -1)

        criterion = DiceLoss(mode='multilabel', classes=[0])
        loss = criterion(y_pred, y_true)
        assert float(loss) == pytest.approx(0.520958, abs=self.eps)

        criterion = DiceLoss(mode='multilabel', classes=[0], ignore_index=0)
        loss = criterion(y_pred, y_true)
        assert float(loss) == pytest.approx(0.215321, abs=self.eps)

    @torch.no_grad()
    def test_multiclass_jaccard_loss(self):
        y_pred = torch.tensor([[0.0, 0.1, 0.4], [0.8, 0.3, 0.5], [0.7, 0.9, 0.8]]).view(3, 3, -1)
        y_true = torch.tensor([[1], [0], [2]]).view(3, -1)

        criterion = JaccardLoss(mode='multiclass', classes=[0])
        loss = criterion(y_pred, y_true)

        assert float(loss) == pytest.approx(0.730136, abs=self.eps)

    @torch.no_grad()
    def test_multilabel_jaccard_loss(self):
        y_pred = torch.tensor([[0.6, 0.6, 0.6], [0.1, 0.1, 0.1], [0.1, 0.9, 0.1]]).view(3, 3, -1)
        y_true = torch.tensor([[1, 1, 1], [0, 0, 0], [0, 0, 1]]).view(3, 3, -1)

        criterion = JaccardLoss(mode='multilabel', classes=[0])
        loss = criterion(y_pred, y_true)
        assert float(loss) == pytest.approx(0.68503928, abs=self.eps)

    @pytest.mark.parametrize('criterion', [DiceLoss, JaccardLoss])
    def test_binary_not_supported(self, criterion):
        with pytest.raises(ValueError):
            criterion(mode='binary', classes=[0])


@torch.no_grad()
def test_ldam_loss():
    criterion = LDAMLoss(num_class_list=[1, 2, 3, 4])

    y_pred = torch.FloatTensor([[-0.5, -0.25, 0.25, 0.5], [0.8, -0.25, 0.25, 0.5]])
    y_true = torch.LongTensor([3, 0])
    loss = criterion(y_pred, y_true)

    assert loss.item() == pytest.approx(4.5767049, abs=1e-6)


def test_bi_tempered_log_loss_func():
    y_pred = torch.FloatTensor(
        [[0.1, 0.2, 0.3, 0.4], [0.1, 0.5, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]
    )
    y_true = torch.LongTensor([0, 1, 2, 3])

    loss = bi_tempered_logistic_loss(y_pred, y_true, t1=0.5, t2=1.0, reduction='mean')
    assert loss == pytest.approx(0.6417, abs=1e-4)

    loss = bi_tempered_logistic_loss(y_pred, y_true, t1=0.5, t2=1.0, reduction='sum')
    assert loss == pytest.approx(2.5668, abs=1e-4)


def test_bi_tempered_log_loss_bwd():
    model = MultiClassExample(num_classes=4)

    y_pred = model(torch.randn(4, 1))
    y_true = torch.LongTensor([0, 1, 2, 3])

    loss = bi_tempered_logistic_loss(y_pred, y_true, t1=0.5, t2=0.5, reduction='mean')
    loss.backward()


def test_binary_bi_tempered_log_loss_exception():
    criterion = BinaryBiTemperedLogisticLoss(0.8, 2.0, label_smooth=0.1, ignore_index=-100, reduction='mean')
    with pytest.raises(ValueError):
        criterion(torch.zeros(1, 1), torch.zeros(1, 2))


@torch.no_grad()
@pytest.mark.parametrize(
    'recipe',
    [('mean', 0.939503), ('sum', 3.758012), ('none', torch.FloatTensor([0.9840, 0.9139, 0.9412, 0.9190]))],
)
def test_bi_tempered_log_loss(recipe):
    reduction, expected_loss = recipe

    criterion = BiTemperedLogisticLoss(1.0, 2.0, label_smooth=0.1, ignore_index=-100, reduction=reduction)

    y_pred = torch.FloatTensor(
        [[0.1, 0.2, 0.3, 0.4], [0.1, 0.5, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]
    )
    y_true = torch.LongTensor([0, 1, 2, 3])

    loss = criterion(y_pred, y_true)

    if reduction == 'none':
        torch.testing.assert_close(loss, expected_loss, rtol=1e-4, atol=1e-4)
    else:
        assert float(loss) == pytest.approx(expected_loss, abs=1e-6)


@torch.no_grad()
@pytest.mark.parametrize(
    'recipe', [('mean', 0.0306684), ('sum', 0.0613368), ('none', torch.FloatTensor([[[0.0000, 0.0613]]]))]
)
def test_binary_bi_tempered_log_loss(recipe):
    reduction, expected_loss = recipe

    criterion = BinaryBiTemperedLogisticLoss(0.8, 2.0, label_smooth=0.1, ignore_index=-100, reduction=reduction)

    y_pred = torch.FloatTensor([[[-0.9108, -1.2545]]])
    y_true = (y_pred > 0).type_as(y_pred)
    y_true[:, :, ::2] = -100

    loss = criterion(y_pred, y_true)

    if reduction == 'none':
        torch.testing.assert_close(loss, expected_loss, rtol=1e-4, atol=1e-4)
    else:
        assert float(loss) == pytest.approx(expected_loss, abs=1e-6)


@torch.no_grad()
def test_tverysky_loss():
    criterion = TverskyLoss(alpha=0.5, beta=0.5)

    y_pred = torch.arange(0.0, 1.0, 0.1)
    y_true = torch.FloatTensor([0.0] * 5 + [1.0] * 5)

    loss = criterion(y_pred, y_true)

    assert float(loss) == pytest.approx(0.3978933, abs=1e-6)


@torch.no_grad()
def test_focal_tverysky_loss():
    criterion = FocalTverskyLoss(alpha=0.5, beta=0.5, gamma=0.5)

    y_pred = torch.arange(0.0, 1.0, 0.1)
    y_true = torch.FloatTensor([0.0] * 5 + [1.0] * 5)

    loss = criterion(y_pred, y_true)

    assert float(loss) == pytest.approx(0.6307878, abs=1e-6)


@torch.no_grad()
@pytest.mark.parametrize('recipe', [(True, 1.74925303), (False, 1.08580458)])
def test_lovasz_hinge_loss(recipe):
    per_image, expected_loss = recipe

    criterion = LovaszHingeLoss(per_image)

    y_pred = torch.FloatTensor(
        [
            [
                [
                    [1.9269, 1.4873, 0.9007, -2.1055],
                    [0.6784, -1.2345, -0.0431, -1.6047],
                    [-0.7521, 1.6487, -0.3925, -1.4036],
                    [-0.7279, -0.5594, -0.7688, 0.7624],
                ]
            ],
            [
                [
                    [1.6423, -0.1596, -0.4974, 0.4396],
                    [-0.7581, 1.0783, 0.8008, 1.6806],
                    [1.2791, 1.2964, 0.6105, 1.3347],
                    [-0.2316, 0.0418, -0.2516, 0.8599],
                ]
            ],
        ]
    )
    y_true = torch.zeros_like(y_pred)
    y_true[1] = 1.0

    loss = criterion(y_pred, y_true)

    assert float(loss) == pytest.approx(expected_loss, abs=1e-6)
