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
from tests.fixtures import make_parameter

BINARY_DICE_RECIPES: tuple[tuple, ...] = (
    ([1.0, 1.0, 1.0], [1, 1, 1], (1, 1, 1, -1), 0.0),
    ([1.0, 0.0, 1.0], [1, 0, 1], (1, 1, 1, -1), 0.0),
    ([0.0, 0.0, 0.0], [0, 0, 0], (1, 1, 1, -1), 0.0),
    ([1.0, 1.0, 1.0], [0, 0, 0], (1, 1, -1), 0.0),
    ([1.0, 0.0, 1.0], [0, 1, 0], (1, 1, -1), 0.996677),
    ([0.0, 0.0, 0.0], [1, 1, 1], (1, 1, -1), 0.996677),
)


class TestBinaryCE:
    @pytest.mark.parametrize('recipe', [('train', 0.42595610), ('eval', 0.30851572)])
    def test_bce_loss(self, recipe, binary_predictions):
        mode, expected_loss = recipe

        criterion = BCELoss(label_smooth=0.1, eps=1e-6)
        criterion.train(mode == 'train')

        y_pred, y_true = binary_predictions
        loss = criterion(y_pred, y_true)

        assert float(loss) == pytest.approx(expected_loss, abs=1e-6)

    def test_focal_loss(self, binary_predictions):
        criterion = FocalLoss(alpha=1.0, gamma=2.0)

        y_pred = torch.arange(-1.0, 1.0, 0.2)
        _, y_true = binary_predictions
        loss = criterion(y_pred, y_true)

        assert float(loss) == pytest.approx(0.07848126, abs=1e-6)

    @pytest.mark.parametrize(
        ('reduction', 'expected_loss'),
        [
            ('none', [0.024584262909110033, 0.04368160706201334, 0.655790168737243]),
            ('mean', 0.24135201290278882),
            ('sum', 0.7240560387083664),
        ],
    )
    def test_focal_cosine_loss(self, reduction, expected_loss):
        criterion = FocalCosineLoss(reduction=reduction)
        y_pred = torch.FloatTensor([[0.9, 0.1, 0.1], [0.2, 0.9, 0.1], [0.2, 0.1, 0.1]])
        y_true = torch.LongTensor([0, 1, 2])
        loss = criterion(y_pred, y_true)
        torch.testing.assert_close(loss, torch.tensor(expected_loss), atol=1e-6, rtol=0)

    def test_soft_f1_loss(self, binary_predictions):
        criterion = SoftF1Loss()

        y_pred = torch.sigmoid(torch.arange(-1.0, 1.0, 0.2))
        _, y_true = binary_predictions
        loss = criterion(y_pred, y_true)

        assert float(loss) == pytest.approx(0.38905364, abs=1e-6)

    @pytest.mark.parametrize('recipe', [(0.5, 0.375), (2.0, 0.70588235)])
    def test_soft_f1_loss_beta(self, recipe):
        beta, expected_loss = recipe

        criterion = SoftF1Loss(beta=beta)

        y_pred = torch.FloatTensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        y_true = torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        loss = criterion(y_pred, y_true)

        assert float(loss) == pytest.approx(expected_loss, abs=1e-5)


class TestDiceAndJaccard:
    def test_soft_dice_score(self):
        y_pred = torch.tensor([1.0, 1.0, 1.0]).view(1, 1, 1, -1)
        y_true = torch.tensor(([1, 0, 1])).view(1, 1, 1, -1)

        dice_score = soft_dice_score(y_pred, y_true, dims=None)
        assert float(dice_score) == pytest.approx(0.8, abs=1e-6)

        dice_score = soft_dice_score(y_pred, y_true, dims=(1, 2)).mean()
        assert float(dice_score) == pytest.approx(0.666666, abs=1e-6)

    def test_soft_jaccard_score(self):
        y_pred = torch.tensor([1.0, 1.0, 1.0]).view(1, 1, 1, -1)
        y_true = torch.tensor(([1, 0, 1])).view(1, 1, 1, -1)

        jaccard_score = soft_jaccard_score(y_pred, y_true, dims=None)
        assert float(jaccard_score) == pytest.approx(0.666666, abs=1e-6)

        jaccard_score = soft_jaccard_score(y_pred, y_true, dims=(1, 2)).mean()
        assert float(jaccard_score) == pytest.approx(0.666666, abs=1e-6)

    @pytest.mark.parametrize(('y_pred', 'y_true', 'pred_view', 'expected_loss'), BINARY_DICE_RECIPES)
    @pytest.mark.parametrize(('criterion', 'options'), [(DiceLoss, {'ignore_index': -100}), (JaccardLoss, {})])
    def test_binary_dice_loss(self, y_pred, y_true, pred_view, expected_loss, criterion, options):
        y_pred = torch.tensor(y_pred).view(*pred_view)
        y_true = torch.tensor(y_true).view(1, 1, 1, -1)
        loss = criterion(mode='binary', from_logits=False, label_smooth=0.01, **options)(y_pred, y_true)

        assert float(loss) == pytest.approx(expected_loss, abs=1e-6)

    @pytest.mark.parametrize(
        ('criterion', 'options', 'expected_loss'),
        [(DiceLoss, {}, 0.5749718), (DiceLoss, {'ignore_index': 1}, 0.506536), (JaccardLoss, {}, 0.730136)],
    )
    def test_multiclass_loss(self, criterion, options, expected_loss):
        y_pred = torch.tensor([[0.0, 0.1, 0.4], [0.8, 0.3, 0.5], [0.7, 0.9, 0.8]]).view(3, 3, 1)
        y_true = torch.tensor([[1], [0], [2]])
        loss = criterion(mode='multiclass', classes=[0], **options)(y_pred, y_true)
        assert loss.item() == pytest.approx(expected_loss, abs=1e-6)

    @pytest.mark.parametrize(
        ('criterion', 'options', 'expected_loss'),
        [(DiceLoss, {}, 0.520958), (DiceLoss, {'ignore_index': 0}, 0.215321), (JaccardLoss, {}, 0.68503928)],
    )
    def test_multilabel_loss(self, criterion, options, expected_loss):
        y_pred = torch.tensor([[0.6, 0.6, 0.6], [0.1, 0.1, 0.1], [0.1, 0.9, 0.1]]).view(3, 3, 1)
        y_true = torch.tensor([[1, 1, 1], [0, 0, 0], [0, 0, 1]]).view(3, 3, 1)
        loss = criterion(mode='multilabel', classes=[0], **options)(y_pred, y_true)
        assert loss.item() == pytest.approx(expected_loss, abs=1e-6)

    @pytest.mark.parametrize('criterion', [DiceLoss, JaccardLoss])
    def test_binary_not_supported(self, criterion):
        with pytest.raises(ValueError):
            criterion(mode='binary', classes=[0])


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
    y_pred = make_parameter((4, 4), grad=None)
    y_true = torch.LongTensor([0, 1, 2, 3])

    loss = bi_tempered_logistic_loss(y_pred, y_true, t1=0.5, t2=0.5, reduction='mean')
    loss.backward()

    assert torch.isfinite(y_pred.grad).all()
    assert y_pred.grad.abs().sum() > 0


def test_binary_bi_tempered_log_loss_exception():
    criterion = BinaryBiTemperedLogisticLoss(0.8, 2.0, label_smooth=0.1, ignore_index=-100, reduction='mean')
    with pytest.raises(ValueError):
        criterion(torch.zeros(1, 1), torch.zeros(1, 2))


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


def test_tverysky_loss():
    criterion = TverskyLoss(alpha=0.5, beta=0.5)

    y_pred = torch.arange(0.0, 1.0, 0.1)
    y_true = torch.FloatTensor([0.0] * 5 + [1.0] * 5)

    loss = criterion(y_pred, y_true)

    assert float(loss) == pytest.approx(0.3978933, abs=1e-6)


def test_focal_tverysky_loss():
    criterion = FocalTverskyLoss(alpha=0.5, beta=0.5, gamma=0.5)

    y_pred = torch.arange(0.0, 1.0, 0.1)
    y_true = torch.FloatTensor([0.0] * 5 + [1.0] * 5)

    loss = criterion(y_pred, y_true)

    assert float(loss) == pytest.approx(0.6307878, abs=1e-6)


@pytest.mark.parametrize(('per_image', 'expected_loss'), [(True, 1.625), (False, 7.0 / 6.0)])
def test_lovasz_hinge_loss(per_image, expected_loss):
    criterion = LovaszHingeLoss(per_image)

    y_pred = torch.tensor([[2.0, -1.0], [0.5, 1.5]]).view(2, 1, 1, 2)
    y_true = torch.zeros_like(y_pred)
    y_true[1] = 1.0

    loss = criterion(y_pred, y_true)

    assert float(loss) == pytest.approx(expected_loss, abs=1e-6)


@pytest.mark.parametrize(
    ('dtype', 'alpha', 'gamma', 'target_kind', 'reduction'),
    [
        (torch.float32, 0.25, 2.0, 'mixed', 'none'),
        (torch.float64, 0.25, 2.0, 'mixed', 'mean'),
        (torch.float64, 0.25, 2.0, 'mixed', 'sum'),
        (torch.float32, 0.0, 2.0, 'mixed', 'none'),
        (torch.float32, 1.0, 2.0, 'mixed', 'none'),
        (torch.float64, 0.25, 0.0, 'negative', 'mean'),
        (torch.float64, 0.25, 0.0, 'positive', 'sum'),
        (torch.float64, 0.25, 2.0, 'negative', 'none'),
        (torch.float64, 0.25, 2.0, 'positive', 'none'),
    ],
)
def test_bce_focal_class_weights(dtype, alpha, gamma, target_kind, reduction):
    probabilities = torch.tensor([[0.1, 0.4], [0.8, 0.9], [0.3, 0.7]], dtype=dtype).t()
    targets = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=dtype).t()
    if target_kind != 'mixed':
        targets.fill_(float(target_kind == 'positive'))
    original_targets = targets.clone()
    probabilities.requires_grad_()
    reference_probabilities = probabilities.detach().clone().requires_grad_()

    # Independent positive/negative branches of Lin et al.'s alpha-balanced focal loss.
    positive = -alpha * (1 - reference_probabilities).pow(gamma) * reference_probabilities.log()
    negative = -(1 - alpha) * reference_probabilities.pow(gamma) * torch.log1p(-reference_probabilities)
    expected = torch.where(targets == 1, positive, negative)
    if reduction == 'mean':
        expected = expected.mean()
    elif reduction == 'sum':
        expected = expected.sum()

    actual = BCEFocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)(probabilities, targets)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.sum(), probabilities)[0]
    expected_grad = torch.autograd.grad(expected.sum(), reference_probabilities)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    torch.testing.assert_close(probabilities, reference_probabilities)
    torch.testing.assert_close(targets, original_targets)


@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
def test_bce_focal_label_smoothing(training, reduction):
    probabilities = torch.tensor([[0.2, 0.8], [0.4, 0.6]], dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    reference_probabilities = probabilities.detach().clone().requires_grad_()
    smoothed_targets = 0.8 * targets + 0.1 if training else targets
    expected_bce = -(
        smoothed_targets * reference_probabilities.log()
        + (1 - smoothed_targets) * torch.log1p(-reference_probabilities)
    )
    weights = torch.where(
        targets == 1,
        0.25 * (1 - reference_probabilities).square(),
        0.75 * reference_probabilities.square(),
    )
    expected = weights * expected_bce
    if reduction == 'mean':
        expected = expected.mean()
    elif reduction == 'sum':
        expected = expected.sum()

    criterion = BCEFocalLoss(alpha=0.25, gamma=2, label_smooth=0.2, reduction=reduction)
    criterion.train(training)
    actual = criterion(probabilities, targets)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), probabilities)[0],
        torch.autograd.grad(expected.sum(), reference_probabilities)[0],
    )


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('gamma', [0.0, 2.0])
def test_bce_focal_probability_boundaries(dtype, gamma):
    probabilities = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=dtype, requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=dtype)
    criterion = BCEFocalLoss(alpha=0.25, gamma=gamma, reduction='none')
    actual = criterion(probabilities, targets)
    clamped = probabilities.detach().clamp(1e-6, 1 - 1e-6)
    positive = -0.25 * (1 - probabilities.detach()).pow(gamma) * clamped.log()
    negative = -0.75 * probabilities.detach().pow(gamma) * torch.log1p(-clamped)
    torch.testing.assert_close(actual, torch.where(targets == 1, positive, negative))
    assert torch.isfinite(torch.autograd.grad(actual.sum(), probabilities)[0]).all()
