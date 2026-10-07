import pytest

from pytorch_optimizer.optimizer import load_optimizer
from tests.fixtures import make_parameter


@pytest.mark.parametrize(
    ('optimizer_name', 'option', 'value'),
    [
        ('shampoo', 'matrix_eps', -1e-6),
        ('scalableshampoo', 'diagonal_eps', -1e-6),
        ('scalableshampoo', 'matrix_eps', -1e-6),
        ('adafactor', 'eps1', -1e-6),
        ('adafactor', 'eps2', -1e-6),
        ('came', 'eps1', -1e-6),
        ('came', 'eps2', -1e-6),
        ('accsgd', 'xi', -0.1),
        ('accsgd', 'kappa', -0.1),
        ('accsgd', 'constant', 42),
        ('asgd', 'amplifier', -1.0),
        ('lars', 'dampening', -0.1),
        ('lars', 'trust_coefficient', -1e-3),
        ('ranger', 'alpha', -0.1),
        ('ranger', 'k', -1),
        ('prodigy', 'beta3', -0.1),
        ('apollodqn', 'rebound', 'dummy'),
        ('apollodqn', 'weight_decay_type', 'dummy'),
    ],
)
def test_invalid_optimizer_options(optimizer_name, option, value):
    with pytest.raises(ValueError):
        load_optimizer(optimizer_name)([make_parameter()], **{option: value})
