import pytest

from pytorch_optimizer.optimizer.spam import CosineDecay
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def test_spam_optimizer():
    optimizer = build_optimizer('spam', [make_parameter(grad=None)], density=0.0)
    optimizer.step()

    optimizer = build_optimizer('spam', [make_parameter()], grad_accu_steps=0, update_proj_gap=1)
    optimizer.step()


def test_cosine_decay_rates():
    decay = CosineDecay(1.0, t_max=3, eta_min=0.1)

    rates = [decay.get_death_rate(step) for step in range(4)]

    assert rates == pytest.approx([0.8681980515, 0.55, 0.2318019485, 0.1])
