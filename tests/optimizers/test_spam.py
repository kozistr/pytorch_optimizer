from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def test_spam_optimizer():
    optimizer = build_optimizer('spam', [make_parameter(grad=None)], density=0.0)
    optimizer.step()

    optimizer = build_optimizer('spam', [make_parameter()], grad_accu_steps=0, update_proj_gap=1)
    optimizer.step()
