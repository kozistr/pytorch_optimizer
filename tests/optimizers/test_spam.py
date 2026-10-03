from pytorch_optimizer.optimizer import load_optimizer
from tests.fixtures import make_parameter


def test_spam_optimizer():
    optimizer = load_optimizer('spam')([make_parameter(grad=None)], density=0.0)
    optimizer.step()

    optimizer = load_optimizer('spam')([make_parameter()], grad_accu_steps=0, update_proj_gap=1)
    optimizer.step()
