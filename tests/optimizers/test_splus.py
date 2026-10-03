from pytorch_optimizer.optimizer import load_optimizer
from tests.fixtures import make_parameter


def test_splus_methods():
    optimizer = load_optimizer('splus')([make_parameter()])
    optimizer.step()

    optimizer.eval()
    optimizer.train()
