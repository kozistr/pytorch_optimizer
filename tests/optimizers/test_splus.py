from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def test_splus_methods():
    optimizer = build_optimizer('splus', [make_parameter()])
    optimizer.step()

    optimizer.eval()
    optimizer.train()
