from pytorch_optimizer.optimizer import load_optimizer
from tests.fixtures import make_parameter


def test_fromage_zero_norm():
    optimizer = load_optimizer('fromage')([make_parameter(requires_grad=True)])
    optimizer.step()
