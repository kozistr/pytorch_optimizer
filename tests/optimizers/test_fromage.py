from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def test_fromage_zero_norm():
    optimizer = build_optimizer('fromage', [make_parameter(requires_grad=True)])
    optimizer.step()
