from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def test_mars_c_t_norm():
    param = make_parameter()
    param.grad[0] = 100.0

    optimizer = build_optimizer('mars', [param], optimize_1d=True)
    optimizer.step()
