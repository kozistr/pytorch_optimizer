from pytorch_optimizer.optimizer import load_optimizer
from tests.fixtures import make_parameter


def test_mars_c_t_norm():
    param = make_parameter()
    param.grad[0] = 100.0

    optimizer = load_optimizer('mars')([param], optimize_1d=True)
    optimizer.step()
