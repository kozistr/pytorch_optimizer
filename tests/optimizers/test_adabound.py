from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def test_adabound_zero_lr():
    optimizer = build_optimizer('adabound', [make_parameter()], lr=0.0)
    optimizer.param_groups[0]['lr'] = 1e-3
    optimizer.step()
