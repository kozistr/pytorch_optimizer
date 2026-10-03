from pytorch_optimizer.optimizer import load_optimizer
from tests.fixtures import make_parameter


def test_adabound_zero_lr():
    optimizer = load_optimizer('adabound')([make_parameter()], lr=0.0)
    optimizer.param_groups[0]['lr'] = 1e-3
    optimizer.step()
