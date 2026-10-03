import pytest

from pytorch_optimizer.optimizer import load_optimizer
from tests.fixtures import make_parameter


@pytest.mark.parametrize('optimizer_name', ['emonavi', 'emolynx', 'emofact'])
def test_emo_optimizers(optimizer_name):
    optimizer = load_optimizer(optimizer_name)([make_parameter()], use_shadow=True)

    optimizer.init_group(optimizer.param_groups[0])
    optimizer.state['ema'] = {'short': 1.0, 'medium': 5.0, 'long': 40.0}

    optimizer.step()
