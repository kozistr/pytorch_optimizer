from tests.fixtures import TrainingModel
from tests.utils import build_optimizer


def test_adam_mini_optimizer():
    optimizer = build_optimizer('AdamMini', TrainingModel())
    optimizer.step()
