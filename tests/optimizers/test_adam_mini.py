from pytorch_optimizer.optimizer import load_optimizer
from tests.fixtures import TrainingModel


def test_adam_mini_optimizer():
    optimizer = load_optimizer('AdamMini')(TrainingModel())
    optimizer.step()
