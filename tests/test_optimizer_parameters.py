from typing import List

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from pytorch_optimizer import SAM, Lookahead, load_optimizers


class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


OPTIMIZER_NAMES: List[str] = [
    'adamp',
    'sgdp',
    'madgrad',
    'ranger',
    'ranger21',
    'radam',
    'adabound',
    'adahessian',
    'adabelief',
    'diffgrad',
    'diffrgrad',
    'lamb',
    'ralamb',
]

BETA_OPTIMIZER_NAMES: List[str] = [
    'adabelief',
    'adabound',
    'adahessian',
    'adamp',
    'diffgrad',
    'diffrgrad',
    'lamb',
    'radam',
    'ranger',
    'ranger21',
    'ralamb',
]


@pytest.mark.parametrize('optimizer_names', OPTIMIZER_NAMES)
def test_learning_rate(optimizer_names):
    with pytest.raises(ValueError):
        optimizer = load_optimizers(optimizer_names)
        optimizer(None, lr=-1e-2)


@pytest.mark.parametrize('optimizer_names', OPTIMIZER_NAMES)
def test_epsilon(optimizer_names):
    with pytest.raises(ValueError):
        optimizer = load_optimizers(optimizer_names)
        optimizer(None, eps=-1e-6)


@pytest.mark.parametrize('optimizer_names', OPTIMIZER_NAMES)
def test_weight_decay(optimizer_names):
    with pytest.raises(ValueError):
        optimizer = load_optimizers(optimizer_names)
        optimizer(None, weight_decay=-1e-3)


@pytest.mark.parametrize('optimizer_names', BETA_OPTIMIZER_NAMES)
def test_betas(optimizer_names):
    with pytest.raises(ValueError):
        optimizer = load_optimizers(optimizer_names)
        optimizer(None, betas=(-0.1, 0.1))

    with pytest.raises(ValueError):
        optimizer = load_optimizers(optimizer_names)
        optimizer(None, betas=(0.1, -0.1))


def test_sam_parameters():
    with pytest.raises(ValueError):
        SAM(None, load_optimizers('adamp'), rho=-0.1)


def test_lookahead_parameters():
    model: nn.Module = LogisticRegression()
    parameters = model.parameters()

    Lookahead(load_optimizers('adamp')(parameters), pullback_momentum='reset')
    Lookahead(load_optimizers('adamp')(parameters), pullback_momentum='pullback')

    with pytest.raises(ValueError):
        Lookahead(load_optimizers('adamp')(parameters), k=0)

    with pytest.raises(ValueError):
        Lookahead(load_optimizers('adamp')(parameters), alpha=0)

    with pytest.raises(ValueError):
        Lookahead(load_optimizers('adamp')(parameters), pullback_momentum='asdf')
