import pytest

from pytorch_optimizer import AdamP
from pytorch_optimizer.lr_scheduler.cosine_anealing import CosineAnnealingWarmupRestarts
from tests.utils import Example


def test_cosine_annealing_warmup_restarts_params():
    model = Example()
    optimizer = AdamP(model.parameters())

    with pytest.raises(ValueError):
        CosineAnnealingWarmupRestarts(
            optimizer=optimizer,
            first_cycle_steps=10,
            warmup_steps=20,
        )
