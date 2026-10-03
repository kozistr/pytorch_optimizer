import pytest
import torch

from pytorch_optimizer.optimizer import load_optimizer
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def test_rose_optimizer():
    with pytest.raises(ValueError):
        load_optimizer('rose')([make_parameter()], compute_dtype=torch.bfloat16)

    opt = build_optimizer('rose', [make_parameter(())])
    opt.step()

    opt = build_optimizer('rose', [make_parameter()], bf16_sr=False, compute_dtype=torch.float32)
    opt.step()
