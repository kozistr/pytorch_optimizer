import pytest
import torch

from pytorch_optimizer.optimizer import load_optimizer
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


def test_rose_optimizer():
    with pytest.raises(ValueError):
        load_optimizer('rose')([make_parameter()], compute_dtype=torch.bfloat16)

    scalar, matrix = make_parameter((), grad=1.0), make_parameter(grad=1.0)
    optimizer = build_optimizer('rose', [scalar, matrix], lr=0.1, bf16_sr=False, compute_dtype=torch.float32)
    optimizer.step()

    torch.testing.assert_close(scalar, torch.tensor(-0.1))
    torch.testing.assert_close(matrix, torch.zeros_like(matrix))
    torch.testing.assert_close(matrix.grad, torch.ones_like(matrix))
