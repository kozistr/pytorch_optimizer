import pytest

from pytorch_optimizer import create_optimizer
from tests.utils import LogisticRegression


@pytest.mark.utils
def test_create_optimizer():
    model = LogisticRegression()

    create_optimizer(model, 'adamp', lr=1e-2, weight_decay=1e-3, use_gc=True, use_lookahead=True)
    create_optimizer(model, 'alig', lr=1e-2, use_lookahead=True)
