import pytest

from pytorch_optimizer import create_optimizer, load_optimizer
from tests.utils import LogisticRegression


def test_create_optimizer():
    model = LogisticRegression()

    create_optimizer(model, 'adamp', lr=1e-2, weight_decay=1e-3, use_gc=True, use_lookahead=True)
    create_optimizer(model, 'alig', lr=1e-2, use_lookahead=True)


def test_bnb_optimizer():
    with pytest.raises(ImportError):
        load_optimizer('bnb_adamw8bit')


def test_q_galore_optimizer():
    with pytest.raises(ImportError):
        load_optimizer('q_galore_adamw8bit')
