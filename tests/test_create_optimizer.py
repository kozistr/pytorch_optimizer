from pytorch_optimizer import create_optimizer
from tests.utils import LogisticRegression


def test_create_optimizer():
    model = LogisticRegression()

    optimizer = create_optimizer(model, 'adamp', lr=1e-2, weight_decay=1e-3, use_gc=True)
    optimizer = create_optimizer(model, 'alig', lr=1e-2)
