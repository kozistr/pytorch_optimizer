import pytest
import torch

from pytorch_optimizer.optimizer.psgd import initialize_q_expressions, precondition_update_prob_schedule
from pytorch_optimizer.optimizer.psgd_utils import (
    damped_pair_vg,
    norm_lower_bound,
    triu_with_diagonal_and_above,
    update_precondition_dense,
    woodbury_identity,
)
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


@pytest.mark.parametrize('probability', [1.0, precondition_update_prob_schedule()])
def test_kron_optimizer(probability):
    params = [make_parameter(shape, grad=1.0) for shape in ((1, 1), (1,))]
    optimizer = build_optimizer(
        'kron',
        params,
        weight_decay=1e-3,
        pre_conditioner_update_probability=probability,
        balance_prob=1.0,
        mu_dtype=torch.bfloat16,
    )
    optimizer.step()
    for param in params:
        assert torch.isfinite(param).all()
        assert torch.count_nonzero(param) == param.numel()


class TestPSGDUtils:
    def test_damped_pair_vg(self):
        x = torch.zeros(2)
        y = damped_pair_vg(x)[1]

        torch.testing.assert_close(x, y)

    def test_norm_lower_bound(self):
        x = torch.zeros(1)
        y = norm_lower_bound(x)
        torch.testing.assert_close(y, x.squeeze())

        x = torch.FloatTensor([[1, 1]])
        y = norm_lower_bound(x)
        torch.testing.assert_close(y, torch.tensor(1.4142135))

        x = torch.FloatTensor([[2, 1], [2, 1]])
        y = norm_lower_bound(x)
        torch.testing.assert_close(y, torch.tensor(3.16227769))

    def test_woodbury_identity(self):
        x = torch.FloatTensor([[1]])
        woodbury_identity(x, x, x)

    def test_triu_with_diagonal_and_above(self):
        x = torch.FloatTensor([[1, 2], [3, 4]])
        y = triu_with_diagonal_and_above(x)
        torch.testing.assert_close(y, torch.FloatTensor([[1, 4], [0, 4]]))

    def test_update_precondition_dense(self):
        q = torch.FloatTensor([[1]])
        dxs = [q] * 1
        dgs = [q] * 1

        y = update_precondition_dense(q, dxs, dgs)

        torch.testing.assert_close(y, q)

    def test_initialize_q_expressions(self):
        x = torch.zeros(1)
        _ = initialize_q_expressions(x.squeeze(), 0.0, 0, 0, None)

        with pytest.raises(ValueError):
            initialize_q_expressions(x.expand(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), 0.0, 0, 0, None)

        with pytest.raises(NotImplementedError):
            initialize_q_expressions(x, 0.0, 0, 0, 'invalid')

        for memory_save_mode in ('one_diag', 'all_diag', 'smart_one_diag'):
            initialize_q_expressions(torch.FloatTensor([[1], [2]]), 0.0, 0, 0, memory_save_mode)
