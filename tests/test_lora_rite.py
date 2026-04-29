import pytest
import torch

from pytorch_optimizer.base.exception import NegativeLRError, NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.optimizer import LoRARite, load_optimizer
from pytorch_optimizer.optimizer.lora_rite import _LoRARiteHelper
from tests.utils import simple_complex_parameter, simple_parameter, simple_sparse_parameter


def _paired_parameters():
    param_left = torch.nn.Parameter(torch.tensor([[1.0, 0.2, -0.3], [0.8, 0.5, -0.7]]))
    param_right = torch.nn.Parameter(torch.tensor([[0.6, -0.4], [0.1, 0.9], [-0.8, 0.3], [0.7, -0.2]]))
    param_left.grad = torch.tensor([[0.2, 0.3, -0.2], [-0.1, 0.4, 0.5]])
    param_right.grad = torch.tensor([[0.1, -0.3], [0.2, 0.4], [-0.5, 0.1], [0.3, -0.2]])
    return param_left, param_right


def test_lora_rite_helper_methods():
    helper = _LoRARiteHelper()
    tensor = torch.arange(6.0).reshape(2, 3)
    moved, shape = helper.move_lora_dim_to_last(tensor, 0)

    assert moved.shape == (3, 2)
    assert torch.equal(helper.restore_original_shape_and_dim(moved, 0, shape), tensor)
    assert helper.move_lora_dim_to_last(torch.tensor(1.0), 0)[0].shape == (1, 1)
    assert torch.isnan(helper.inf_to_nan(torch.tensor(float('inf'))))
    assert torch.isinf(_LoRARiteHelper(maybe_inf_to_nan=False).inf_to_nan(torch.tensor(float('inf'))))
    assert helper.bias_corrected_decay(0, 0.9) == 0.0

    symmetric = helper.make_symmetric(torch.tensor([[1.0, 2.0], [0.0, 3.0]]))
    assert torch.allclose(symmetric, torch.tensor([[1.0, 1.0], [1.0, 3.0]]))

    preconditioner = helper.create_preconditioner(torch.zeros(3, 2))
    assert preconditioner.shape == (2, 2)

    inverse_root = helper.inverse_sqrt(torch.eye(2), torch.tensor(0.0), eps=1e-6, eps_root=0.0, relative_epsilon=False)
    assert torch.allclose(inverse_root, torch.eye(2) / (1.0 + 1e-6))

    relative_inverse_root = helper.inverse_sqrt(
        torch.eye(2), torch.tensor(0.0), eps=1e-6, eps_root=1e-4, relative_epsilon=True
    )
    assert torch.isfinite(relative_inverse_root).all()

    update = torch.tensor([[3.0, 4.0]])
    assert torch.allclose(helper.skip_update(update, 1.0), torch.zeros_like(update))
    assert torch.equal(helper.skip_update(update, 10.0), update)
    assert helper.reduce_rms(helper.clip_update(update, 1.0)) <= 1.0

    escape = helper.get_unmagnified_rotate_second_escape(torch.zeros(2, 2), torch.eye(2))
    assert torch.allclose(escape, torch.tensor(1.0))


def test_lora_rite_updates_pair_and_state():
    param_left, param_right = _paired_parameters()
    initial_left = param_left.detach().clone()
    initial_right = param_right.detach().clone()
    optimizer = LoRARite(
        [param_left, param_right],
        lr=1e-2,
        betas=(0.0, 0.0),
        weight_decay=1e-3,
        clip_unmagnified_grad=0.0,
        update_skipping=0.0,
    )

    assert str(optimizer) == 'LoRARite'
    assert len(optimizer.iter_lora_pairs({'params': [param_left, param_right, simple_parameter()]})) == 1
    assert optimizer.step(lambda: 1.0) == 1.0

    state = optimizer.state[param_left]
    assert state['step'] == 1
    assert not torch.allclose(param_left, initial_left)
    assert not torch.allclose(param_right, initial_right)
    for key in ('v_l', 'v_r', 'm_l', 'm_r', 'escape_l', 'escape_r'):
        assert torch.isfinite(state[key]).all()


def test_lora_rite_rich_options_and_existing_state():
    param_left, param_right = _paired_parameters()
    optimizer = LoRARite(
        [param_left, param_right],
        lr=5e-3,
        betas=(0.5, 0.9),
        eps=1e-4,
        relative_epsilon=True,
        clip_unmagnified_grad=1e-3,
        update_capping=1e-2,
        update_skipping=10.0,
        apply_escape=True,
        balance_param=True,
        maximize=True,
        maybe_inf_to_nan=False,
    )

    optimizer.step()
    param_left.grad = torch.full_like(param_left, 0.3)
    param_right.grad = torch.full_like(param_right, -0.2)
    optimizer.step()

    state = optimizer.state[param_left]
    assert state['step'] == 2
    assert torch.isfinite(param_left).all()
    assert torch.isfinite(param_right).all()
    assert torch.linalg.norm(param_left).sub(torch.linalg.norm(param_right)).abs() < 1e-4


def test_lora_rite_skips_large_updates_and_missing_pair():
    param_left, param_right = _paired_parameters()
    initial_left = param_left.detach().clone()
    initial_right = param_right.detach().clone()
    optimizer = LoRARite([param_left, param_right], lr=1e-1, betas=(0.0, 0.0), update_skipping=1e-12)
    optimizer.step()

    assert torch.allclose(param_left, initial_left)
    assert torch.allclose(param_right, initial_right)

    orphan = simple_parameter(require_grad=True)
    orphan.grad = torch.ones_like(orphan)
    no_pair_optimizer = LoRARite([orphan])
    no_pair_optimizer.step()
    assert torch.allclose(orphan, torch.zeros_like(orphan))

    paired_left, paired_right = _paired_parameters()
    paired_right.grad = None
    missing_grad_optimizer = LoRARite([paired_left, paired_right])
    missing_grad_optimizer.step()
    assert torch.allclose(paired_left, torch.tensor([[1.0, 0.2, -0.3], [0.8, 0.5, -0.7]]))


def test_lora_rite_sparse_and_complex_gradients():
    sparse_param = simple_sparse_parameter()[1]
    sparse_optimizer = LoRARite([sparse_param])
    with pytest.raises(NoSparseGradientError):
        sparse_optimizer.step()

    complex_param = simple_complex_parameter()
    complex_optimizer = LoRARite([complex_param])
    with pytest.raises(NoComplexParameterError):
        complex_optimizer.step()


@pytest.mark.parametrize(
    ('kwargs', 'error'),
    [
        ({'lr': -1e-3}, NegativeLRError),
        ({'betas': (-0.1, 0.999)}, ValueError),
        ({'betas': (0.9, 1.0)}, ValueError),
        ({'eps': -1e-6}, ValueError),
        ({'clip_unmagnified_grad': -1.0}, ValueError),
        ({'update_capping': -1.0}, ValueError),
        ({'update_skipping': -1.0}, ValueError),
        ({'weight_decay': -1e-3}, ValueError),
        ({'relative_epsilon': 1}, ValueError),
        ({'apply_escape': 1}, ValueError),
        ({'maybe_inf_to_nan': 1}, ValueError),
        ({'balance_param': 1}, ValueError),
        ({'maximize': 1}, ValueError),
        ({'lora_l_dim': 0.0}, ValueError),
        ({'lora_r_dim': True}, ValueError),
    ],
)
def test_lora_rite_invalid_parameters(kwargs, error):
    with pytest.raises(error):
        LoRARite(None, **kwargs)


def test_load_lora_rite():
    assert load_optimizer('lorarite') is LoRARite
