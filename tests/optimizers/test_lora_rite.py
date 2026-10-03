import torch

from pytorch_optimizer.optimizer.lora_rite import LoRARiteHelper
from tests.fixtures import make_lora_parameters, make_parameter
from tests.utils import build_optimizer


class TestLoraRite:
    def test_lora_rite_helper_methods(self):
        helper = LoRARiteHelper()

        tensor = torch.arange(6.0).reshape(2, 3)
        moved, shape = helper.move_lora_dim_to_last(tensor, 0)

        assert torch.equal(helper.restore_original_shape_and_dim(moved, 0, shape), tensor)
        assert helper.move_lora_dim_to_last(torch.tensor(1.0), 0)[0].shape == (1, 1)
        assert torch.isnan(helper.inf_to_nan(torch.tensor(float('inf'))))
        assert torch.isinf(LoRARiteHelper(maybe_inf_to_nan=False).inf_to_nan(torch.tensor(float('inf'))))
        assert helper.bias_corrected_decay(0, 0.9) == 0.0

        symmetric = helper.make_symmetric(torch.tensor([[1.0, 2.0], [0.0, 3.0]]))
        assert torch.allclose(symmetric, torch.tensor([[1.0, 1.0], [1.0, 3.0]]))

        preconditioner = helper.create_preconditioner(torch.zeros(3, 2))
        assert preconditioner.shape == (2, 2)

        inverse_root = helper.inverse_sqrt(
            torch.eye(2), torch.tensor(0.0), eps=1e-6, eps_root=0.0, relative_epsilon=False
        )
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

    def test_lora_rite_rich_options_and_existing_state(self):
        param_left, param_right = make_lora_parameters()

        optimizer = build_optimizer(
            'lorarite',
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

        assert torch.linalg.norm(param_left).sub(torch.linalg.norm(param_right)).abs() < 1e-4

    def test_lora_rite_skips_large_updates_and_missing_pair(self):
        param_left, param_right = make_lora_parameters()
        initial_left = param_left.detach().clone()
        initial_right = param_right.detach().clone()
        optimizer = build_optimizer(
            'lorarite', [param_left, param_right], lr=1e-1, betas=(0.0, 0.0), update_skipping=1e-12
        )
        optimizer.step()

        assert torch.allclose(param_left, initial_left)
        assert torch.allclose(param_right, initial_right)

        orphan = make_parameter(requires_grad=True)
        orphan.grad = torch.ones_like(orphan)
        no_pair_optimizer = build_optimizer('lorarite', [orphan])
        no_pair_optimizer.step()
        assert torch.allclose(orphan, torch.zeros_like(orphan))

        paired_left, paired_right = make_lora_parameters()
        paired_right.grad = None
        missing_grad_optimizer = build_optimizer('lorarite', [paired_left, paired_right])
        missing_grad_optimizer.step()
        assert torch.allclose(paired_left, torch.tensor([[1.0, 0.2, -0.3], [0.8, 0.5, -0.7]]))
