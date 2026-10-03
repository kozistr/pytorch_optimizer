import pytest
import torch
from torch import nn

from pytorch_optimizer.optimizer import DynamicLossScaler, SafeFP16Optimizer, load_optimizer
from tests.fixtures import make_parameter
from tests.utils import sphere_loss


class TestLomo:
    @pytest.mark.parametrize('optimizer_name', ['lomo', 'adalomo'])
    def test_lomo_deepspeed_zero3(self, optimizer_name):
        model = nn.Linear(1, 1)

        model.weight.__setattr__('ds_tensor', 0)

        optimizer = load_optimizer(optimizer_name)(model)
        optimizer.init_group({})

        assert str(optimizer).lower() == optimizer_name

    def test_lomo_clip_grad_norm_with_fp16(self):
        model = nn.Linear(1, 1)

        model.weight.data = torch.randn(1, 1, dtype=torch.float16)

        with pytest.raises(ValueError):
            load_optimizer('lomo')(model, clip_grad_norm=None)

    @pytest.mark.parametrize('optimizer_name', ['lomo'])
    def test_lomo_fused_backward(self, optimizer_name):
        optimizer = load_optimizer(optimizer_name)(nn.Linear(1, 1), clip_grad_norm=1.0)
        with pytest.raises(ValueError):
            optimizer.fused_backward(loss=0.1, lr=0.1)

    @pytest.mark.parametrize('optimizer_name', ['lomo', 'adalomo'])
    @pytest.mark.parametrize('precision', [16, 32])
    def test_lomo_optimizer(self, optimizer_name, precision):
        model = nn.Linear(1, 1)

        model.bias.data = torch.randn(1, dtype=torch.float32)
        model.bias.grad = torch.randn(1, dtype=torch.float32)

        if precision == 16:
            model.weight.data = torch.randn(1, 1, dtype=torch.float16)
            model.weight.grad = torch.zeros(1, 1, dtype=torch.float16)

        optimizer = load_optimizer(optimizer_name)(model, clip_grad_norm=1.0, clip_grad_value=1.0)

        if precision == 16:
            optimizer.clip_coef = 0.9

        parameters = iter(model.parameters())

        loss = sphere_loss(next(parameters))
        optimizer.grad_norm(loss)
        optimizer.fused_backward(loss, lr=0.1)

        loss = sphere_loss(next(parameters))
        optimizer.grad_norm(loss)
        optimizer.fused_backward(loss, lr=0.1)

    def test_dynamic_scaler(self):
        scaler = DynamicLossScaler(init_scale=2.0 ** 15, scale_window=1, threshold=1e-2)  # fmt: skip
        scaler.decrease_loss_scale()
        scaler.update_scale(overflow=False)

    def test_safe_fp16_methods(self):
        optimizer = SafeFP16Optimizer(load_optimizer('adamp')([make_parameter()], lr=5e-1))
        optimizer.load_state_dict(optimizer.state_dict())
        optimizer.scaler.decrease_loss_scale()
        optimizer.zero_grad()
        optimizer.update_main_grads()
        optimizer.clip_main_grads(100.0)
        optimizer.multiply_grads(100.0)

        with pytest.raises(AttributeError):
            optimizer.get_lr()

        with pytest.raises(AttributeError):
            optimizer.set_lr(lr=5e-1)

        assert optimizer.loss_scale == 2.0 ** (15 - 1)
