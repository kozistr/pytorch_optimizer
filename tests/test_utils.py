import numpy as np
import pytest
import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits

from pytorch_optimizer.optimizer import get_optimizer_parameters, load_optimizer
from pytorch_optimizer.optimizer.sam import get_global_gradient_norm
from pytorch_optimizer.optimizer.utils import (
    CPUOffloadOptimizer,
    StochasticAccumulator,
    clip_grad_norm,
    compare_versions,
    copy_stochastic,
    disable_running_stats,
    enable_running_stats,
    has_overflow,
    is_valid_parameters,
    normalize_gradient,
    parse_pytorch_version,
    reg_noise,
    unit_norm,
)
from tests.fixtures import TrainingModel, make_parameter
from tests.utils import build_optimizer


class TestVersionUtils:
    def test_parse_version_invalid(self):
        with pytest.raises(ValueError):
            parse_pytorch_version('a.s.d.f')

    def test_parse_version(self):
        pytorch_version: list[int] = parse_pytorch_version(torch.__version__)

        assert len(pytorch_version) == 3
        assert pytorch_version == [2, 14, 0]

    def test_compare_versions(self):
        assert compare_versions('2.9.1', '2.4.0') >= 0


class TestOverflowUtils:
    def test_has_overflow(self):
        assert has_overflow(torch.tensor(torch.inf))
        assert has_overflow(torch.tensor(-torch.inf))
        assert has_overflow(torch.tensor(torch.nan))
        assert not has_overflow(torch.Tensor([1]))


class TestGradientUtils:
    def test_normalized_gradient(self):
        x = torch.arange(0, 10, dtype=torch.float32)
        normalize_gradient(x)

        np.testing.assert_allclose(
            x.numpy(),
            np.asarray([0.0000, 0.3303, 0.6606, 0.9909, 1.3212, 1.6514, 1.9817, 2.3120, 2.6423, 2.9726]),
            rtol=1e-4,
            atol=1e-4,
        )

        x = torch.arange(0, 10, dtype=torch.float32)
        normalize_gradient(x.view(1, 10), use_channels=True)

        np.testing.assert_allclose(
            x.numpy(),
            np.asarray([0.0000, 0.3303, 0.6606, 0.9909, 1.3212, 1.6514, 1.9817, 2.3120, 2.6423, 2.9726]),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_clip_grad_norm(self):
        x = torch.arange(0, 10, dtype=torch.float32, requires_grad=True)
        x.grad = torch.arange(0, 10, dtype=torch.float32)

        np.testing.assert_approx_equal(clip_grad_norm(x), 16.88194, significant=6)
        np.testing.assert_approx_equal(clip_grad_norm(x, max_norm=2), 16.88194, significant=6)

        with pytest.raises(ValueError):
            clip_grad_norm(None)

    def test_get_global_gradient_norm(self):
        np.testing.assert_approx_equal(get_global_gradient_norm(None, torch.device('cpu')).item(), 0.0)


class TestNormUtils:
    def test_unit_norm(self):
        x = torch.arange(0, 10, dtype=torch.float32)

        np.testing.assert_approx_equal(unit_norm(x).numpy(), 16.8819, significant=5)
        np.testing.assert_approx_equal(unit_norm(x.view(1, 10)).numpy().reshape(-1)[0], 16.8819, significant=5)
        np.testing.assert_approx_equal(unit_norm(x.view(1, 10, 1, 1)).numpy().reshape(-1)[0], 16.8819, significant=5)
        np.testing.assert_approx_equal(
            unit_norm(x.view(1, 10, 1, 1, 1, 1)).numpy().reshape(-1)[0], 16.8819, significant=5
        )


class TestParameterUtils:
    def test_get_optimizer_parameters(self):
        model: nn.Module = TrainingModel()
        wd_ban_list: list[str] = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'LayerNorm']

        before_parameters = list(model.named_parameters())

        _ = get_optimizer_parameters(before_parameters, weight_decay=1e-3, wd_ban_list=wd_ban_list)
        after_parameters = get_optimizer_parameters(model, weight_decay=1e-3, wd_ban_list=wd_ban_list)

        for before, after in zip(before_parameters, after_parameters):
            layer_name: str = before[0]
            if layer_name.find('bias') != -1 or layer_name.find('LayerNorm') != -1:
                assert after['weight_decay'] == 0.0

    def test_is_valid_parameters(self):
        model: nn.Module = TrainingModel()
        wd_ban_list: list[str] = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        after_parameters = get_optimizer_parameters(model, weight_decay=1e-3, wd_ban_list=wd_ban_list)

        assert is_valid_parameters(after_parameters)


class TestRunningStats:
    def test_running_stats(self):
        model = nn.BatchNorm2d(1, momentum=0.1)

        disable_running_stats(model)

        assert model.momentum == 0
        assert model.backup_momentum == 0.1

        enable_running_stats(model)

        assert model.momentum == 0.1


class TestMiscUtils:
    def test_emcmc(self):
        torch.random.manual_seed(42)

        network1 = TrainingModel()
        network2 = TrainingModel()

        loss = reg_noise(network1, network2, int(5e4), 1e-1).detach().numpy()
        np.testing.assert_almost_equal(loss, 0.0017181)

    def test_cpu_offload_optimizer(self):
        if not torch.cuda.is_available():
            pytest.skip('need GPU to run a test')

        params = [make_parameter(grad=None)]

        opt = CPUOffloadOptimizer(params, load_optimizer('adamw'), fused=False, offload_gradients=True)

        with pytest.raises(ValueError):
            CPUOffloadOptimizer([], load_optimizer('adamw'))

        opt.zero_grad()

        _ = opt.param_groups

        state_dict = opt.state_dict()
        opt.load_state_dict(state_dict)

    def test_orthograd_name(self):
        optimizer = build_optimizer('orthograd', [make_parameter()])
        optimizer.zero_grad()

        _ = optimizer.param_groups
        _ = optimizer.state

        assert str(optimizer).lower() == 'orthograd'

    def test_copy_stochastic(self):
        n: int = 512

        a = torch.full((n,), 1.0, dtype=torch.bfloat16)
        b = torch.full((n,), 0.0002, dtype=torch.bfloat16)
        result = torch.full((n,), 0.0, dtype=torch.bfloat16)

        added = a.to(dtype=torch.float32) + b

        result.copy_(added)
        np.testing.assert_almost_equal(1.0000, result.to(dtype=torch.float32).mean().item(), decimal=4)

        copy_stochastic(result, added)
        np.testing.assert_almost_equal(1.0002, result.to(dtype=torch.float32).mean().item(), decimal=4)

    def test_stochastic_accumulation_hook(self):
        model = TrainingModel().bfloat16()
        x = torch.randn(1, 2, dtype=torch.bfloat16)

        StochasticAccumulator.assign_hooks(model)

        optimizer = build_optimizer('orthograd', model.parameters())

        for _ in range(2):
            binary_cross_entropy_with_logits(model(x), x[:, :1]).backward()

        StochasticAccumulator.reassign_grad_buffer(model)

        optimizer.step()
        optimizer.zero_grad()
