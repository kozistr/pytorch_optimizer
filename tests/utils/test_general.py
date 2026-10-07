import numpy as np
import pytest
import torch
from torch import nn

from pytorch_optimizer.optimizer import get_optimizer_parameters, load_optimizer
from pytorch_optimizer.optimizer.utils import (
    CPUOffloadOptimizer,
    StochasticAccumulator,
    clip_grad_norm,
    compare_versions,
    copy_stochastic,
    disable_running_stats,
    enable_running_stats,
    get_global_gradient_norm,
    has_overflow,
    is_valid_parameters,
    normalize_gradient,
    parse_pytorch_version,
    reg_noise,
    unit_norm,
)
from tests.fixtures import TrainingModel, make_parameter


class TestVersionUtils:
    def test_parse_version_invalid(self):
        with pytest.raises(ValueError):
            parse_pytorch_version('a.s.d.f')

    def test_parse_version(self):
        assert parse_pytorch_version('2.14.0+cpu') == [2, 14, 0]

    def test_compare_versions(self):
        assert compare_versions('2.9.1', '2.4.0') >= 0


class TestOverflowUtils:
    def test_has_overflow(self):
        assert has_overflow(torch.tensor(torch.inf))
        assert has_overflow(torch.tensor(-torch.inf))
        assert has_overflow(torch.tensor(torch.nan))
        assert not has_overflow(torch.Tensor([1]))


class TestGradientUtils:
    def test_normalize_singleton_channels(self):
        gradient = torch.tensor([[1.0], [2.0], [3.0]])
        expected = gradient.clone()

        normalize_gradient(gradient, use_channels=True)

        torch.testing.assert_close(gradient, expected)

    @pytest.mark.parametrize('use_channels', [False, True])
    def test_normalized_gradient(self, use_channels):
        gradient = torch.arange(10.0).view(1, 10)
        expected = gradient / gradient.std()

        normalize_gradient(gradient, use_channels=use_channels)

        torch.testing.assert_close(gradient, expected)

    def test_clip_grad_norm(self):
        x = torch.arange(0, 10, dtype=torch.float32, requires_grad=True)
        x.grad = torch.arange(0, 10, dtype=torch.float32)

        np.testing.assert_approx_equal(clip_grad_norm(x), 16.88194, significant=6)
        np.testing.assert_approx_equal(clip_grad_norm(x, max_norm=2), 16.88194, significant=6)

        with pytest.raises(ValueError):
            clip_grad_norm(None)

    def test_get_global_gradient_norm(self):
        np.testing.assert_approx_equal(get_global_gradient_norm(None, torch.device('cpu')).item(), 0.0)

        parameter = nn.Parameter(torch.full((4,), 2.0, dtype=torch.float16))
        parameter.grad = torch.full_like(parameter, 32768.0)
        groups = [{'params': [parameter], 'adaptive': True}]

        torch.testing.assert_close(get_global_gradient_norm(groups), torch.tensor([2.0**32]))
        torch.testing.assert_close(
            get_global_gradient_norm(groups, weight_adaptive=True), torch.tensor([2.0**34])
        )


class TestNormUtils:
    @pytest.mark.parametrize('shape', [(10,), (1, 10), (1, 10, 1, 1), (1, 10, 1, 1, 1, 1)])
    def test_unit_norm(self, shape):
        x = torch.arange(10.0).view(shape)
        torch.testing.assert_close(unit_norm(x).squeeze(), torch.tensor(285.0**0.5))


class TestParameterUtils:
    @pytest.mark.parametrize('use_model', [False, True])
    def test_get_optimizer_parameters(self, use_model):
        model = TrainingModel()
        parameters = model if use_model else list(model.named_parameters())
        groups = get_optimizer_parameters(parameters, weight_decay=1e-3)

        assert is_valid_parameters(groups)
        assert [group['weight_decay'] for group in groups] == [1e-3, 0.0]
        assert [[id(param) for param in group['params']] for group in groups] == [
            [id(model.fc1.weight), id(model.fc2.weight)], [id(model.fc1.bias), id(model.fc2.bias)]
        ]


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

        params = [make_parameter(grad=None, device='cuda')]

        opt = CPUOffloadOptimizer(params, load_optimizer('adamw'), fused=False, offload_gradients=True)

        with pytest.raises(ValueError):
            CPUOffloadOptimizer([], load_optimizer('adamw'))

        opt.zero_grad()

        _ = opt.param_groups

        state_dict = opt.state_dict()
        opt.load_state_dict(state_dict)

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
        model = TrainingModel(dtype=torch.bfloat16)
        StochasticAccumulator.assign_hooks(model)
        for _ in range(2):
            sum(param.sum() for param in model.parameters()).backward()

        StochasticAccumulator.reassign_grad_buffer(model)

        for param in model.parameters():
            torch.testing.assert_close(param.grad, torch.full_like(param, 2.0))
            assert not hasattr(param, 'acc_grad')
