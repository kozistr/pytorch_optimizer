from typing import List

import numpy as np
import pytest
import torch
from torch import nn

from pytorch_optimizer.optimizer.shampoo_utils import (
    BlockPartitioner,
    PreConditioner,
    compute_power_schur_newton,
    merge_small_dims,
)
from pytorch_optimizer.optimizer.utils import (
    CPUOffloadOptimizer,
    clip_grad_norm,
    compare_versions,
    disable_running_stats,
    enable_running_stats,
    get_optimizer_parameters,
    has_overflow,
    is_valid_parameters,
    neuron_mean,
    neuron_norm,
    normalize_gradient,
    parse_pytorch_version,
    reduce_max_except_dim,
    reg_noise,
    to_real,
    unit_norm,
)
from tests.utils import Example


def test_has_overflow():
    assert has_overflow(torch.tensor(torch.inf))
    assert has_overflow(torch.tensor(-torch.inf))
    assert has_overflow(torch.tensor(torch.nan))
    assert not has_overflow(torch.Tensor([1]))


def test_normalized_gradient():
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


def test_clip_grad_norm():
    x = torch.arange(0, 10, dtype=torch.float32, requires_grad=True)
    x.grad = torch.arange(0, 10, dtype=torch.float32)

    np.testing.assert_approx_equal(clip_grad_norm(x), 16.88194, significant=6)
    np.testing.assert_approx_equal(clip_grad_norm(x, max_norm=2), 16.88194, significant=6)


def test_unit_norm():
    x = torch.arange(0, 10, dtype=torch.float32)

    np.testing.assert_approx_equal(unit_norm(x).numpy(), 16.8819, significant=5)
    np.testing.assert_approx_equal(unit_norm(x.view(1, 10)).numpy().reshape(-1)[0], 16.8819, significant=5)
    np.testing.assert_approx_equal(unit_norm(x.view(1, 10, 1, 1)).numpy().reshape(-1)[0], 16.8819, significant=5)
    np.testing.assert_approx_equal(unit_norm(x.view(1, 10, 1, 1, 1, 1)).numpy().reshape(-1)[0], 16.8819, significant=5)


def test_neuron_mean_norm():
    x = torch.arange(-5, 5, dtype=torch.float32)

    with pytest.raises(ValueError) as error_info:
        neuron_mean(x)

    assert str(error_info.value) == '[-] neuron_mean not defined on 1D tensors.'

    np.testing.assert_array_equal(
        neuron_mean(x.view(-1, 1)).numpy(),
        np.asarray([[-5.0], [-4.0], [-3.0], [-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0], [4.0]]),
    )
    np.testing.assert_array_equal(
        neuron_norm(x).numpy(), np.asarray([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    )
    np.testing.assert_array_equal(
        neuron_norm(x.view(-1, 1)).numpy(),
        np.asarray([[5.0], [4.0], [3.0], [2.0], [1.0], [0.0], [1.0], [2.0], [3.0], [4.0]]),
    )


def test_get_optimizer_parameters():
    model: nn.Module = Example()
    wd_ban_list: List[str] = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'LayerNorm']

    before_parameters = list(model.named_parameters())

    _ = get_optimizer_parameters(before_parameters, weight_decay=1e-3, wd_ban_list=wd_ban_list)
    after_parameters = get_optimizer_parameters(model, weight_decay=1e-3, wd_ban_list=wd_ban_list)

    for before, after in zip(before_parameters, after_parameters):
        layer_name: str = before[0]
        if layer_name.find('bias') != -1 or layer_name.find('LayerNorm') != -1:
            assert after['weight_decay'] == 0.0


def test_is_valid_parameters():
    model: nn.Module = Example()
    wd_ban_list: List[str] = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    after_parameters = get_optimizer_parameters(model, weight_decay=1e-3, wd_ban_list=wd_ban_list)

    assert is_valid_parameters(after_parameters)


def test_running_stats():
    model = nn.Sequential(
        nn.Linear(1, 1),
        nn.BatchNorm2d(1),
    )
    model[1].momentum = 0.1

    disable_running_stats(model)

    assert model[1].momentum == 0
    assert model[1].backup_momentum == 0.1

    enable_running_stats(model)

    assert model[1].momentum == 0.1


def test_compute_power():
    x = compute_power_schur_newton(torch.zeros((1,)), p=1)
    assert torch.tensor([1000000.0]) == x

    x = compute_power_schur_newton(torch.zeros((1, 2)), p=1)
    assert torch.tensor([1.0]) == x

    _ = compute_power_schur_newton(torch.ones((2, 2)), p=3)

    x = compute_power_schur_newton(torch.ones((2, 2)), p=1)
    assert np.sum(x.numpy() - np.asarray([[252206.4062, -252205.8750], [-252205.8750, 252206.4062]])) < 200

    _ = compute_power_schur_newton(torch.ones((2, 2)), p=8)

    _ = compute_power_schur_newton(torch.ones((2, 2)), p=16)

    x = compute_power_schur_newton(torch.ones((2, 2)), p=16, max_error_ratio=0.0)
    np.testing.assert_array_almost_equal(
        np.asarray([[1.0946, 0.0000], [0.0000, 1.0946]]),
        x.numpy(),
        decimal=2,
    )

    x = compute_power_schur_newton(torch.ones((2, 2)), p=2)
    assert np.sum(x.numpy() - np.asarray([[359.1108, -358.4036], [-358.4036, 359.1108]])) < 50


def test_merge_small_dims():
    case1 = [1, 2, 512, 1, 2048, 1, 3, 4]
    expected_case1 = [1024, 2048, 12]
    assert expected_case1 == merge_small_dims(case1, max_dim=1024)

    case2 = [1, 2, 768, 1, 2048]
    expected_case2 = [2, 768, 2048]
    assert expected_case2 == merge_small_dims(case2, max_dim=1024)

    case3 = [1, 1, 1]
    expected_case3 = [1]
    assert expected_case3 == merge_small_dims(case3, max_dim=1)


def test_to_real():
    complex_tensor = torch.tensor(1.0j + 2.0, dtype=torch.complex64)
    assert to_real(complex_tensor) == 2.0

    real_tensor = torch.tensor(1.0, dtype=torch.float32)
    assert to_real(real_tensor) == 1.0


def test_block_partitioner():
    var = torch.zeros((2, 2))
    target_var = torch.zeros((1, 1))

    partitioner = BlockPartitioner(var, block_size=2, rank=2, pre_conditioner_type=0)
    with pytest.raises(ValueError):
        partitioner.partition(target_var)


def test_pre_conditioner():
    var = torch.zeros((1024, 128))
    grad = torch.zeros((1024, 128))

    pre_conditioner = PreConditioner(var, 0.9, 0, 128, 1, 8192, True, 0)
    pre_conditioner.add_statistics(grad)
    pre_conditioner.compute_pre_conditioners()


@pytest.mark.parametrize('pre_conditioner_type', [0, 1, 2, 3])
def test_pre_conditioner_type(pre_conditioner_type):
    var = torch.zeros((4, 4, 32))
    if pre_conditioner_type in (0, 1, 2):
        PreConditioner(var, 0.9, 0, 128, 1, 8192, True, pre_conditioner_type=pre_conditioner_type)
    else:
        with pytest.raises(ValueError):
            PreConditioner(var, 0.9, 0, 128, 1, 8192, True, pre_conditioner_type=pre_conditioner_type)


def test_max_reduce_except_dim():
    x = torch.tensor(1.0)
    assert reduce_max_except_dim(x, 0) == x

    x = torch.zeros((1, 1))
    with pytest.raises(ValueError):
        reduce_max_except_dim(x, 3)


def test_emcmc():
    torch.random.manual_seed(42)

    network1 = Example()
    network2 = Example()

    loss = reg_noise(network1, network2, int(5e4), 1e-1).detach().numpy()
    np.testing.assert_almost_equal(loss, 0.0011383)


def test_version_utils():
    with pytest.raises(ValueError):
        parse_pytorch_version('a.s.d.f')

    assert parse_pytorch_version(torch.__version__) == [2, 5, 0]

    assert compare_versions('2.5.0', '2.4.0') >= 0


def test_cpu_offload_optimizer():
    params = Example().parameters()

    opt = CPUOffloadOptimizer(params, torch.optim.AdamW, fused=False, offload_gradients=True)

    with pytest.raises(ValueError):
        CPUOffloadOptimizer([], torch.optim.AdamW)

    opt.zero_grad()

    _ = opt.param_groups

    state_dict = opt.state_dict()
    opt.load_state_dict(state_dict)
