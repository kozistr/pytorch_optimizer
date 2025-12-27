import pytest
import torch

from pytorch_optimizer.optimizer.foreach_utils import (
    foreach_add_,
    foreach_addcdiv_,
    foreach_addcmul_,
    foreach_clamp_min_,
    foreach_copy_,
    foreach_div_,
    foreach_lerp_,
    foreach_maximum_,
    foreach_mul_,
    foreach_neg_,
    foreach_sign_,
    foreach_sqrt,
    foreach_sqrt_,
    foreach_sub_,
    foreach_zero_,
    group_tensors_by_device_and_dtype,
    has_foreach_support,
)


class TestHasForeachSupport:
    def test_empty_list(self):
        assert not has_foreach_support([])

    def test_cpu_tensors(self):
        tensors = [torch.randn(1), torch.randn(1)]
        assert not has_foreach_support(tensors)

    def test_different_devices(self):
        tensors = [torch.randn(1, device='meta'), torch.randn(1, device='cpu')]
        assert not has_foreach_support(tensors)

    def test_different_dtypes(self):
        tensors = [torch.randn(1, dtype=torch.float32), torch.randn(1, dtype=torch.float16)]
        assert not has_foreach_support(tensors)

    def test_sparse_tensors(self):
        sparse_tensor = torch.sparse_coo_tensor([[0, 1]], [1.0, 2.0], (3,))
        tensors = [torch.randn(3), sparse_tensor]
        assert not has_foreach_support(tensors)


class TestGroupTensorsByDeviceAndDtype:
    def test_single_group(self):
        params = [torch.randn(1), torch.randn(2)]
        grads = [torch.randn(1), torch.randn(2)]

        groups = group_tensors_by_device_and_dtype(params, grads)

        assert len(groups) == 1
        assert len(groups[0]['params']) == 2
        assert len(groups[0]['grads']) == 2
        assert groups[0]['indices'] == [0, 1]

    def test_multiple_groups_by_dtype(self):
        params = [torch.randn(1, dtype=torch.float32), torch.randn(2, dtype=torch.float16)]
        grads = [torch.randn(1, dtype=torch.float32), torch.randn(2, dtype=torch.float16)]

        groups = group_tensors_by_device_and_dtype(params, grads)

        assert len(groups) == 2

    def test_with_state_lists(self):
        params = [torch.randn(1), torch.randn(2)]
        grads = [torch.randn(1), torch.randn(2)]
        state_lists = {'exp_avg': [torch.randn(1), torch.randn(2)], 'exp_avg_sq': [torch.randn(1), torch.randn(2)]}

        groups = group_tensors_by_device_and_dtype(params, grads, state_lists)

        assert len(groups) == 1
        assert 'exp_avg' in groups[0]
        assert 'exp_avg_sq' in groups[0]
        assert len(groups[0]['exp_avg']) == 2
        assert len(groups[0]['exp_avg_sq']) == 2

    def test_empty_state_lists(self):
        params = [torch.randn(1)]
        grads = [torch.randn(1)]

        groups = group_tensors_by_device_and_dtype(params, grads, None)

        assert len(groups) == 1


@pytest.mark.parametrize('foreach', [False, True])
class TestForeachAdd:
    def test_add_scalar(self, foreach: bool):
        tensors = [torch.ones(1), torch.ones(2)]
        foreach_add_(tensors, 2.0, alpha=0.5, foreach=foreach)

        torch.testing.assert_close(tensors[0], torch.full((1,), 2.0))
        torch.testing.assert_close(tensors[1], torch.full((2,), 2.0))

    def test_add_tensor_as_other(self, foreach: bool):
        tensors = [torch.ones(1), torch.ones(2)]
        other = torch.tensor(2.0)
        foreach_add_(tensors, other, foreach=foreach)

        torch.testing.assert_close(tensors[0], torch.full((1,), 3.0))
        torch.testing.assert_close(tensors[1], torch.full((2,), 3.0))

    def test_add_tensor_list(self, foreach: bool):
        tensors = [torch.ones(1), torch.ones(2)]
        other = [torch.full((1,), 2.0), torch.full((2,), 4.0)]
        foreach_add_(tensors, other, alpha=0.5, foreach=foreach)

        torch.testing.assert_close(tensors[0], torch.full((1,), 2.0))
        torch.testing.assert_close(tensors[1], torch.full((2,), 3.0))


@pytest.mark.parametrize('foreach', [False, True])
class TestForeachMul:
    def test_mul_scalar(self, foreach: bool):
        tensors = [torch.full((1,), 2.0), torch.full((2,), 3.0)]
        foreach_mul_(tensors, 2.0, foreach=foreach)

        torch.testing.assert_close(tensors[0], torch.full((1,), 4.0))
        torch.testing.assert_close(tensors[1], torch.full((2,), 6.0))

    def test_mul_tensor_list(self, foreach: bool):
        tensors = [torch.full((1,), 2.0), torch.full((2,), 3.0)]
        scalars = [torch.full((1,), 2.0), torch.full((2,), 0.5)]
        foreach_mul_(tensors, scalars, foreach=foreach)

        torch.testing.assert_close(tensors[0], torch.full((1,), 4.0))
        torch.testing.assert_close(tensors[1], torch.full((2,), 1.5))


@pytest.mark.parametrize('foreach', [False, True])
class TestForeachLerp:
    def test_lerp(self, foreach: bool):
        tensors = [torch.zeros(1), torch.zeros(2)]
        other = [torch.ones(1), torch.ones(2)]
        foreach_lerp_(tensors, other, weight=0.5, foreach=foreach)

        torch.testing.assert_close(tensors[0], torch.full((1,), 0.5))
        torch.testing.assert_close(tensors[1], torch.full((2,), 0.5))


@pytest.mark.parametrize('foreach', [False, True])
class TestForeachAddCMul:
    def test_addcmul(self, foreach: bool):
        tensors = [torch.zeros(1), torch.zeros(2)]
        tensor1 = [torch.full((1,), 2.0), torch.full((2,), 3.0)]
        tensor2 = [torch.full((1,), 3.0), torch.full((2,), 2.0)]
        foreach_addcmul_(tensors, tensor1, tensor2, value=0.5, foreach=foreach)

        torch.testing.assert_close(tensors[0], torch.full((1,), 3.0))
        torch.testing.assert_close(tensors[1], torch.full((2,), 3.0))


@pytest.mark.parametrize('foreach', [False, True])
class TestForeachAddCDiv:
    def test_addcdiv_with_value_fallback(self, foreach: bool):
        tensors = [torch.zeros(1), torch.zeros(2)]
        tensor1 = [torch.full((1,), 6.0), torch.full((2,), 8.0)]
        tensor2 = [torch.full((1,), 2.0), torch.full((2,), 2.0)]
        foreach_addcdiv_(tensors, tensor1, tensor2, value=0.5, foreach=foreach)

        torch.testing.assert_close(tensors[0], torch.full((1,), 1.5))
        torch.testing.assert_close(tensors[1], torch.full((2,), 2.0))


@pytest.mark.parametrize('foreach', [False, True])
class TestForeachSqrt:
    def test_sqrt(self, foreach: bool):
        tensors = [torch.full((1,), 4.0), torch.full((2,), 9.0)]
        result = foreach_sqrt(tensors, foreach=foreach)

        torch.testing.assert_close(result[0], torch.full((1,), 2.0))
        torch.testing.assert_close(result[1], torch.full((2,), 3.0))

    def test_sqrt_inplace(self, foreach: bool):
        tensors = [torch.full((1,), 4.0), torch.full((2,), 9.0)]
        foreach_sqrt_(tensors, foreach=foreach)

        torch.testing.assert_close(tensors[0], torch.full((1,), 2.0))
        torch.testing.assert_close(tensors[1], torch.full((2,), 3.0))


@pytest.mark.parametrize('foreach', [False, True])
class TestForeachMaximum:
    def test_maximum(self, foreach: bool):
        tensors = [torch.tensor([1.0, 3.0, 2.0]), torch.tensor([4.0, 1.0])]
        other = [torch.tensor([2.0, 2.0, 2.0]), torch.tensor([3.0, 3.0])]
        foreach_maximum_(tensors, other, foreach=foreach)

        torch.testing.assert_close(tensors[0], torch.tensor([2.0, 3.0, 2.0]))
        torch.testing.assert_close(tensors[1], torch.tensor([4.0, 3.0]))


@pytest.mark.parametrize('foreach', [False, True])
class TestForeachSign:
    def test_sign_fallback(self, foreach: bool):
        tensors = [torch.tensor([-2.0, 0.0, 3.0]), torch.tensor([4.0, -1.0])]
        foreach_sign_(tensors, foreach=foreach)

        torch.testing.assert_close(tensors[0], torch.tensor([-1.0, 0.0, 1.0]))
        torch.testing.assert_close(tensors[1], torch.tensor([1.0, -1.0]))


@pytest.mark.parametrize('foreach', [False, True])
class TestForeachNeg:
    def test_neg(self, foreach: bool):
        tensors = [torch.tensor([1.0, -2.0, 3.0]), torch.tensor([-4.0, 5.0])]
        foreach_neg_(tensors, foreach=foreach)

        torch.testing.assert_close(tensors[0], torch.tensor([-1.0, 2.0, -3.0]))
        torch.testing.assert_close(tensors[1], torch.tensor([4.0, -5.0]))


@pytest.mark.parametrize('foreach', [False, True])
class TestForeachSub:
    def test_sub_scalar(self, foreach: bool):
        tensors = [torch.full((1,), 5.0), torch.full((2,), 3.0)]
        foreach_sub_(tensors, 2.0, foreach=foreach)

        torch.testing.assert_close(tensors[0], torch.full((1,), 3.0))
        torch.testing.assert_close(tensors[1], torch.full((2,), 1.0))

    def test_sub_tensor_as_other(self, foreach: bool):
        tensors = [torch.full((1,), 5.0), torch.full((2,), 3.0)]
        other = torch.tensor(2.0)
        foreach_sub_(tensors, other, foreach=foreach)

        torch.testing.assert_close(tensors[0], torch.full((1,), 3.0))
        torch.testing.assert_close(tensors[1], torch.full((2,), 1.0))

    def test_sub_tensor_list(self, foreach: bool):
        tensors = [torch.full((1,), 5.0), torch.full((2,), 6.0)]
        other = [torch.full((1,), 2.0), torch.full((2,), 4.0)]
        foreach_sub_(tensors, other, alpha=0.5, foreach=foreach)

        torch.testing.assert_close(tensors[0], torch.full((1,), 4.0))
        torch.testing.assert_close(tensors[1], torch.full((2,), 4.0))


@pytest.mark.parametrize('foreach', [False, True])
class TestForeachDiv:
    def test_div_scalar(self, foreach: bool):
        tensors = [torch.full((1,), 6.0), torch.full((2,), 8.0)]
        foreach_div_(tensors, 2.0, foreach=foreach)

        torch.testing.assert_close(tensors[0], torch.full((1,), 3.0))
        torch.testing.assert_close(tensors[1], torch.full((2,), 4.0))

    def test_div_tensor(self, foreach: bool):
        tensors = [torch.full((1,), 6.0), torch.full((2,), 9.0)]
        other = [torch.full((1,), 2.0), torch.full((2,), 3.0)]
        foreach_div_(tensors, other, foreach=foreach)

        torch.testing.assert_close(tensors[0], torch.full((1,), 3.0))
        torch.testing.assert_close(tensors[1], torch.full((2,), 3.0))


@pytest.mark.parametrize('foreach', [False, True])
class TestForeachZero:
    def test_zero_fallback(self, foreach: bool):
        tensors = [torch.full((1,), 5.0), torch.full((2,), 3.0)]
        foreach_zero_(tensors, foreach=foreach)

        torch.testing.assert_close(tensors[0], torch.zeros(1))
        torch.testing.assert_close(tensors[1], torch.zeros(2))


@pytest.mark.parametrize('foreach', [False, True])
class TestForeachCopy:
    def test_copy(self, foreach: bool):
        tensors = [torch.zeros(1), torch.zeros(2)]
        src = [torch.full((1,), 2.0), torch.full((2,), 3.0)]
        foreach_copy_(tensors, src, foreach=foreach)

        torch.testing.assert_close(tensors[0], torch.full((1,), 2.0))
        torch.testing.assert_close(tensors[1], torch.full((2,), 3.0))


@pytest.mark.parametrize('foreach', [False, True])
class TestForeachClampMin:
    def test_clamp_min(self, foreach: bool):
        tensors = [torch.tensor([-1.0, 0.5, 2.0]), torch.tensor([-2.0, 3.0])]
        foreach_clamp_min_(tensors, 0.0, foreach=foreach)

        torch.testing.assert_close(tensors[0], torch.tensor([0.0, 0.5, 2.0]))
        torch.testing.assert_close(tensors[1], torch.tensor([0.0, 3.0]))
