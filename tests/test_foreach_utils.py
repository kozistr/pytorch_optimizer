import copy

import pytest
import torch

from pytorch_optimizer.optimizer.adabelief import AdaBelief
from pytorch_optimizer.optimizer.adamw import StableAdamW
from pytorch_optimizer.optimizer.adan import Adan
from pytorch_optimizer.optimizer.adopt import ADOPT
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
from pytorch_optimizer.optimizer.lamb import Lamb
from pytorch_optimizer.optimizer.lars import LARS
from pytorch_optimizer.optimizer.lion import Lion
from pytorch_optimizer.optimizer.sgd import SGDW, SignSGD
from pytorch_optimizer.optimizer.tiger import Tiger


class TestHasForeachSupport:
    def test_empty_list(self):
        assert not has_foreach_support([])

    def test_cpu_tensors(self):
        tensors = [torch.randn(3), torch.randn(3)]
        assert not has_foreach_support(tensors)

    def test_different_devices(self):
        if not torch.cuda.is_available():
            pytest.skip('need GPU to run this test')
        tensors = [torch.randn(3).cuda(), torch.randn(3)]
        assert not has_foreach_support(tensors)

    def test_different_dtypes(self):
        tensors = [torch.randn(3, dtype=torch.float32), torch.randn(3, dtype=torch.float16)]
        assert not has_foreach_support(tensors)

    def test_sparse_tensors(self):
        sparse_tensor = torch.sparse_coo_tensor([[0, 1]], [1.0, 2.0], (3,))
        tensors = [torch.randn(3), sparse_tensor]
        assert not has_foreach_support(tensors)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_cuda_tensors_supported(self):
        tensors = [torch.randn(3, device='cuda'), torch.randn(3, device='cuda')]
        assert has_foreach_support(tensors)


class TestGroupTensorsByDeviceAndDtype:
    def test_single_group(self):
        params = [torch.randn(3), torch.randn(4)]
        grads = [torch.randn(3), torch.randn(4)]

        groups = group_tensors_by_device_and_dtype(params, grads)

        assert len(groups) == 1
        assert len(groups[0]['params']) == 2
        assert len(groups[0]['grads']) == 2
        assert groups[0]['indices'] == [0, 1]

    def test_multiple_groups_by_dtype(self):
        params = [torch.randn(3, dtype=torch.float32), torch.randn(4, dtype=torch.float16)]
        grads = [torch.randn(3, dtype=torch.float32), torch.randn(4, dtype=torch.float16)]

        groups = group_tensors_by_device_and_dtype(params, grads)

        assert len(groups) == 2

    def test_with_state_lists(self):
        params = [torch.randn(3), torch.randn(4)]
        grads = [torch.randn(3), torch.randn(4)]
        state_lists = {'exp_avg': [torch.randn(3), torch.randn(4)], 'exp_avg_sq': [torch.randn(3), torch.randn(4)]}

        groups = group_tensors_by_device_and_dtype(params, grads, state_lists)

        assert len(groups) == 1
        assert 'exp_avg' in groups[0]
        assert 'exp_avg_sq' in groups[0]
        assert len(groups[0]['exp_avg']) == 2
        assert len(groups[0]['exp_avg_sq']) == 2

    def test_empty_state_lists(self):
        params = [torch.randn(3)]
        grads = [torch.randn(3)]

        groups = group_tensors_by_device_and_dtype(params, grads, None)

        assert len(groups) == 1


class TestForeachAdd:
    def test_add_scalar_fallback(self):
        tensors = [torch.ones(3), torch.ones(4)]
        foreach_add_(tensors, 1.0, foreach=False)
        torch.testing.assert_close(tensors[0], torch.full((3,), 2.0))
        torch.testing.assert_close(tensors[1], torch.full((4,), 2.0))

    def test_add_scalar_with_alpha_fallback(self):
        tensors = [torch.ones(3), torch.ones(4)]
        foreach_add_(tensors, 2.0, alpha=0.5, foreach=False)
        torch.testing.assert_close(tensors[0], torch.full((3,), 2.0))
        torch.testing.assert_close(tensors[1], torch.full((4,), 2.0))

    def test_add_tensor_list_fallback(self):
        tensors = [torch.ones(3), torch.ones(4)]
        other = [torch.full((3,), 2.0), torch.full((4,), 3.0)]
        foreach_add_(tensors, other, alpha=1.0, foreach=False)
        torch.testing.assert_close(tensors[0], torch.full((3,), 3.0))
        torch.testing.assert_close(tensors[1], torch.full((4,), 4.0))

    def test_add_tensor_list_with_alpha_fallback(self):
        tensors = [torch.ones(3), torch.ones(4)]
        other = [torch.full((3,), 2.0), torch.full((4,), 4.0)]
        foreach_add_(tensors, other, alpha=0.5, foreach=False)
        torch.testing.assert_close(tensors[0], torch.full((3,), 2.0))
        torch.testing.assert_close(tensors[1], torch.full((4,), 3.0))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_add_scalar_cuda(self):
        tensors = [torch.ones(3, device='cuda'), torch.ones(4, device='cuda')]
        foreach_add_(tensors, 1.0, foreach=True)
        torch.testing.assert_close(tensors[0], torch.full((3,), 2.0, device='cuda'))
        torch.testing.assert_close(tensors[1], torch.full((4,), 2.0, device='cuda'))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_add_tensor_list_cuda(self):
        tensors = [torch.ones(3, device='cuda'), torch.ones(4, device='cuda')]
        other = [torch.full((3,), 2.0, device='cuda'), torch.full((4,), 3.0, device='cuda')]
        foreach_add_(tensors, other, alpha=1.0, foreach=True)
        torch.testing.assert_close(tensors[0], torch.full((3,), 3.0, device='cuda'))
        torch.testing.assert_close(tensors[1], torch.full((4,), 4.0, device='cuda'))

    def test_add_integer_scalar_fallback(self):
        tensors = [torch.ones(3), torch.ones(4)]
        foreach_add_(tensors, 2, alpha=1.0, foreach=False)
        torch.testing.assert_close(tensors[0], torch.full((3,), 3.0))


class TestForeachMul:
    def test_mul_scalar_fallback(self):
        tensors = [torch.full((3,), 2.0), torch.full((4,), 3.0)]
        foreach_mul_(tensors, 2.0, foreach=False)
        torch.testing.assert_close(tensors[0], torch.full((3,), 4.0))
        torch.testing.assert_close(tensors[1], torch.full((4,), 6.0))

    def test_mul_tensor_list_fallback(self):
        tensors = [torch.full((3,), 2.0), torch.full((4,), 3.0)]
        scalars = [torch.full((3,), 2.0), torch.full((4,), 0.5)]
        foreach_mul_(tensors, scalars, foreach=False)
        torch.testing.assert_close(tensors[0], torch.full((3,), 4.0))
        torch.testing.assert_close(tensors[1], torch.full((4,), 1.5))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_mul_scalar_cuda(self):
        tensors = [torch.full((3,), 2.0, device='cuda'), torch.full((4,), 3.0, device='cuda')]
        foreach_mul_(tensors, 2.0, foreach=True)
        torch.testing.assert_close(tensors[0], torch.full((3,), 4.0, device='cuda'))
        torch.testing.assert_close(tensors[1], torch.full((4,), 6.0, device='cuda'))


class TestForeachLerp:
    def test_lerp_fallback(self):
        tensors = [torch.zeros(3), torch.zeros(4)]
        other = [torch.ones(3), torch.ones(4)]
        foreach_lerp_(tensors, other, weight=0.5, foreach=False)
        torch.testing.assert_close(tensors[0], torch.full((3,), 0.5))
        torch.testing.assert_close(tensors[1], torch.full((4,), 0.5))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_lerp_cuda(self):
        tensors = [torch.zeros(3, device='cuda'), torch.zeros(4, device='cuda')]
        other = [torch.ones(3, device='cuda'), torch.ones(4, device='cuda')]
        foreach_lerp_(tensors, other, weight=0.5, foreach=True)
        torch.testing.assert_close(tensors[0], torch.full((3,), 0.5, device='cuda'))
        torch.testing.assert_close(tensors[1], torch.full((4,), 0.5, device='cuda'))


class TestForeachAddcmul:
    def test_addcmul_fallback(self):
        tensors = [torch.zeros(3), torch.zeros(4)]
        tensor1 = [torch.full((3,), 2.0), torch.full((4,), 3.0)]
        tensor2 = [torch.full((3,), 3.0), torch.full((4,), 2.0)]
        foreach_addcmul_(tensors, tensor1, tensor2, value=1.0, foreach=False)
        torch.testing.assert_close(tensors[0], torch.full((3,), 6.0))
        torch.testing.assert_close(tensors[1], torch.full((4,), 6.0))

    def test_addcmul_with_value_fallback(self):
        tensors = [torch.zeros(3), torch.zeros(4)]
        tensor1 = [torch.full((3,), 2.0), torch.full((4,), 3.0)]
        tensor2 = [torch.full((3,), 3.0), torch.full((4,), 2.0)]
        foreach_addcmul_(tensors, tensor1, tensor2, value=0.5, foreach=False)
        torch.testing.assert_close(tensors[0], torch.full((3,), 3.0))
        torch.testing.assert_close(tensors[1], torch.full((4,), 3.0))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_addcmul_cuda(self):
        tensors = [torch.zeros(3, device='cuda'), torch.zeros(4, device='cuda')]
        tensor1 = [torch.full((3,), 2.0, device='cuda'), torch.full((4,), 3.0, device='cuda')]
        tensor2 = [torch.full((3,), 3.0, device='cuda'), torch.full((4,), 2.0, device='cuda')]
        foreach_addcmul_(tensors, tensor1, tensor2, value=1.0, foreach=True)
        torch.testing.assert_close(tensors[0], torch.full((3,), 6.0, device='cuda'))
        torch.testing.assert_close(tensors[1], torch.full((4,), 6.0, device='cuda'))


class TestForeachAddcdiv:
    def test_addcdiv_fallback(self):
        tensors = [torch.zeros(3), torch.zeros(4)]
        tensor1 = [torch.full((3,), 6.0), torch.full((4,), 8.0)]
        tensor2 = [torch.full((3,), 2.0), torch.full((4,), 2.0)]
        foreach_addcdiv_(tensors, tensor1, tensor2, value=1.0, foreach=False)
        torch.testing.assert_close(tensors[0], torch.full((3,), 3.0))
        torch.testing.assert_close(tensors[1], torch.full((4,), 4.0))

    def test_addcdiv_with_value_fallback(self):
        tensors = [torch.zeros(3), torch.zeros(4)]
        tensor1 = [torch.full((3,), 6.0), torch.full((4,), 8.0)]
        tensor2 = [torch.full((3,), 2.0), torch.full((4,), 2.0)]
        foreach_addcdiv_(tensors, tensor1, tensor2, value=0.5, foreach=False)
        torch.testing.assert_close(tensors[0], torch.full((3,), 1.5))
        torch.testing.assert_close(tensors[1], torch.full((4,), 2.0))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_addcdiv_cuda(self):
        tensors = [torch.zeros(3, device='cuda'), torch.zeros(4, device='cuda')]
        tensor1 = [torch.full((3,), 6.0, device='cuda'), torch.full((4,), 8.0, device='cuda')]
        tensor2 = [torch.full((3,), 2.0, device='cuda'), torch.full((4,), 2.0, device='cuda')]
        foreach_addcdiv_(tensors, tensor1, tensor2, value=1.0, foreach=True)
        torch.testing.assert_close(tensors[0], torch.full((3,), 3.0, device='cuda'))
        torch.testing.assert_close(tensors[1], torch.full((4,), 4.0, device='cuda'))


class TestForeachSqrt:
    def test_sqrt_fallback(self):
        tensors = [torch.full((3,), 4.0), torch.full((4,), 9.0)]
        result = foreach_sqrt(tensors, foreach=False)
        torch.testing.assert_close(result[0], torch.full((3,), 2.0))
        torch.testing.assert_close(result[1], torch.full((4,), 3.0))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_sqrt_cuda(self):
        tensors = [torch.full((3,), 4.0, device='cuda'), torch.full((4,), 9.0, device='cuda')]
        result = foreach_sqrt(tensors, foreach=True)
        torch.testing.assert_close(result[0], torch.full((3,), 2.0, device='cuda'))
        torch.testing.assert_close(result[1], torch.full((4,), 3.0, device='cuda'))


class TestForeachSqrtInplace:
    def test_sqrt_inplace_fallback(self):
        tensors = [torch.full((3,), 4.0), torch.full((4,), 9.0)]
        foreach_sqrt_(tensors, foreach=False)
        torch.testing.assert_close(tensors[0], torch.full((3,), 2.0))
        torch.testing.assert_close(tensors[1], torch.full((4,), 3.0))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_sqrt_inplace_cuda(self):
        tensors = [torch.full((3,), 4.0, device='cuda'), torch.full((4,), 9.0, device='cuda')]
        foreach_sqrt_(tensors, foreach=True)
        torch.testing.assert_close(tensors[0], torch.full((3,), 2.0, device='cuda'))
        torch.testing.assert_close(tensors[1], torch.full((4,), 3.0, device='cuda'))


class TestForeachMaximum:
    def test_maximum_fallback(self):
        tensors = [torch.tensor([1.0, 3.0, 2.0]), torch.tensor([4.0, 1.0])]
        other = [torch.tensor([2.0, 2.0, 2.0]), torch.tensor([3.0, 3.0])]
        foreach_maximum_(tensors, other, foreach=False)
        torch.testing.assert_close(tensors[0], torch.tensor([2.0, 3.0, 2.0]))
        torch.testing.assert_close(tensors[1], torch.tensor([4.0, 3.0]))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_maximum_cuda(self):
        tensors = [torch.tensor([1.0, 3.0, 2.0], device='cuda'), torch.tensor([4.0, 1.0], device='cuda')]
        other = [torch.tensor([2.0, 2.0, 2.0], device='cuda'), torch.tensor([3.0, 3.0], device='cuda')]
        foreach_maximum_(tensors, other, foreach=True)
        torch.testing.assert_close(tensors[0], torch.tensor([2.0, 3.0, 2.0], device='cuda'))
        torch.testing.assert_close(tensors[1], torch.tensor([4.0, 3.0], device='cuda'))


class TestForeachSign:
    def test_sign_fallback(self):
        tensors = [torch.tensor([-2.0, 0.0, 3.0]), torch.tensor([4.0, -1.0])]
        foreach_sign_(tensors, foreach=False)
        torch.testing.assert_close(tensors[0], torch.tensor([-1.0, 0.0, 1.0]))
        torch.testing.assert_close(tensors[1], torch.tensor([1.0, -1.0]))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_sign_cuda(self):
        tensors = [torch.tensor([-2.0, 0.0, 3.0], device='cuda'), torch.tensor([4.0, -1.0], device='cuda')]
        foreach_sign_(tensors, foreach=True)
        torch.testing.assert_close(tensors[0], torch.tensor([-1.0, 0.0, 1.0], device='cuda'))
        torch.testing.assert_close(tensors[1], torch.tensor([1.0, -1.0], device='cuda'))


class TestForeachNeg:
    def test_neg_fallback(self):
        tensors = [torch.tensor([1.0, -2.0, 3.0]), torch.tensor([-4.0, 5.0])]
        foreach_neg_(tensors, foreach=False)
        torch.testing.assert_close(tensors[0], torch.tensor([-1.0, 2.0, -3.0]))
        torch.testing.assert_close(tensors[1], torch.tensor([4.0, -5.0]))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_neg_cuda(self):
        tensors = [torch.tensor([1.0, -2.0, 3.0], device='cuda'), torch.tensor([-4.0, 5.0], device='cuda')]
        foreach_neg_(tensors, foreach=True)
        torch.testing.assert_close(tensors[0], torch.tensor([-1.0, 2.0, -3.0], device='cuda'))
        torch.testing.assert_close(tensors[1], torch.tensor([4.0, -5.0], device='cuda'))


class TestForeachSub:
    def test_sub_scalar_fallback(self):
        tensors = [torch.full((3,), 5.0), torch.full((4,), 3.0)]
        foreach_sub_(tensors, 2.0, foreach=False)
        torch.testing.assert_close(tensors[0], torch.full((3,), 3.0))
        torch.testing.assert_close(tensors[1], torch.full((4,), 1.0))

    def test_sub_tensor_list_fallback(self):
        tensors = [torch.full((3,), 5.0), torch.full((4,), 6.0)]
        other = [torch.full((3,), 2.0), torch.full((4,), 3.0)]
        foreach_sub_(tensors, other, alpha=1.0, foreach=False)
        torch.testing.assert_close(tensors[0], torch.full((3,), 3.0))
        torch.testing.assert_close(tensors[1], torch.full((4,), 3.0))

    def test_sub_tensor_list_with_alpha_fallback(self):
        tensors = [torch.full((3,), 5.0), torch.full((4,), 6.0)]
        other = [torch.full((3,), 2.0), torch.full((4,), 4.0)]
        foreach_sub_(tensors, other, alpha=0.5, foreach=False)
        torch.testing.assert_close(tensors[0], torch.full((3,), 4.0))
        torch.testing.assert_close(tensors[1], torch.full((4,), 4.0))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_sub_scalar_cuda(self):
        tensors = [torch.full((3,), 5.0, device='cuda'), torch.full((4,), 3.0, device='cuda')]
        foreach_sub_(tensors, 2.0, foreach=True)
        torch.testing.assert_close(tensors[0], torch.full((3,), 3.0, device='cuda'))
        torch.testing.assert_close(tensors[1], torch.full((4,), 1.0, device='cuda'))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_sub_tensor_list_cuda(self):
        tensors = [torch.full((3,), 5.0, device='cuda'), torch.full((4,), 6.0, device='cuda')]
        other = [torch.full((3,), 2.0, device='cuda'), torch.full((4,), 3.0, device='cuda')]
        foreach_sub_(tensors, other, alpha=1.0, foreach=True)
        torch.testing.assert_close(tensors[0], torch.full((3,), 3.0, device='cuda'))
        torch.testing.assert_close(tensors[1], torch.full((4,), 3.0, device='cuda'))

    def test_sub_integer_scalar_fallback(self):
        tensors = [torch.full((3,), 5.0), torch.full((4,), 3.0)]
        foreach_sub_(tensors, 2, alpha=1.0, foreach=False)
        torch.testing.assert_close(tensors[0], torch.full((3,), 3.0))


class TestForeachDiv:
    def test_div_scalar_fallback(self):
        tensors = [torch.full((3,), 6.0), torch.full((4,), 8.0)]
        foreach_div_(tensors, 2.0, foreach=False)
        torch.testing.assert_close(tensors[0], torch.full((3,), 3.0))
        torch.testing.assert_close(tensors[1], torch.full((4,), 4.0))

    def test_div_tensor_list_fallback(self):
        tensors = [torch.full((3,), 6.0), torch.full((4,), 9.0)]
        other = [torch.full((3,), 2.0), torch.full((4,), 3.0)]
        foreach_div_(tensors, other, foreach=False)
        torch.testing.assert_close(tensors[0], torch.full((3,), 3.0))
        torch.testing.assert_close(tensors[1], torch.full((4,), 3.0))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_div_scalar_cuda(self):
        tensors = [torch.full((3,), 6.0, device='cuda'), torch.full((4,), 8.0, device='cuda')]
        foreach_div_(tensors, 2.0, foreach=True)
        torch.testing.assert_close(tensors[0], torch.full((3,), 3.0, device='cuda'))
        torch.testing.assert_close(tensors[1], torch.full((4,), 4.0, device='cuda'))


class TestForeachZero:
    def test_zero_fallback(self):
        tensors = [torch.full((3,), 5.0), torch.full((4,), 3.0)]
        foreach_zero_(tensors, foreach=False)
        torch.testing.assert_close(tensors[0], torch.zeros(3))
        torch.testing.assert_close(tensors[1], torch.zeros(4))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_zero_cuda(self):
        tensors = [torch.full((3,), 5.0, device='cuda'), torch.full((4,), 3.0, device='cuda')]
        foreach_zero_(tensors, foreach=True)
        torch.testing.assert_close(tensors[0], torch.zeros(3, device='cuda'))
        torch.testing.assert_close(tensors[1], torch.zeros(4, device='cuda'))


class TestForeachCopy:
    def test_copy_fallback(self):
        tensors = [torch.zeros(3), torch.zeros(4)]
        src = [torch.full((3,), 2.0), torch.full((4,), 3.0)]
        foreach_copy_(tensors, src, foreach=False)
        torch.testing.assert_close(tensors[0], torch.full((3,), 2.0))
        torch.testing.assert_close(tensors[1], torch.full((4,), 3.0))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_copy_cuda(self):
        tensors = [torch.zeros(3, device='cuda'), torch.zeros(4, device='cuda')]
        src = [torch.full((3,), 2.0, device='cuda'), torch.full((4,), 3.0, device='cuda')]
        foreach_copy_(tensors, src, foreach=True)
        torch.testing.assert_close(tensors[0], torch.full((3,), 2.0, device='cuda'))
        torch.testing.assert_close(tensors[1], torch.full((4,), 3.0, device='cuda'))


class TestForeachClampMin:
    def test_clamp_min_fallback(self):
        tensors = [torch.tensor([-1.0, 0.5, 2.0]), torch.tensor([-2.0, 3.0])]
        foreach_clamp_min_(tensors, 0.0, foreach=False)
        torch.testing.assert_close(tensors[0], torch.tensor([0.0, 0.5, 2.0]))
        torch.testing.assert_close(tensors[1], torch.tensor([0.0, 3.0]))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_clamp_min_cuda(self):
        tensors = [torch.tensor([-1.0, 0.5, 2.0], device='cuda'), torch.tensor([-2.0, 3.0], device='cuda')]
        foreach_clamp_min_(tensors, 0.0, foreach=True)
        torch.testing.assert_close(tensors[0], torch.tensor([0.0, 0.5, 2.0], device='cuda'))
        torch.testing.assert_close(tensors[1], torch.tensor([0.0, 3.0], device='cuda'))


class TestForeachEdgeCases:
    def test_add_tensor_as_other(self):
        tensors = [torch.ones(3), torch.ones(4)]
        other = torch.tensor(2.0)
        foreach_add_(tensors, other, foreach=False)
        torch.testing.assert_close(tensors[0], torch.full((3,), 3.0))
        torch.testing.assert_close(tensors[1], torch.full((4,), 3.0))

    def test_sub_tensor_as_other(self):
        tensors = [torch.full((3,), 5.0), torch.full((4,), 3.0)]
        other = torch.tensor(2.0)
        foreach_sub_(tensors, other, foreach=False)
        torch.testing.assert_close(tensors[0], torch.full((3,), 3.0))
        torch.testing.assert_close(tensors[1], torch.full((4,), 1.0))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_add_tensor_cuda(self):
        tensors = [torch.ones(3, device='cuda'), torch.ones(4, device='cuda')]
        other = torch.tensor(2.0, device='cuda')
        foreach_add_(tensors, other, foreach=True)
        torch.testing.assert_close(tensors[0], torch.full((3,), 3.0, device='cuda'))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_sub_tensor_cuda(self):
        tensors = [torch.full((3,), 5.0, device='cuda'), torch.full((4,), 3.0, device='cuda')]
        other = torch.tensor(2.0, device='cuda')
        foreach_sub_(tensors, other, foreach=True)
        torch.testing.assert_close(tensors[0], torch.full((3,), 3.0, device='cuda'))


class TestForeachOptimizerConvergence:
    @staticmethod
    def create_model_and_data(device='cpu'):
        torch.manual_seed(42)
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        ).to(device)
        x = torch.randn(32, 10, device=device)
        y = torch.randn(32, 1, device=device)
        return model, x, y

    @staticmethod
    def train_steps(optimizer, model, x, y, steps=10):
        losses = []
        for _ in range(steps):
            optimizer.zero_grad()
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses

    @staticmethod
    def clone_model(model):
        return copy.deepcopy(model)

    def test_lion_foreach_convergence(self):
        model, x, y = self.create_model_and_data()
        model_clone = self.clone_model(model)

        opt_foreach = Lion(model.parameters(), lr=1e-4, foreach=False)
        opt_no_foreach = Lion(model_clone.parameters(), lr=1e-4, foreach=False)

        losses_foreach = self.train_steps(opt_foreach, model, x, y, steps=20)
        losses_no_foreach = self.train_steps(opt_no_foreach, model_clone, x, y, steps=20)

        for l1, l2 in zip(losses_foreach, losses_no_foreach):
            assert abs(l1 - l2) < 1e-5, f'Loss mismatch: {l1} vs {l2}'

        for p1, p2 in zip(model.parameters(), model_clone.parameters()):
            torch.testing.assert_close(p1, p2, rtol=1e-5, atol=1e-5)

    def test_stableadamw_foreach_convergence(self):
        model, x, y = self.create_model_and_data()
        model_clone = self.clone_model(model)

        opt1 = StableAdamW(model.parameters(), lr=1e-3, kahan_sum=False, foreach=False)
        opt2 = StableAdamW(model_clone.parameters(), lr=1e-3, kahan_sum=False, foreach=False)

        losses1 = self.train_steps(opt1, model, x, y, steps=20)
        losses2 = self.train_steps(opt2, model_clone, x, y, steps=20)

        for l1, l2 in zip(losses1, losses2):
            assert abs(l1 - l2) < 1e-5, f'Loss mismatch: {l1} vs {l2}'

        for p1, p2 in zip(model.parameters(), model_clone.parameters()):
            torch.testing.assert_close(p1, p2, rtol=1e-5, atol=1e-5)

    def test_lamb_foreach_convergence(self):
        model, x, y = self.create_model_and_data()
        model_clone = self.clone_model(model)

        opt1 = Lamb(model.parameters(), lr=1e-3, foreach=False)
        opt2 = Lamb(model_clone.parameters(), lr=1e-3, foreach=False)

        losses1 = self.train_steps(opt1, model, x, y, steps=20)
        losses2 = self.train_steps(opt2, model_clone, x, y, steps=20)

        for l1, l2 in zip(losses1, losses2):
            assert abs(l1 - l2) < 1e-5, f'Loss mismatch: {l1} vs {l2}'

        for p1, p2 in zip(model.parameters(), model_clone.parameters()):
            torch.testing.assert_close(p1, p2, rtol=1e-5, atol=1e-5)

    def test_lars_foreach_convergence(self):
        model, x, y = self.create_model_and_data()
        model_clone = self.clone_model(model)

        opt1 = LARS(model.parameters(), lr=1e-3, foreach=False)
        opt2 = LARS(model_clone.parameters(), lr=1e-3, foreach=False)

        losses1 = self.train_steps(opt1, model, x, y, steps=20)
        losses2 = self.train_steps(opt2, model_clone, x, y, steps=20)

        for l1, l2 in zip(losses1, losses2):
            assert abs(l1 - l2) < 1e-5, f'Loss mismatch: {l1} vs {l2}'

        for p1, p2 in zip(model.parameters(), model_clone.parameters()):
            torch.testing.assert_close(p1, p2, rtol=1e-5, atol=1e-5)

    def test_lion_loss_decreases(self):
        model, x, y = self.create_model_and_data()
        opt = Lion(model.parameters(), lr=1e-4, foreach=False)
        losses = self.train_steps(opt, model, x, y, steps=50)

        assert losses[-1] < losses[0], f'Loss should decrease: {losses[0]} -> {losses[-1]}'

    def test_stableadamw_loss_decreases(self):
        model, x, y = self.create_model_and_data()
        opt = StableAdamW(model.parameters(), lr=1e-3, kahan_sum=False, foreach=False)
        losses = self.train_steps(opt, model, x, y, steps=50)

        assert losses[-1] < losses[0], f'Loss should decrease: {losses[0]} -> {losses[-1]}'

    def test_lamb_loss_decreases(self):
        model, x, y = self.create_model_and_data()
        opt = Lamb(model.parameters(), lr=1e-3, foreach=False)
        losses = self.train_steps(opt, model, x, y, steps=50)

        assert losses[-1] < losses[0], f'Loss should decrease: {losses[0]} -> {losses[-1]}'

    def test_lars_loss_decreases(self):
        model, x, y = self.create_model_and_data()
        opt = LARS(model.parameters(), lr=1e-3, foreach=False)
        losses = self.train_steps(opt, model, x, y, steps=50)

        assert losses[-1] < losses[0], f'Loss should decrease: {losses[0]} -> {losses[-1]}'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_lion_foreach_vs_no_foreach_cuda(self):
        model, x, y = self.create_model_and_data(device='cuda')
        model_clone = self.clone_model(model)

        opt_foreach = Lion(model.parameters(), lr=1e-4, foreach=True)
        opt_no_foreach = Lion(model_clone.parameters(), lr=1e-4, foreach=False)

        losses_foreach = self.train_steps(opt_foreach, model, x, y, steps=20)
        losses_no_foreach = self.train_steps(opt_no_foreach, model_clone, x, y, steps=20)

        for l1, l2 in zip(losses_foreach, losses_no_foreach):
            assert abs(l1 - l2) < 1e-4, f'Loss mismatch: {l1} vs {l2}'

        for p1, p2 in zip(model.parameters(), model_clone.parameters()):
            torch.testing.assert_close(p1, p2, rtol=1e-4, atol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_stableadamw_foreach_vs_no_foreach_cuda(self):
        model, x, y = self.create_model_and_data(device='cuda')
        model_clone = self.clone_model(model)

        opt_foreach = StableAdamW(model.parameters(), lr=1e-3, kahan_sum=False, foreach=True)
        opt_no_foreach = StableAdamW(model_clone.parameters(), lr=1e-3, kahan_sum=False, foreach=False)

        losses_foreach = self.train_steps(opt_foreach, model, x, y, steps=20)
        losses_no_foreach = self.train_steps(opt_no_foreach, model_clone, x, y, steps=20)

        assert losses_foreach[-1] < losses_foreach[0]
        assert losses_no_foreach[-1] < losses_no_foreach[0]
        assert abs(losses_foreach[-1] - losses_no_foreach[-1]) < 0.1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_lamb_foreach_vs_no_foreach_cuda(self):
        model, x, y = self.create_model_and_data(device='cuda')
        model_clone = self.clone_model(model)

        opt_foreach = Lamb(model.parameters(), lr=1e-3, foreach=True)
        opt_no_foreach = Lamb(model_clone.parameters(), lr=1e-3, foreach=False)

        losses_foreach = self.train_steps(opt_foreach, model, x, y, steps=20)
        losses_no_foreach = self.train_steps(opt_no_foreach, model_clone, x, y, steps=20)

        for l1, l2 in zip(losses_foreach, losses_no_foreach):
            assert abs(l1 - l2) < 1e-4, f'Loss mismatch: {l1} vs {l2}'

        for p1, p2 in zip(model.parameters(), model_clone.parameters()):
            torch.testing.assert_close(p1, p2, rtol=1e-4, atol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_lars_foreach_vs_no_foreach_cuda(self):
        model, x, y = self.create_model_and_data(device='cuda')
        model_clone = self.clone_model(model)

        opt_foreach = LARS(model.parameters(), lr=1e-3, foreach=True)
        opt_no_foreach = LARS(model_clone.parameters(), lr=1e-3, foreach=False)

        losses_foreach = self.train_steps(opt_foreach, model, x, y, steps=20)
        losses_no_foreach = self.train_steps(opt_no_foreach, model_clone, x, y, steps=20)

        for l1, l2 in zip(losses_foreach, losses_no_foreach):
            assert abs(l1 - l2) < 1e-4, f'Loss mismatch: {l1} vs {l2}'

        for p1, p2 in zip(model.parameters(), model_clone.parameters()):
            torch.testing.assert_close(p1, p2, rtol=1e-4, atol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_adabelief_foreach_vs_no_foreach_cuda(self):
        model, x, y = self.create_model_and_data(device='cuda')
        model_clone = self.clone_model(model)

        opt_foreach = AdaBelief(model.parameters(), lr=1e-3, foreach=True)
        opt_no_foreach = AdaBelief(model_clone.parameters(), lr=1e-3, foreach=False)

        losses_foreach = self.train_steps(opt_foreach, model, x, y, steps=20)
        losses_no_foreach = self.train_steps(opt_no_foreach, model_clone, x, y, steps=20)

        assert losses_foreach[-1] < losses_foreach[0]
        assert losses_no_foreach[-1] < losses_no_foreach[0]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_adan_foreach_vs_no_foreach_cuda(self):
        model, x, y = self.create_model_and_data(device='cuda')
        model_clone = self.clone_model(model)

        opt_foreach = Adan(model.parameters(), lr=1e-3, foreach=True)
        opt_no_foreach = Adan(model_clone.parameters(), lr=1e-3, foreach=False)

        losses_foreach = self.train_steps(opt_foreach, model, x, y, steps=20)
        losses_no_foreach = self.train_steps(opt_no_foreach, model_clone, x, y, steps=20)

        assert losses_foreach[-1] < losses_foreach[0]
        assert losses_no_foreach[-1] < losses_no_foreach[0]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_adan_foreach_with_weight_decouple_cuda(self):
        model, x, y = self.create_model_and_data(device='cuda')
        model_clone = self.clone_model(model)

        opt_foreach = Adan(model.parameters(), lr=1e-3, weight_decay=0.01, weight_decouple=True, foreach=True)
        opt_no_foreach = Adan(
            model_clone.parameters(), lr=1e-3, weight_decay=0.01, weight_decouple=True, foreach=False
        )

        losses_foreach = self.train_steps(opt_foreach, model, x, y, steps=20)
        losses_no_foreach = self.train_steps(opt_no_foreach, model_clone, x, y, steps=20)

        assert losses_foreach[-1] < losses_foreach[0]
        assert losses_no_foreach[-1] < losses_no_foreach[0]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_adan_foreach_without_weight_decouple_cuda(self):
        model, x, y = self.create_model_and_data(device='cuda')

        opt = Adan(model.parameters(), lr=1e-3, weight_decay=0.01, weight_decouple=False, foreach=True)
        losses = self.train_steps(opt, model, x, y, steps=20)

        assert losses[-1] < losses[0]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_adan_foreach_with_grad_norm_cuda(self):
        model, x, y = self.create_model_and_data(device='cuda')

        opt = Adan(model.parameters(), lr=1e-3, max_grad_norm=1.0, foreach=True)
        losses = self.train_steps(opt, model, x, y, steps=20)

        assert losses[-1] < losses[0]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_adopt_foreach_vs_no_foreach_cuda(self):
        model, x, y = self.create_model_and_data(device='cuda')
        model_clone = self.clone_model(model)

        opt_foreach = ADOPT(model.parameters(), lr=1e-3, foreach=True)
        opt_no_foreach = ADOPT(model_clone.parameters(), lr=1e-3, foreach=False)

        losses_foreach = self.train_steps(opt_foreach, model, x, y, steps=20)
        losses_no_foreach = self.train_steps(opt_no_foreach, model_clone, x, y, steps=20)

        assert losses_foreach[-1] < losses_foreach[0]
        assert losses_no_foreach[-1] < losses_no_foreach[0]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_adopt_foreach_decoupled_cuda(self):
        model, x, y = self.create_model_and_data(device='cuda')

        opt = ADOPT(model.parameters(), lr=1e-3, weight_decay=0.01, decouple=True, foreach=True)
        losses = self.train_steps(opt, model, x, y, steps=20)

        assert losses[-1] < losses[0]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_tiger_foreach_vs_no_foreach_cuda(self):
        model, x, y = self.create_model_and_data(device='cuda')
        model_clone = self.clone_model(model)

        opt_foreach = Tiger(model.parameters(), lr=1e-3, foreach=True)
        opt_no_foreach = Tiger(model_clone.parameters(), lr=1e-3, foreach=False)

        losses_foreach = self.train_steps(opt_foreach, model, x, y, steps=20)
        losses_no_foreach = self.train_steps(opt_no_foreach, model_clone, x, y, steps=20)

        for l1, l2 in zip(losses_foreach, losses_no_foreach):
            assert abs(l1 - l2) < 1e-4, f'Loss mismatch: {l1} vs {l2}'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_sgdw_foreach_vs_no_foreach_cuda(self):
        model, x, y = self.create_model_and_data(device='cuda')
        model_clone = self.clone_model(model)

        opt_foreach = SGDW(model.parameters(), lr=1e-2, momentum=0.9, foreach=True)
        opt_no_foreach = SGDW(model_clone.parameters(), lr=1e-2, momentum=0.9, foreach=False)

        losses_foreach = self.train_steps(opt_foreach, model, x, y, steps=20)
        losses_no_foreach = self.train_steps(opt_no_foreach, model_clone, x, y, steps=20)

        assert losses_foreach[-1] < losses_foreach[0]
        assert losses_no_foreach[-1] < losses_no_foreach[0]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_signsgd_foreach_vs_no_foreach_cuda(self):
        model, x, y = self.create_model_and_data(device='cuda')
        model_clone = self.clone_model(model)

        opt_foreach = SignSGD(model.parameters(), lr=1e-2, momentum=0.9, foreach=True)
        opt_no_foreach = SignSGD(model_clone.parameters(), lr=1e-2, momentum=0.9, foreach=False)

        losses_foreach = self.train_steps(opt_foreach, model, x, y, steps=20)
        losses_no_foreach = self.train_steps(opt_no_foreach, model_clone, x, y, steps=20)

        assert losses_foreach[-1] < losses_foreach[0]
        assert losses_no_foreach[-1] < losses_no_foreach[0]


class TestForeachCanUseChecks:
    """Test _can_use_foreach method conditions."""

    @staticmethod
    def create_model(device='cpu'):
        torch.manual_seed(42)
        return torch.nn.Linear(10, 1).to(device)

    def test_lion_foreach_disabled_with_maximize(self):
        model = self.create_model()
        opt = Lion(model.parameters(), lr=1e-4, maximize=True, foreach=True)

        x = torch.randn(4, 10)
        y = torch.randn(4, 1)

        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()

    def test_tiger_foreach_disabled_with_maximize(self):
        model = self.create_model()
        opt = Tiger(model.parameters(), lr=1e-3, maximize=True, foreach=True)

        x = torch.randn(4, 10)
        y = torch.randn(4, 1)

        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()

    def test_adabelief_foreach_disabled_with_rectify(self):
        model = self.create_model()
        opt = AdaBelief(model.parameters(), lr=1e-3, rectify=True, foreach=True)

        x = torch.randn(4, 10)
        y = torch.randn(4, 1)

        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()

    def test_adabelief_foreach_disabled_with_ams_bound(self):
        model = self.create_model()
        opt = AdaBelief(model.parameters(), lr=1e-3, ams_bound=True, foreach=True)

        x = torch.randn(4, 10)
        y = torch.randn(4, 1)

        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()

    def test_adabelief_foreach_disabled_with_maximize(self):
        model = self.create_model()
        opt = AdaBelief(model.parameters(), lr=1e-3, maximize=True, foreach=True)

        x = torch.randn(4, 10)
        y = torch.randn(4, 1)

        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()

    def test_adan_foreach_disabled_with_maximize(self):
        model = self.create_model()
        opt = Adan(model.parameters(), lr=1e-3, maximize=True, foreach=True)

        x = torch.randn(4, 10)
        y = torch.randn(4, 1)

        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()

    def test_adopt_foreach_disabled_with_maximize(self):
        model = self.create_model()
        opt = ADOPT(model.parameters(), lr=1e-3, maximize=True, foreach=True)

        x = torch.randn(4, 10)
        y = torch.randn(4, 1)

        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()

    def test_foreach_explicit_false(self):
        model = self.create_model()
        opt = Lion(model.parameters(), lr=1e-4, foreach=False)

        x = torch.randn(4, 10)
        y = torch.randn(4, 1)

        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()

    def test_foreach_with_no_grad_params(self):
        model = self.create_model()
        for p in model.parameters():
            p.requires_grad = False

        opt = Lion(model.parameters(), lr=1e-4, foreach=True)
        opt.step()

    def test_tiger_foreach_disabled_with_complex_tensor(self):
        model = torch.nn.Linear(10, 1, dtype=torch.complex64)
        opt = Tiger(model.parameters(), lr=1e-3, foreach=True)

        x = torch.randn(4, 10, dtype=torch.complex64)
        y = torch.randn(4, 1, dtype=torch.complex64)

        opt.zero_grad()
        loss = (model(x) - y).abs().mean()
        loss.backward()
        opt.step()

    def test_tiger_foreach_disabled_with_sparse_grad(self):
        from pytorch_optimizer.base.exception import NoSparseGradientError

        embedding = torch.nn.Embedding(100, 10, sparse=True)
        opt = Tiger(embedding.parameters(), lr=1e-3, foreach=True)

        indices = torch.randint(0, 100, (4,))
        opt.zero_grad()
        output = embedding(indices).sum()
        output.backward()
        with pytest.raises(NoSparseGradientError):
            opt.step()

    def test_lion_foreach_disabled_with_sparse_grad(self):
        from pytorch_optimizer.base.exception import NoSparseGradientError

        embedding = torch.nn.Embedding(100, 10, sparse=True)
        opt = Lion(embedding.parameters(), lr=1e-4, foreach=True)

        indices = torch.randint(0, 100, (4,))
        opt.zero_grad()
        output = embedding(indices).sum()
        output.backward()
        with pytest.raises(NoSparseGradientError):
            opt.step()

    def test_lamb_foreach_disabled_with_maximize(self):
        model = self.create_model()
        opt = Lamb(model.parameters(), lr=1e-3, maximize=True, foreach=True)

        x = torch.randn(4, 10)
        y = torch.randn(4, 1)

        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()

    def test_lamb_foreach_disabled_with_sparse_grad(self):
        embedding = torch.nn.Embedding(100, 10, sparse=True)
        opt = Lamb(embedding.parameters(), lr=1e-3, foreach=True)

        indices = torch.randint(0, 100, (4,))
        opt.zero_grad()
        output = embedding(indices).sum()
        output.backward()
        opt.step()

    def test_lars_foreach_disabled_with_maximize(self):
        model = self.create_model()
        opt = LARS(model.parameters(), lr=1e-3, maximize=True, foreach=True)

        x = torch.randn(4, 10)
        y = torch.randn(4, 1)

        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()

    def test_lars_foreach_disabled_with_sparse_grad(self):
        embedding = torch.nn.Embedding(100, 10, sparse=True)
        opt = LARS(embedding.parameters(), lr=1e-3, foreach=True)

        indices = torch.randint(0, 100, (4,))
        opt.zero_grad()
        output = embedding(indices).sum()
        output.backward()
        opt.step()

    def test_sgdw_foreach_disabled_with_maximize(self):
        model = self.create_model()
        opt = SGDW(model.parameters(), lr=1e-2, momentum=0.9, maximize=True, foreach=True)

        x = torch.randn(4, 10)
        y = torch.randn(4, 1)

        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()

    def test_sgdw_foreach_disabled_with_no_momentum(self):
        model = self.create_model()
        opt = SGDW(model.parameters(), lr=1e-2, momentum=0.0, foreach=True)

        x = torch.randn(4, 10)
        y = torch.randn(4, 1)

        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()

    def test_sgdw_foreach_disabled_with_sparse_grad(self):
        embedding = torch.nn.Embedding(100, 10, sparse=True)
        opt = SGDW(embedding.parameters(), lr=1e-2, momentum=0.9, foreach=True)

        indices = torch.randint(0, 100, (4,))
        opt.zero_grad()
        output = embedding(indices).sum()
        output.backward()
        opt.step()

    def test_sgdw_foreach_disabled_with_no_params(self):
        model = self.create_model()
        for p in model.parameters():
            p.requires_grad = False

        opt = SGDW(model.parameters(), lr=1e-2, momentum=0.9, foreach=True)
        opt.step()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='need GPU')
    def test_adan_foreach_with_grad_clipping_cuda(self):
        """Test that grad clipping code path is covered when gradients exceed norm."""
        torch.manual_seed(42)
        model = torch.nn.Linear(10, 100, device='cuda')
        opt = Adan(model.parameters(), lr=1e-3, max_grad_norm=0.01, foreach=True)

        x = torch.randn(32, 10, device='cuda') * 100
        y = torch.randn(32, 100, device='cuda')

        for _ in range(3):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            opt.step()
