import pytest
import torch

from pytorch_optimizer.optimizer.utils.foreach import (
    foreach_rsqrt,
    foreach_rsqrt_,
    group_tensors_by_device_and_dtype,
    has_foreach_support,
)


class TestHasForeachSupport:
    def test_empty_list(self):
        assert not has_foreach_support([])

    def test_cpu_tensors(self):
        tensors = [torch.randn(1), torch.randn(1)]
        assert has_foreach_support(tensors)

    def test_different_devices(self):
        tensors = [torch.randn(1, device='meta'), torch.randn(1, device='cpu')]
        assert not has_foreach_support(tensors)

    def test_different_dtypes(self):
        tensors = [torch.randn(1, dtype=torch.float32), torch.randn(1, dtype=torch.float16)]
        assert not has_foreach_support(tensors)

    def test_sparse_tensors(self):
        sparse_tensor = torch.sparse_coo_tensor([[0, 1]], [1.0, 2.0], (3,), check_invariants=True)
        tensors = [torch.randn(3), sparse_tensor]
        assert not has_foreach_support(tensors)


class TestGroupTensorsByDeviceAndDtype:
    def test_empty_parameters(self):
        assert group_tensors_by_device_and_dtype([], []) == []

    @pytest.mark.parametrize('with_state', [False, True])
    def test_single_group(self, with_state):
        params = [torch.zeros(1), torch.zeros(2)]
        grads = [torch.ones_like(param) for param in params]
        state = {'exp_avg': [torch.zeros_like(param) for param in params]} if with_state else {}

        group, = group_tensors_by_device_and_dtype(params, grads, state)

        assert group['params'] is params
        assert group['grads'] is grads
        assert group['indices'] == [0, 1]
        if with_state:
            assert group['exp_avg'] is state['exp_avg']

    def test_multiple_groups_by_dtype(self):
        params = [
            torch.empty(1, device=device, dtype=dtype)
            for device, dtype in [('cpu', torch.float32), ('meta', torch.float16), ('cpu', torch.float32)]
        ]
        grads = [torch.empty_like(parameter) for parameter in params]
        state_lists = {'exp_avg': [torch.empty_like(parameter, dtype=torch.bfloat16) for parameter in params]}

        groups = group_tensors_by_device_and_dtype(params, grads, state_lists)

        assert [group['indices'] for group in groups] == [[0, 2], [1]]

        for group in groups:
            for position, index in enumerate(group['indices']):
                assert group['params'][position] is params[index]
                assert group['grads'][position] is grads[index]
                assert group['exp_avg'][position] is state_lists['exp_avg'][index]


class TestForeachOperations:
    def test_outplace_rsqrt_by_version(self):
        tensors = [torch.tensor([4.0])]
        tensors = foreach_rsqrt(tensors)
        assert tensors[0].item() == 0.5

    def test_inplace_rsqrt_by_version(self):
        tensors = [torch.tensor([4.0])]
        foreach_rsqrt_(tensors)
        assert tensors[0].item() == 0.5
