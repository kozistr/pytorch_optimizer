import torch

from pytorch_optimizer.optimizer.foreach_utils import group_tensors_by_device_and_dtype, has_foreach_support


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
