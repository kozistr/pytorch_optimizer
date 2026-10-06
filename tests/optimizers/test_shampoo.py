import pytest
import torch

from pytorch_optimizer.optimizer.utils.partition import BlockPartitioner
from pytorch_optimizer.optimizer.utils.precision import to_real
from pytorch_optimizer.optimizer.utils.preconditioner import PreConditioner
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


@pytest.mark.parametrize('pre_conditioner_type', [0, 1, 2])
def test_scalable_shampoo_pre_conditioner_with_svd(pre_conditioner_type):
    params = [make_parameter(shape, grad=1.0) for shape in ((8, 2), (4, 8), (1, 4))]
    reference_params = [param.detach().clone().requires_grad_() for param in params]
    reference = torch.optim.SGD(reference_params, lr=0.1, momentum=0.9, nesterov=True, weight_decay=0.01)
    optimizer = build_optimizer(
        'scalableshampoo',
        params,
        lr=0.1,
        weight_decay=0.01,
        block_size=4,
        start_preconditioning_step=1,
        preconditioning_compute_steps=1,
        pre_conditioner_type=pre_conditioner_type,
        matrix_eps=1e-2,
        use_svd=True,
    )
    for _ in range(2):
        for param in reference_params:
            param.grad = torch.ones_like(param)
        reference.step()
        optimizer.step()

        for param, reference_param in zip(params, reference_params):
            torch.testing.assert_close(param, reference_param)
            torch.testing.assert_close(
                optimizer.state[param]['momentum'], reference.state[reference_param]['momentum_buffer']
            )
            torch.testing.assert_close(param.grad, torch.ones_like(param))


class TestShampooUtils:
    def test_to_real(self):
        complex_tensor = torch.tensor(1.0j + 2.0, dtype=torch.complex64)
        assert to_real(complex_tensor) == 2.0

        real_tensor = torch.tensor(1.0, dtype=torch.float32)
        assert to_real(real_tensor) == 1.0

    def test_block_partitioner(self):
        var = torch.zeros((2, 2))
        target_var = torch.zeros((1, 1))

        partitioner = BlockPartitioner(var, block_size=2, rank=2, pre_conditioner_type=0)
        with pytest.raises(ValueError):
            partitioner.partition(target_var)

    def test_pre_conditioner(self):
        var = torch.zeros((16, 4))
        grad = torch.zeros((16, 4))

        pre_conditioner = PreConditioner(var, 0.9, 0, 4, 1, 64, True, 0)
        pre_conditioner.add_statistics(grad)
        statistics = [statistic.clone() for statistic in pre_conditioner.statistics]
        pre_conditioner.compute_pre_conditioners()

        for statistic, original in zip(pre_conditioner.statistics, statistics):
            torch.testing.assert_close(statistic, original, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize('pre_conditioner_type', [0, 1, 2, 3])
    def test_pre_conditioner_type(self, pre_conditioner_type):
        var = torch.zeros((4, 4, 32))
        if pre_conditioner_type in (0, 1, 2):
            PreConditioner(var, 0.9, 0, 128, 1, 8192, True, pre_conditioner_type=pre_conditioner_type)
        else:
            with pytest.raises(ValueError):
                PreConditioner(var, 0.9, 0, 128, 1, 8192, True, pre_conditioner_type=pre_conditioner_type)
