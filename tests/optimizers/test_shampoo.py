import pytest
import torch

from pytorch_optimizer.optimizer.utils.partition import BlockPartitioner
from pytorch_optimizer.optimizer.utils.precision import to_real
from pytorch_optimizer.optimizer.utils.preconditioner import PreConditioner
from tests.fixtures import make_parameter
from tests.utils import build_optimizer


@pytest.mark.parametrize('pre_conditioner_type', [0, 1, 2])
def test_scalable_shampoo_pre_conditioner_with_svd(pre_conditioner_type):
    params = [make_parameter(shape) for shape in ((8, 2), (4, 8), (1, 4))]
    optimizer = build_optimizer(
        'scalableshampoo',
        params,
        block_size=4,
        start_preconditioning_step=1,
        preconditioning_compute_steps=1,
        pre_conditioner_type=pre_conditioner_type,
        use_svd=True,
    )
    optimizer.step()
    assert all(torch.isfinite(param).all() for param in params)


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
        pre_conditioner.compute_pre_conditioners()

    @pytest.mark.parametrize('pre_conditioner_type', [0, 1, 2, 3])
    def test_pre_conditioner_type(self, pre_conditioner_type):
        var = torch.zeros((4, 4, 32))
        if pre_conditioner_type in (0, 1, 2):
            PreConditioner(var, 0.9, 0, 128, 1, 8192, True, pre_conditioner_type=pre_conditioner_type)
        else:
            with pytest.raises(ValueError):
                PreConditioner(var, 0.9, 0, 128, 1, 8192, True, pre_conditioner_type=pre_conditioner_type)
