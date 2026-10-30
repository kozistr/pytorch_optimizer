import pytest
import torch

from pytorch_optimizer.optimizer.shampoo.utils import BlockPartitioner, PreConditioner
from pytorch_optimizer.optimizer.utils.precision import to_real
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

        pre_conditioner = PreConditioner(var, 0.9, 0, 4, 1, 64, True, 0)
        pre_conditioner.add_statistics(torch.zeros_like(var))
        statistics = [statistic.clone() for statistic in pre_conditioner.statistics]
        pre_conditioner.compute_pre_conditioners()

        for statistic, original in zip(pre_conditioner.statistics, statistics):
            torch.testing.assert_close(statistic, original, rtol=0.0, atol=0.0)

    def test_invalid_pre_conditioner_type(self):
        with pytest.raises(ValueError):
            PreConditioner(torch.zeros((4, 4, 32)), 0.9, 0, 128, 1, 8192, True, pre_conditioner_type=3)


class TestKLShampoo:
    @pytest.mark.parametrize('optimizer_name', ['klshampoo', 'klsoap'])
    def test_maximize_batched(self, optimizer_name, device):
        params = [make_parameter((2, 2), dtype=torch.float64, device=device) for _ in range(2)]
        params.append(make_parameter((), dtype=torch.float64, device=device))
        gradients = [
            torch.diag(torch.tensor(values, dtype=torch.float64, device=device))
            for values in ((1.0, 2.0), (3.0, 4.0))
        ]
        gradients.append(torch.tensor(0.5, dtype=torch.float64, device=device))
        optimizer = build_optimizer(
            optimizer_name,
            params,
            lr=0.1,
            betas=(0.0, 0.0),
            shampoo_beta=0.5,
            weight_decay=0.0,
            init_factor=1.0,
            max_preconditioner_value=None,
            eps=0.1,
            maximize=True,
            foreach=True,
        )

        for _ in range(2):
            for param, gradient in zip(params, gradients):
                param.grad = gradient.clone()

            optimizer.step()

        matrix_updates = [
            grad / (grad.abs() + 0.1) if optimizer_name == 'klsoap' else grad / 1.1
            for grad in gradients[:2]
        ]
        expected = [0.1 * update for update in matrix_updates]
        expected.append(0.2 * gradients[-1] / (gradients[-1].abs() + 0.1))

        torch.testing.assert_close(params, expected, atol=1e-12, rtol=1e-12)
