import torch
from torch import nn
from torch.nn.functional import relu


class TrainingModel(nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32, output_features: int = 1):
        super().__init__()

        self.fc1 = nn.Linear(2, 2, dtype=dtype)
        self.fc2 = nn.Linear(2, output_features, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = relu(x.real) + 1.0j * relu(x.imag) if x.is_complex() else relu(x)

        return self.fc2(x).real


def make_parameter(
    shape: tuple[int, ...] = (1, 1),
    *,
    requires_grad: bool = True,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = 'cpu',
    grad: float | None = 0.0,
) -> torch.Tensor:
    param = torch.zeros(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    if requires_grad and grad is not None:
        param.grad = torch.full_like(param, grad)

    return param


def make_sparse_parameters() -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.randn(5, 1, requires_grad=True)
    sparse_weight = weight.detach().clone().requires_grad_()

    weight.grad = torch.rand_like(weight)
    weight.grad[0] = 0.0
    sparse_weight.grad = weight.grad.to_sparse()

    return weight, sparse_weight


def build_model(use_complex: bool = False, device: str | torch.device = 'cpu'):
    torch.manual_seed(42)
    model = TrainingModel(dtype=torch.complex64 if use_complex else torch.float32)
    return model.to(device), nn.BCEWithLogitsLoss().to(device)


def make_lora_parameters():
    param_left = nn.Parameter(torch.tensor([[1.0, 0.2, -0.3], [0.8, 0.5, -0.7]]))
    param_right = nn.Parameter(torch.tensor([[0.6, -0.4], [0.1, 0.9], [-0.8, 0.3], [0.7, -0.2]]))

    param_left.grad = torch.tensor([[0.2, 0.3, -0.2], [-0.1, 0.4, 0.5]])
    param_right.grad = torch.tensor([[0.1, -0.3], [0.2, 0.4], [-0.5, 0.1], [0.3, -0.2]])

    return param_left, param_right
