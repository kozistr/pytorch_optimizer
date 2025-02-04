import math
from pathlib import Path
from typing import Callable, Dict, Tuple, Union

import torch
from matplotlib import pyplot as plt
from torch import nn

from pytorch_optimizer.optimizer import OPTIMIZERS
from pytorch_optimizer.optimizer.alig import l2_projection


class Model(nn.Module):
    def __init__(self, func: Callable, initial_state: Tuple[float, float]) -> None:
        super().__init__()
        self.func = func
        self.x: torch.Tensor = nn.Parameter(torch.tensor(initial_state, dtype=torch.float32, requires_grad=True))

    def forward(self) -> torch.Tensor:
        return self.func(self.x)


def rosenbrock(x: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2  # fmt: skip


def rastrigin(x: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], a: float = 10.0) -> torch.Tensor:
    return (
        a * 2 +
        (x[0] ** 2 - a * torch.cos(x[0] * math.pi * 2)) +
        (x[1] ** 2 - a * torch.cos(x[1] * math.pi * 2))
    )  # fmt: skip


def execute_steps(
    func: Callable, initial_state: Tuple[float, float], optimizer_class, optimizer_config: Dict, num_iters: int = 500
) -> torch.Tensor:
    def closure(x):
        def _closure() -> float:
            return x

        return _closure

    model = Model(func, initial_state)

    optimizer_name: str = optimizer_class.__name__.lower()
    parameters = list(model.parameters())

    if optimizer_name == 'ranger21':
        optimizer_config.update({'num_iterations': num_iters})
    elif optimizer_name == 'ranger25':
        optimizer_config.update({'orthograd': False})
    elif optimizer_name == 'adashift':
        optimizer_config.update({'keep_num': 1})
    elif optimizer_name == 'alig':
        optimizer_config.update({'projection_fn': lambda: l2_projection(parameters, max_norm=1)})
    elif optimizer_name == 'bsam':
        optimizer_config.update({'num_data': 1})

    if optimizer_name in ('lomo', 'adalomo', 'adammini'):
        optimizer = optimizer_class(model, **optimizer_config)
    else:
        optimizer = optimizer_class(parameters, **optimizer_config)

    steps = torch.zeros((2, num_iters + 1), dtype=torch.float32)
    steps[:, 0] = model.x.detach()

    for i in range(1, num_iters + 1):
        optimizer.zero_grad()

        loss = model()
        loss.backward(create_graph=optimizer_name in ('adahessian', 'sophiah'))

        nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step(closure(loss) if optimizer_name in ('alig', 'bsam') else None)

        steps[:, i] = model.x.detach()

    return steps


def objective(
    func: Callable,
    optimizer_class,
    lr: float,
    initial_state: Tuple[float, float],
    num_iters: int = 500,
    minimum: Tuple[float, float] = (0, 0),
) -> float:
    steps = execute_steps(func, initial_state, optimizer_class, {'lr': lr}, num_iters)
    return ((steps[0, -1] - minimum[0]) ** 2 + (steps[1, -1] - minimum[1]) ** 2).item()


def plot_function(
    func: Callable,
    grad_iter,
    optimizer_plot_path: Path,
    optimizer_name: str,
    lr: float,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
):
    x = torch.linspace(x_range[0], x_range[1], 200)
    y = torch.linspace(y_range[0], y_range[1], 200)

    x, y = torch.meshgrid(x, y, indexing='ij')
    z = func([x, y])

    iter_x, iter_y = grad_iter[0, :], grad_iter[1, :]

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(1, 1, 1)
    ax.contour(x.numpy(), y.numpy(), z.numpy(), 20, cmap='jet')
    ax.plot(iter_x, iter_y, color='r', marker='x')
    ax.set_title(f'{func.__name__} func: {optimizer_name} with {len(iter_x)} iterations, lr={lr:.6f}')

    plt.plot(0, 0, 'gD')
    plt.plot(iter_x[-1], iter_y[-1], 'rD')
    plt.savefig(str(optimizer_plot_path))
    plt.close()


def execute_experiments(
    optimizers,
    func: Callable,
    initial_state: Tuple[float, float],
    root_path: Path,
    exp_name: str,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
) -> None:
    for optimizer_class, lr_low, lr_hi in optimizers:
        optimizer_plot_path = root_path / f'{exp_name}_{optimizer_class.__name__}.png'
        if optimizer_plot_path.exists():
            continue

        lr_range = torch.logspace(lr_low, lr_hi, 200)
        losses = torch.tensor([objective(func, optimizer_class, lr.item(), initial_state) for lr in lr_range])

        best_loss: float = min(losses).item()
        best_lr: float = lr_range[losses.argmin()].item()

        steps = execute_steps(func, initial_state, optimizer_class, {'lr': best_lr}, 500)
        print(  # noqa: T201
            f'optimizer: {optimizer_class.__name__}, best loss: {best_loss:.6f}, best lr: {best_lr:.6f}'
        )

        plot_function(func, steps, optimizer_plot_path, optimizer_class.__name__, best_lr, x_range, y_range)


def main():
    torch.manual_seed(42)

    root_path = Path('.') / 'docs' / 'visualizations'
    root_path.mkdir(parents=True, exist_ok=True)

    optimizers = [
        (optimizer, -5.0, 1.0)
        for optimizer_name, optimizer in OPTIMIZERS.items()
        if optimizer_name not in ('lomo', 'adalomo', 'demo')
    ]

    execute_experiments(optimizers, rastrigin, (-2.0, 3.5), root_path, 'rastrigin', (-4.5, 4.5), (-4.5, 4.5))
    execute_experiments(optimizers, rosenbrock, (-2.0, 2.0), root_path, 'rosenbrock', (-2, 2), (-1, 3))


if __name__ == '__main__':
    main()
