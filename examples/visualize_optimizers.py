import math
from pathlib import Path

import hyperopt.exceptions
import numpy as np
import torch
from hyperopt import fmin, hp, tpe
from matplotlib import pyplot as plt

from pytorch_optimizer.optimizer import OPTIMIZERS


def rosenbrock(tensors) -> torch.Tensor:
    """https://en.wikipedia.org/wiki/Test_functions_for_optimization"""
    x, y = tensors
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2  # fmt: skip


def rastrigin(tensors, a: float = 10) -> torch.tensor:
    """https://en.wikipedia.org/wiki/Test_functions_for_optimization"""
    x, y = tensors
    return (
        a * 2
        + (x ** 2 - a * torch.cos(x * math.pi * 2))
        + (y ** 2 - a * torch.cos(y * math.pi * 2))
    )  # fmt: skip


def execute_steps(func, initial_state, optimizer_class, optimizer_config, num_iters: int = 500):
    x = torch.Tensor(initial_state).requires_grad_(True)

    if optimizer_class.__name__ == 'Ranger21':
        optimizer_config.update({'num_iterations': num_iters})

    optimizer = optimizer_class([x], **optimizer_config)

    steps = np.zeros((2, num_iters + 1), dtype=np.float32)
    steps[:, 0] = np.array(initial_state)

    for i in range(1, num_iters + 1):
        optimizer.zero_grad()

        output = func(x)
        output.backward(create_graph=True, retain_graph=True)

        torch.nn.utils.clip_grad_norm_(x, 1.0)
        optimizer.step()

        steps[:, i] = x.detach().numpy()

    return steps


def objective_rastrigin(params, minimum=(0, 0)):
    steps = execute_steps(rastrigin, (-2.0, 3.5), params['optimizer_class'], {'lr': params['lr']}, 100)

    return (steps[0][-1] - minimum[0]) ** 2 + (steps[1][-1] - minimum[1]) ** 2


def objective_rosenbrok(params, minimum=(1.0, 1.0)):
    steps = execute_steps(rastrigin, (-2.0, 2.0), params['optimizer_class'], {'lr': params['lr']}, 100)

    return (steps[0][-1] - minimum[0]) ** 2 + (steps[1][-1] - minimum[1]) ** 2


def plot_rastrigin(grad_iter, optimizer_name, lr) -> None:
    x = torch.linspace(-4.5, 4.5, 250)
    y = torch.linspace(-4.5, 4.5, 250)

    x, y = torch.meshgrid(x, y)
    z = rastrigin([x, y])

    iter_x, iter_y = grad_iter[0, :], grad_iter[1, :]

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(1, 1, 1)
    ax.contour(x.numpy(), y.numpy(), z.numpy(), 20, cmap='jet')
    ax.plot(iter_x, iter_y, color='r', marker='x')
    ax.set_title(f'Rastrigin func: {optimizer_name} with {len(iter_x)} iterations, lr={lr:.6f}')

    plt.plot(0, 0, 'gD')
    plt.plot(iter_x[-1], iter_y[-1], 'rD')
    plt.savefig(f'../docs/visualizations/rastrigin_{optimizer_name}.png')
    plt.close()


def plot_rosenbrok(grad_iter, optimizer_name, lr):
    x = torch.linspace(-2, 2, 250)
    y = torch.linspace(-1, 3, 250)

    x, y = torch.meshgrid(x, y)
    z = rosenbrock([x, y])

    iter_x, iter_y = grad_iter[0, :], grad_iter[1, :]

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(1, 1, 1)
    ax.contour(x.numpy(), y.numpy(), z.numpy(), 90, cmap='jet')
    ax.plot(iter_x, iter_y, color='r', marker='x')

    ax.set_title(f'Rosenbrock func: {optimizer_name} with {len(iter_x)} iterations, lr={lr:.6f}')
    plt.plot(1.0, 1.0, 'gD')
    plt.plot(iter_x[-1], iter_y[-1], 'rD')
    plt.savefig(f'../docs/visualizations/rosenbrock_{optimizer_name}.png')
    plt.close()


def execute_experiments(
    optimizers, objective, func, plot_func, initial_state, root_path: Path, exp_name: str, seed: int = 42
):
    for item in optimizers:
        optimizer_class, lr_low, lr_hi = item

        if (root_path / f'{exp_name}_{optimizer_class.__name__}.png').exists():
            continue

        space = {
            'optimizer_class': hp.choice('optimizer_class', [optimizer_class]),
            'lr': hp.loguniform('lr', lr_low, lr_hi),
        }

        try:
            best = fmin(
                fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=200,
                rstate=np.random.default_rng(seed),
            )
        except hyperopt.exceptions.AllTrialsFailed:
            continue

        steps = execute_steps(
            func,
            initial_state,
            optimizer_class,
            {'lr': best['lr']},
            500,
        )

        plot_func(steps, optimizer_class.__name__, best['lr'])


def main():
    # referred https://github.com/jettify/pytorch-optimizer/blob/master/examples/viz_optimizers.py

    np.random.seed(42)
    torch.manual_seed(42)

    root_path = Path('..') / 'docs' / 'visualizations'

    optimizers = [
        (optimizer, -6, 0.5)
        for optimizer_name, optimizer in OPTIMIZERS.items()
        if optimizer_name.lower() not in {'alig', 'lomo', 'adalomo', 'bsam', 'adammini'}
    ]
    optimizers.extend([(torch.optim.AdamW, -6, 0.5), (torch.optim.Adam, -6, 0.5), (torch.optim.SGD, -6, -1.0)])

    execute_experiments(
        optimizers,
        objective_rastrigin,
        rastrigin,
        plot_rastrigin,
        (-2.0, 3.5),
        root_path,
        'rastrigin',
    )

    execute_experiments(
        optimizers,
        objective_rosenbrok,
        rosenbrock,
        plot_rosenbrok,
        (-2.0, 2.0),
        root_path,
        'rosenbrok',
    )


if __name__ == '__main__':
    main()
