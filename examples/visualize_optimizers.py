import math
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union
from warnings import filterwarnings

import numpy as np
import torch
from hyperopt import fmin, hp, tpe
from hyperopt.exceptions import AllTrialsFailed
from matplotlib import pyplot as plt
from torch import nn

from pytorch_optimizer.optimizer import OPTIMIZERS
from pytorch_optimizer.optimizer.alig import l2_projection

filterwarnings('ignore', category=UserWarning)

IMG_FORMAT = 'jpg'
OPTIMIZERS_IGNORE = ('lomo', 'adalomo', 'demo', 'a2grad', 'muon', 'alice', 'adamc', 'adamwsn', 'adamuon', 'splus')
OPTIMIZERS_MODEL_INPUT_NEEDED = ('lomo', 'adalomo', 'adammini')
OPTIMIZERS_GRAPH_NEEDED = ('adahessian', 'sophiah')
OPTIMIZERS_CLOSURE_NEEDED = ('alig', 'bsam')
EVAL_PER_HYPERPARAM: int = 600
OPTIMIZATION_STEPS: int = 200
TESTING_OPTIMIZATION_STEPS: dict[str, int] = {'rastrigin': 150, 'rosenbrock': 400}
DIFFICULT_RASTRIGIN: bool = False
CONVERGENCE_LOSS_PENALTY_FACTOR: float = 0.2
OSCILLATIONS_LOSS_PENALTY_FACTOR: float = 0.1
AVERAGE_LOSS_PENALTY_FACTOR: float = 0.4
SEARCH_SEED: int = 42
LOSS_MIN_THRESHOLD: float = 0.0

DEFAULT_SEARCH_SPACES = {'lr': hp.uniform('lr', 0, 2)}
SPECIAL_SEARCH_SPACES = {
    'adafactor': {'lr': hp.uniform('lr', 0, 10)},
    'adams': {'lr': hp.uniform('lr', 0, 10)},
    'dadaptadagrad': {'lr': hp.uniform('lr', 0, 10)},
    'dadaptlion': {'lr': hp.uniform('lr', 0, 10)},
    'padam': {'lr': hp.uniform('lr', 0, 10)},
    'dadaptadam': {'lr': hp.uniform('lr', 0, 10)},
    'adahessian': {'lr': hp.uniform('lr', 0, 800)},
    'sophiah': {'lr': hp.uniform('lr', 0, 60)},
    'pid': {
        'lr': hp.uniform('lr', 0, 0.5),
        'derivative': hp.quniform('derivative', 2, 14, 0.5),
        'integral': hp.quniform('integral', 1, 10, 0.5),
        'momentum': hp.quniform('momentum', 0, 0.99, 0.01),
    },
    'sgdp': {
        'lr': hp.uniform('lr', 0, 0.8),
        'momentum': hp.quniform('momentum', 0, 0.99, 0.01),
    },
    'accsgd': {
        'lr': hp.uniform('lr', 0, 0.8),
        'momentum': hp.quniform('momentum', 0, 0.99, 0.01),
        'kappa': hp.uniformint('kappa', 1, 1000),
    },
    'sgdw': {
        'lr': hp.uniform('lr', 0, 0.8),
        'momentum': hp.quniform('momentum', 0, 0.99, 0.01),
    },
    'signsgd': {
        'lr': hp.uniform('lr', 0, 0.8),
        'momentum': hp.quniform('momentum', 0, 0.99, 0.01),
    },
    'sgdsai': {
        'lr': hp.uniform('lr', 0, 0.8),
        'momentum': hp.quniform('momentum', 0, 0.99, 0.01),
    },
    'sgd': {
        'lr': hp.uniform('lr', 0, 0.8),
        'momentum': hp.quniform('momentum', 0, 0.99, 0.01),
    },
    'alig': {
        'max_lr': hp.uniform('max_lr', 0, 0.8),
        'momentum': hp.quniform('momentum', 0, 0.99, 0.01),
    },
    'asgd': {
        'lr': hp.uniform('lr', 0, 0.8),
        'amplifier': hp.uniform('amplifier', 0, 0.5),
    },
    'amos': {
        'lr': hp.uniform('lr', 0, 0.8),
        'momentum': hp.quniform('momentum', 0, 0.99, 0.01),
    },
    'schedulefreesgd': {
        'lr': hp.uniform('lr', 0, 3),
        'momentum': hp.quniform('momentum', 0, 0.99, 0.01),
    },
    'schedulefreeradam': {
        'lr': hp.uniform('lr', 1, 10),
    },
    'kron': {
        'lr': hp.uniform('lr', 0, 0.8),
        'momentum': hp.quniform('momentum', 0, 0.99, 0.01),
    },
    'scion': {
        'lr': hp.uniform('lr', 0, 1),
        'scale': hp.uniform('scale', 1.0, 1000.0),
    },
    'scionlight': {
        'lr': hp.uniform('lr', 0, 1),
        'scale': hp.uniform('scale', 1.0, 1000.0),
    },
}


def ackley(
    x: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], a: float = 20.0, b: float = 0.2, c: float = 2.0 * np.pi
) -> torch.Tensor:
    """
    Ackley function (non-convex, global minimum at (0, 0)).

    Args:
        x: Input tensor containing 2D coordinates.
        a: hyperparameter.
        b: hyperparameter.
        c: hyperparameter.

    Returns:
        torch.Tensor: Computed function value.
    """
    sum_sq = -a * torch.exp(-b * torch.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2)))
    cos = -torch.exp(0.5 * (torch.cos(c * x[0]) + torch.cos(c * x[1])))
    return sum_sq + cos + np.exp(1) + a


def rosenbrock(x: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """
    Rosenbrock function (non-convex, global minimum at (1, 1)).

    Args:
        x: Input tensor containing 2D coordinates.

    Returns:
        torch.Tensor: Computed function value.
    """
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2  # fmt: skip


def rastrigin(
    x: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
    a: float = 10.0,
    add_noise: bool = DIFFICULT_RASTRIGIN,
    noise_scale: float = 0.1,
) -> torch.Tensor:
    """
    Rastrigin function (non-convex, global minimum at (0, 0)).

    Args:
        x: Input tensor containing 2D coordinates.
        a: Function parameter controlling complexity.
        add_noise: Flag to add noise to the function.
        noise_scale: Scale of the noise to be added.

    Returns:
        torch.Tensor: Computed function value.
    """
    # Compute the Rastrigin function value
    result = a * 2 + (x[0] ** 2 - a * torch.cos(x[0] * math.pi * 2)) + (x[1] ** 2 - a * torch.cos(x[1] * math.pi * 2))

    # Add noise if the flag is True
    if add_noise:
        noise = torch.randn_like(result) * noise_scale
        result += noise

    return result


class Model(nn.Module):
    """
    Simple 2D optimization model maintaining state for parameters being optimized.
    """

    def __init__(self, func: Callable, initial_state: Tuple[float, float]) -> None:
        """
        Args:
            func: Objective function to optimize.
            initial_state: Starting point for optimization (x, y).
        """
        super().__init__()
        self.func = func
        self.x: torch.Tensor = nn.Parameter(torch.tensor(initial_state, dtype=torch.float32, requires_grad=True))

    def forward(self) -> torch.Tensor:
        """
        Compute objective function value at current parameters.

        Returns:
            torch.Tensor: Computed function value.
        """
        return self.func(self.x)


def execute_steps(
    func: Callable,
    initial_state: Tuple[float, float],
    optimizer_class: torch.optim.Optimizer,
    optimizer_config: Dict,
    num_iters: int = 500,
) -> Tuple[torch.Tensor, List[float]]:
    """
    Execute optimization steps for a given configuration.

    Args:
        func: Objective function to minimize.
        initial_state: Starting coordinates (x0, y0).
        optimizer_class: Optimizer class to use.
        optimizer_config: Dictionary of optimizer parameters.
        num_iters: Number of optimization iterations.

    Returns:
        torch.Tensor: Tensor containing optimization path (2 x num_iters+1).
    """

    def create_closure(loss: torch.Tensor) -> Callable:
        """
        Create a closure function required by certain optimizers.

        Args:
            loss: Current loss value.

        Returns:
            Callable: Closure function.
        """

        def closure() -> float:
            return loss.item()

        return closure

    model = Model(func, initial_state)
    parameters = list(model.parameters())
    optimizer_name: str = optimizer_class.__name__.lower()

    if optimizer_name == 'ranger21':
        optimizer_config['num_iterations'] = num_iters
    elif optimizer_name == 'ranger25':
        optimizer_config['orthograd'] = False
    elif optimizer_name == 'adashift':
        optimizer_config['keep_num'] = 1
    elif optimizer_name == 'alig':
        optimizer_config['projection_fn'] = lambda: l2_projection(parameters, max_norm=1)
    elif optimizer_name == 'bsam':
        optimizer_config['num_data'] = 1

    optimizer = optimizer_class(
        model if optimizer_name in OPTIMIZERS_MODEL_INPUT_NEEDED else parameters, **optimizer_config
    )

    steps = torch.zeros((2, num_iters + 1), dtype=torch.float32)
    steps[:, 0] = model.x.detach()

    losses = []
    for i in range(1, num_iters + 1):
        optimizer.zero_grad()

        loss = model()
        losses.append(loss.item())

        loss.backward(create_graph=optimizer_name in OPTIMIZERS_GRAPH_NEEDED)

        nn.utils.clip_grad_norm_(parameters, 1.0)

        closure = create_closure(loss) if optimizer_name in OPTIMIZERS_CLOSURE_NEEDED else None
        optimizer.step(closure)

        steps[:, i] = model.x.detach()

    return steps, losses


def objective(
    params: Dict,
    criterion: Callable,
    optimizer_class: torch.optim.Optimizer,
    initial_state: Tuple[float, float],
    minimum: Tuple[float, float],
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    num_iters: int = 100,
) -> float:
    """Objective function for hyperparameter optimization evaluating multiple performance metrics.

    Args:
        params: Dictionary containing hyperparameters (e.g., learning rate).
        criterion: Objective function to minimize.
        optimizer_class: Optimizer class to evaluate.
        initial_state: Starting coordinates for optimization.
        minimum: Known global minimum coordinates.
        x_bounds: Valid x range (min_x, max_x).
        y_bounds: Valid y range (min_y, max_y).
        num_iters: Number of optimization steps. Defaults to 100.

    Returns:
        float: Combined loss value incorporating:
            - Final distance to minimum
            - Boundary constraint violations
            - Optimization path stability
            - Convergence quality metrics
    """
    steps, losses = execute_steps(criterion, initial_state, optimizer_class, params, num_iters)

    violations = (
        torch.clamp(x_bounds[0] - steps[0], min=0).max()
        + torch.clamp(steps[0] - x_bounds[1], min=0).max()
        + torch.clamp(y_bounds[0] - steps[1], min=0).max()
        + torch.clamp(steps[1] - y_bounds[1], min=0).max()
    )

    final_pos = steps[:, -1]
    distance = ((final_pos[0] - minimum[0]) ** 2 + (final_pos[1] - minimum[1]) ** 2).item()

    avg_loss = sum(losses) / len(losses) if losses else 0.0
    min_loss = min(losses) if losses else 0.0

    return distance + (
        100 * violations.item()
        + AVERAGE_LOSS_PENALTY_FACTOR * avg_loss
        + OSCILLATIONS_LOSS_PENALTY_FACTOR * torch.mean(torch.abs(steps[:, 1:] - steps[:, :-1])).item()
        + CONVERGENCE_LOSS_PENALTY_FACTOR * (avg_loss - min_loss)
    )


def plot_function(
    func: Callable,
    optimization_steps: torch.Tensor,
    output_path: Path,
    optimizer_name: str,
    params: Dict,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    minimum: Tuple[float, float],
) -> None:
    """
    Generate a contour plot of the function with the optimization path.

    Args:
        func: Objective function to plot.
        optimization_steps: Tensor containing optimization path.
        output_path: Path to save generated plot.
        optimizer_name: Name of optimizer used.
        params: Dictionary containing hyperparameters (e.g., learning rate).
        x_range: X-axis range for plotting.
        y_range: Y-axis range for plotting.
        minimum: Known global minimum coordinates.
    """
    x = torch.linspace(x_range[0], x_range[1], 200)
    y = torch.linspace(y_range[0], y_range[1], 200)
    x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
    z = func([x_grid, y_grid])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)

    ax.contour(x_grid.numpy(), y_grid.numpy(), z.numpy(), 20, cmap='jet')
    ax.plot(optimization_steps[0], optimization_steps[1], color='r', marker='x', markersize=3)

    plt.plot(*minimum, 'gD', label='Global Minimum')
    plt.plot(optimization_steps[0, -1], optimization_steps[1, -1], 'bD', label='Final Position')

    config: str = ', '.join(f'{k}={round(v, 4)}' for k, v in params.items())
    ax.set_title(
        f'{func.__name__} func: {optimizer_name} with {TESTING_OPTIMIZATION_STEPS[func.__name__]} iterations\n{config}'
    )
    plt.legend()
    plt.savefig(str(output_path))
    plt.close()


def execute_experiments(
    optimizers: List,
    func: Callable,
    initial_state: Tuple[float, float],
    output_dir: Path,
    experiment_name: str,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    minimum: Tuple[float, float],
    seed: int = SEARCH_SEED,
) -> None:
    """
    Run optimization experiments for multiple optimizers.

    Args:
        optimizers: List of optimizer classes and their search spaces.
        func: Objective function to optimize.
        initial_state: Starting coordinates for optimization.
        output_dir: Directory to save visualizations.
        experiment_name: Base name for output files.
        x_range: X-axis range for plotting.
        y_range: Y-axis range for plotting.
        minimum: Known global minimum coordinates.
        seed: Random seed for reproducibility.
    """
    for i, (optimizer_class, search_space) in enumerate(optimizers, start=1):
        optimizer_name = optimizer_class.__name__
        output_path = output_dir / f'{experiment_name}_{optimizer_name}.{IMG_FORMAT}'
        if output_path.exists():
            continue

        print(  # noqa: T201
            f'({i}/{len(optimizers)}) Processing {optimizer_name}... (Params to tune: {", ".join(search_space.keys())})'  # noqa: E501
        )

        num_hyperparams: int = len(search_space)
        max_evals: int = EVAL_PER_HYPERPARAM * num_hyperparams

        objective_fn = partial(
            objective,
            criterion=func,
            optimizer_class=optimizer_class,
            initial_state=initial_state,
            minimum=minimum,
            x_bounds=x_range,
            y_bounds=y_range,
            num_iters=OPTIMIZATION_STEPS,
        )

        try:
            best_params = fmin(
                fn=objective_fn,
                space=search_space,
                algo=tpe.suggest,
                max_evals=max_evals,
                loss_threshold=LOSS_MIN_THRESHOLD,
                rstate=np.random.default_rng(seed),
                catch_eval_exceptions=True,
            )
        except AllTrialsFailed:
            print(f'{optimizer_name} failed to optimize {func.__name__}')  # noqa: T201
            continue

        steps, _ = execute_steps(
            func, initial_state, optimizer_class, best_params.copy(), TESTING_OPTIMIZATION_STEPS[experiment_name]
        )

        plot_function(func, steps, output_path, optimizer_name, best_params, x_range, y_range, minimum)


def main():
    np.random.seed(SEARCH_SEED)
    torch.manual_seed(SEARCH_SEED)

    output_dir = Path('.') / 'docs' / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizers = [
        (optimizer, SPECIAL_SEARCH_SPACES.get(optimizer_name, DEFAULT_SEARCH_SPACES))
        for optimizer_name, optimizer in OPTIMIZERS.items()
        if optimizer_name not in OPTIMIZERS_IGNORE
    ]

    print('Executing Rastrigin experiments...')  # noqa: T201
    execute_experiments(
        optimizers,
        rastrigin,
        initial_state=(-1.9, 3.35) if DIFFICULT_RASTRIGIN else (-2.0, 3.5),
        output_dir=output_dir,
        experiment_name='rastrigin',
        x_range=(-3.6, 3.6),
        y_range=(-3.6, 3.6),
        minimum=(0.0, 0.0),
        seed=SEARCH_SEED,
    )

    print('Executing Rosenbrock experiments...')  # noqa: T201
    execute_experiments(
        optimizers,
        rosenbrock,
        initial_state=(-2.0, 2.0),
        output_dir=output_dir,
        experiment_name='rosenbrock',
        x_range=(-2, 2),
        y_range=(-1, 3),
        minimum=(1.0, 1.0),
        seed=SEARCH_SEED,
    )


if __name__ == '__main__':
    main()
