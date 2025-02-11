import math
import warnings
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Tuple, Union

import numpy as np
import torch
from hyperopt import fmin, hp, tpe
from hyperopt.exceptions import AllTrialsFailed
from matplotlib import pyplot as plt
from torch import nn

from pytorch_optimizer.optimizer import OPTIMIZERS
from pytorch_optimizer.optimizer.alig import l2_projection

# ─── Warning ignoring ─────────────────────────────────────────────────────────
# Ignore specific warnings to avoid clutter in the output
warnings.filterwarnings('ignore', category=UserWarning)

# ─── Configs ──────────────────────────────────────────────────────────────────
# Configuration constants for the optimization experiments

# List of optimizers to ignore
OPTIMIZERS_IGNORE = ('lomo', 'adalomo', 'demo', 'a2grad', 'alig') # BUG: fix `alig`, invalid .__name__

# Optimizers that require the model as input during initialization
OPTIMIZERS_MODEL_INPUT_NEEDED = ('lomo', 'adalomo', 'adammini')

# Optimizers that require the computation graph to be retained for second-order derivatives
OPTIMIZERS_GRAPH_NEEDED = ('adahessian', 'sophiah')

# Optimizers that require a closure for step
OPTIMIZERS_CLOSURE_NEEDED = ('alig', 'bsam')

# Number of evaluations per hyperparameter during hyperparameter optimization
# - More = better hyperparameter tuning but slower / less = faster but less accurate
EVAL_PER_HYPYPERPARAM = 540

# Number of optimization steps during hyperparameter tuning
OPTIMIZATION_STEPS = 300

# Number of optimization steps during final testing with the best hyperparameters
TESTING_OPTIMIZATION_STEPS = 650

# Makes determine difficult to optimize
DIFFICULT_RASTRIGIN = False

# Enable/disable average loss penalty
USE_AVERAGE_LOSS_PENALTY = True

# Scaling factor for average loss
AVERAGE_LOSS_PENALTY_FACTOR = 1.0

# Random seed for reproducibility
SEARCH_SEED = 42

# Minimum loss threshold to stop hyperparameter optimization early
# - Should be tuned when using USE_AVERAGE_LOSS_PENALTY relative to AVERAGE_LOSS_PENALTY_FACTOR
# - Should be tuned when changing OPTIMIZATION_STEPS
LOSS_MIN_TRESH = 0 # 0 to remove the early stopping

# ─── Optimizer Search Space Configuration ─────────────────────────────────────
# Define the search space for hyperparameters of different optimizers

# Default hyperparameter search space (for most optimizers)
default_search_space = {'lr': hp.uniform('lr', 0, 2)}

# Special cases with additional hyperparameters or different ranges
special_search_spaces = {
    'adafactor': {'lr': hp.uniform('lr', 0, 10)}, # Wider range of lr
    'adams': {'lr': hp.uniform('lr', 0, 10)}, # Wider range of lr
    'dadaptadagrad': {'lr': hp.uniform('lr', 0, 10)},  # Wider range of lr
    'dadaptlion': {'lr': hp.uniform('lr', 0, 10)},  # Wider range of lr
    'padam': {'lr': hp.uniform('lr', 0, 10)},  # Wider range of lr
    'dadaptadam': {'lr': hp.uniform('lr', 0, 10)},  # Wider range of lr
    'adahessian': {'lr': hp.uniform('lr', 0, 800)},  # Wider range for second-order optimizers
    'sophiah': {'lr': hp.uniform('lr', 0, 60)},  # Wider range for second-order optimizers
    'pid': {
        'lr': hp.uniform('lr', 0, 0.5),
        'derivative': hp.quniform('derivative', 2, 14, 0.5),
        'integral': hp.quniform('integral', 1, 10, 0.5),
        'momentum': hp.quniform('momentum', 0, 0.99, 0.01),
    },  # Tune the derivative and integral terms + momentum coefficient
    'sgdp': {
        'lr': hp.uniform('lr', 0, 0.8),
        'momentum': hp.quniform('momentum', 0, 0.99, 0.01),
    },  # Tune the momentum coefficient
    'accsgd': {
        'lr': hp.uniform('lr', 0, 0.8),
        'momentum': hp.quniform('momentum', 0, 0.99, 0.01),
        'kappa': hp.uniformint('kappa', 1, 1000),
    },  # Tune the momentum coefficient + kappa
    'sgdw': {
        'lr': hp.uniform('lr', 0, 0.8),
        'momentum': hp.quniform('momentum', 0, 0.99, 0.01),
    },  # Tune the momentum coefficient
    'signsgd': {
        'lr': hp.uniform('lr', 0, 0.8),
        'momentum': hp.quniform('momentum', 0, 0.99, 0.01),
    },  # Tune the momentum coefficient
    'sgdsai': {
        'lr': hp.uniform('lr', 0, 0.8),
        'momentum': hp.quniform('momentum', 0, 0.99, 0.01),
    },  # Tune the momentum coefficient
    'sgd': {
        'lr': hp.uniform('lr', 0, 0.8),
        'momentum': hp.quniform('momentum', 0, 0.99, 0.01),
    },  # Tune the momentum coefficient
    'AliG': {
        'lr': hp.uniform('lr', 0, 0.8),
        'momentum': hp.quniform('momentum', 0, 0.99, 0.01),
    },  # Tune the momentum coefficient
    'asgd': {
        'lr': hp.uniform('lr', 0, 0.8),
        'amplifier':  hp.uniform('amplifier', 0, 0.5),
    },
    'amos': {
        'lr': hp.uniform('lr', 0, 0.8),
        'momentum': hp.quniform('momentum', 0, 0.99, 0.01),
    }, # Tune the momentum coefficient
    'schedulefreesgd': {
        'lr': hp.uniform('lr', 0, 3),
        'momentum': hp.quniform('momentum', 0, 0.99, 0.01),
    }, # Tune the momentum coefficient + high lr range
    'kron': {
        'lr': hp.uniform('lr', 0, 0.8),
        'momentum': hp.quniform('momentum', 0, 0.99, 0.01),
    }, # Tune the momentum coefficient
    'muon': {
        'lr': hp.uniform('lr', 0, 0.8),
        'momentum': hp.quniform('momentum', 0, 0.99, 0.01),
    }, # Tune the momentum coefficient
}

# ─── Test Functions ──────────────────────────────────────────────────────────
# Define test functions for optimization experiments


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
    noise_scale: float = 0.2,
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


# ─── Model Definition ────────────────────────────────────────────────────────
# Define a simple PyTorch model for optimization


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


# ─── Core Optimization Logic ────────────────────────────────────────────────
# Core logic for executing optimization steps


def execute_steps(
    func: Callable,
    initial_state: Tuple[float, float],
    optimizer_class: torch.optim.Optimizer,
    optimizer_config: Dict,
    num_iters: int = 500,
) -> torch.Tensor:
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

    # Initialize the model and optimizer
    model = Model(func, initial_state)
    parameters = list(model.parameters())
    optimizer_name: str = optimizer_class.__name__.lower()

    # Special handling for optimizers with unique requirements
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

    # Special initialization for memory-efficient optimizers
    if optimizer_name in OPTIMIZERS_MODEL_INPUT_NEEDED:
        optimizer = optimizer_class(model, **optimizer_config)
    else:
        optimizer = optimizer_class(parameters, **optimizer_config)

    # Track optimization path
    losses = []
    steps = torch.zeros((2, num_iters + 1), dtype=torch.float32)
    steps[:, 0] = model.x.detach()

    for i in range(1, num_iters + 1):
        optimizer.zero_grad()
        loss = model()
        losses.append(loss.item())

        # Special handling for second-order optimizers
        create_graph = optimizer_name in OPTIMIZERS_GRAPH_NEEDED
        loss.backward(create_graph=create_graph)

        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(parameters, 1.0)

        # Closure required for certain optimizers
        closure = create_closure(loss) if optimizer_name in OPTIMIZERS_CLOSURE_NEEDED else None
        optimizer.step(closure)

        steps[:, i] = model.x.detach()

    return steps, losses


# ─── Hyperparameter Optimization ─────────────────────────────────────────────
# Logic for optimizing hyperparameters using Hyperopt


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
    """
    Objective function for hyperparameter optimization with boundary constraints and optional average loss penalty.

    Args:
        params (Dict): Dictionary containing hyperparameters (e.g., learning rate).
        criterion (Callable): Objective function to minimize.
        optimizer_class (torch.optim.Optimizer): Optimizer class to evaluate.
        initial_state (Tuple[float, float]): Starting coordinates for optimization.
        minimum (Tuple[float, float]): Known global minimum coordinates.
        x_bounds (Tuple[float, float]): Valid x range (min_x, max_x).
        y_bounds (Tuple[float, float]): Valid y range (min_y, max_y).
        num_iters (int, optional): Number of optimization steps. Defaults to 100.

    Returns:
        float: A combined loss value that includes:
            - The squared distance from the final position to the known minimum.
            - A penalty for boundary violations.
            - An optional penalty for higher average loss during optimization (if enabled).
    """
    # Execute optimization steps and get losses
    steps, losses = execute_steps(  # Modified to unpack losses
        criterion, initial_state, optimizer_class, params, num_iters
    )

    # Calculate boundary violations
    x_min_violation = torch.clamp(x_bounds[0] - steps[0], min=0).max()
    x_max_violation = torch.clamp(steps[0] - x_bounds[1], min=0).max()
    y_min_violation = torch.clamp(y_bounds[0] - steps[1], min=0).max()
    y_max_violation = torch.clamp(steps[1] - y_bounds[1], min=0).max()
    total_violation = x_min_violation + x_max_violation + y_min_violation + y_max_violation

    # Calculate average loss penalty
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    penalty = 75 * total_violation.item()
    if USE_AVERAGE_LOSS_PENALTY:
        penalty += avg_loss * AVERAGE_LOSS_PENALTY_FACTOR

    # Calculate final distance to minimum
    final_position = steps[:, -1]
    final_distance = ((final_position[0] - minimum[0]) ** 2 + (final_position[1] - minimum[1]) ** 2).item()

    return final_distance + penalty


# ─── Visualization ──────────────────────────────────────────────────────────
# Logic for generating visualizations of optimization paths


def plot_function(
    func: Callable,
    optimization_steps: torch.Tensor,
    output_path: Path,
    optimizer_name: str,
    params: dict,
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

    # Plot function contours and optimization path
    ax.contour(x_grid.numpy(), y_grid.numpy(), z.numpy(), 20, cmap='jet')
    ax.plot(optimization_steps[0], optimization_steps[1], color='r', marker='x', markersize=3)

    # Mark global minimum and final position
    plt.plot(*minimum, 'gD', label='Global Minimum')
    plt.plot(optimization_steps[0, -1], optimization_steps[1, -1], 'bD', label='Final Position')

    ax.set_title(
        f'{func.__name__} func: {optimizer_name} with {TESTING_OPTIMIZATION_STEPS} iterations\n{
            ", ".join(f"{k}={round(v, 4)}" for k, v in params.items())
        }'
    )
    plt.legend()
    plt.savefig(str(output_path))
    plt.close()


# ─── Experiment Execution ────────────────────────────────────────────────────
# Logic for running optimization experiments


def execute_experiments(
    optimizers: list,
    func: Callable,
    initial_state: Tuple[float, float],
    output_dir: Path,
    experiment_name: str,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    minimum: Tuple[float, float],
    seed: int = 42,
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
        output_path = output_dir / f'{experiment_name}_{optimizer_name}.png'
        if output_path.exists():
            continue  # Skip already generated plots

        print(f'({i}/{len(optimizers)}) Processing {optimizer_name}... (Params to tune: {", ".join(search_space.keys())})')  # noqa: E501, T201

        # Select hyperparameter search space
        num_hyperparams = len(search_space)
        max_evals = EVAL_PER_HYPYPERPARAM * num_hyperparams  # Scale evaluations based on hyperparameter count

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
                loss_threshold=LOSS_MIN_TRESH,
                max_queue_len=6,
                rstate=np.random.default_rng(seed),
            )
        except AllTrialsFailed:
            print(f'⚠️ {optimizer_name} failed to optimize {func.__name__}')  # noqa: T201
            continue

        # Run final optimization with best parameters
        steps, _ = execute_steps(  # Modified to ignore losses
            func, initial_state, optimizer_class, best_params, TESTING_OPTIMIZATION_STEPS
        )

        # Generate and save visualization
        plot_function(func, steps, output_path, optimizer_name, best_params, x_range, y_range, minimum)


# ─── Main Execution ──────────────────────────────────────────────────────────
# Main function to execute the experiments


def main():
    """Main execution routine for optimization experiments."""
    np.random.seed(SEARCH_SEED)
    torch.manual_seed(SEARCH_SEED)
    output_dir = Path('.') / 'docs' / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare the list of optimizers and their search spaces
    optimizers = [
        (optimizer, special_search_spaces.get(optimizer_name, default_search_space))
        for optimizer_name, optimizer in OPTIMIZERS.items()
        if optimizer_name not in OPTIMIZERS_IGNORE
    ]

    # Run experiments for the Rastrigin function
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

    # Run experiments for the Rosenbrock function
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
