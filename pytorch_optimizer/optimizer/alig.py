from typing import Callable, Optional

import torch

from pytorch_optimizer.base.exception import NoClosureError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, Parameters, ParamGroup
from pytorch_optimizer.optimizer.utils import get_global_gradient_norm


@torch.no_grad()
def l2_projection(parameters: Parameters, max_norm: float = 1e2) -> None:
    r"""Get l2 normalized parameter."""
    global_norm = torch.sqrt(sum(p.norm().pow(2) for p in parameters or []))
    if global_norm > max_norm:
        ratio = max_norm / global_norm
        for param in parameters or []:
            param.mul_(ratio)


class AliG(BaseOptimizer):
    """Adaptive Learning Rates for Interpolation with Gradients.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        max_lr (Optional[float]): Maximum learning rate.
        projection_fn (Callable): Projection function to enforce constraints.
        momentum (float): Momentum factor.
        adjusted_momentum (bool): If True, use PyTorch-like momentum instead of standard Nesterov momentum.
        maximize (bool): Maximize the objective with respect to the parameters, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        max_lr: Optional[float] = None,
        projection_fn: Optional[Callable] = None,
        momentum: float = 0.0,
        adjusted_momentum: bool = False,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(max_lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0)

        self.projection_fn = projection_fn
        self.maximize = maximize

        defaults: Defaults = {'max_lr': max_lr, 'adjusted_momentum': adjusted_momentum, 'momentum': momentum}

        super().__init__(params, defaults)

        if self.projection_fn is not None:
            self.projection_fn()

    def __str__(self) -> str:
        return 'AliG'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        momentum: float = kwargs.get('momentum', 0.9)

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            state = self.state[p]

            if len(state) == 0 and momentum > 0.0:
                state['momentum_buffer'] = torch.zeros_like(p)

    @torch.no_grad()
    def compute_step_size(self, loss: float) -> float:
        r"""Compute step_size."""
        global_grad_norm = get_global_gradient_norm(self.param_groups)
        global_grad_norm.add_(1e-6)

        return loss / global_grad_norm.item()

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        if closure is None:
            raise NoClosureError('AliG', '(e.g. `optimizer.step(lambda: float(loss))`).')

        loss = closure()

        un_clipped_step_size: float = self.compute_step_size(loss)

        for group in self.param_groups:
            momentum = group['momentum']

            self.init_group(group, momentum=momentum)
            group['step'] += 1

            step_size = group['step_size'] = (
                min(un_clipped_step_size, group['max_lr']) if group['max_lr'] is not None else un_clipped_step_size
            )

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                p, grad, buffer = self.view_as_real(p, grad, state.get('momentum_buffer', None))

                p.add_(grad, alpha=-step_size)

                if buffer is not None:
                    if group['adjusted_momentum']:
                        buffer.mul_(momentum).sub_(grad)
                        p.add_(buffer, alpha=step_size * momentum)
                    else:
                        buffer.mul_(momentum).add_(grad, alpha=-step_size)
                        p.add_(buffer, alpha=momentum)

            if self.projection_fn is not None:
                self.projection_fn()

        return loss
