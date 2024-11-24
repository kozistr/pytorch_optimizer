from typing import Callable, Optional

import torch

from pytorch_optimizer.base.exception import NoClosureError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.optimizer.utils import get_global_gradient_norm


@torch.no_grad()
def l2_projection(parameters: PARAMETERS, max_norm: float = 1e2):
    r"""Get l2 normalized parameter."""
    global_norm = torch.sqrt(sum(p.norm().pow(2) for p in parameters))
    if global_norm > max_norm:
        ratio = max_norm / global_norm
        for param in parameters:
            param.mul_(ratio)


class AliG(BaseOptimizer):
    r"""Adaptive Learning Rates for Interpolation with Gradients.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param max_lr: Optional[float]. max learning rate.
    :param projection_fn: Callable. projection function to enforce constraints.
    :param momentum: float. momentum.
    :param adjusted_momentum: bool. if True, use pytorch-like momentum, instead of standard Nesterov momentum.
    """

    def __init__(
        self,
        params: PARAMETERS,
        max_lr: Optional[float] = None,
        projection_fn: Optional[Callable] = None,
        momentum: float = 0.0,
        adjusted_momentum: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(max_lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0)

        self.projection_fn = projection_fn

        defaults: DEFAULTS = {'max_lr': max_lr, 'adjusted_momentum': adjusted_momentum, 'momentum': momentum}
        super().__init__(params, defaults)

        if self.projection_fn is not None:
            self.projection_fn()

    def __str__(self) -> str:
        return 'AliG'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                if group['momentum'] > 0.0:
                    state['momentum_buffer'] = torch.zeros_like(p)

    @torch.no_grad()
    def compute_step_size(self, loss: float) -> float:
        r"""Compute step_size."""
        global_grad_norm = get_global_gradient_norm(self.param_groups)
        global_grad_norm.add_(1e-6)

        return loss / global_grad_norm.item()

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        if closure is None:
            raise NoClosureError('AliG', '(e.g. `optimizer.step(lambda: float(loss))`).')

        loss = closure()

        un_clipped_step_size: float = self.compute_step_size(loss)

        for group in self.param_groups:
            step_size = group['step_size'] = (
                min(un_clipped_step_size, group['max_lr']) if group['max_lr'] is not None else un_clipped_step_size
            )
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                if len(state) == 0 and momentum > 0.0:
                    state['momentum_buffer'] = torch.zeros_like(p)

                p.add_(grad, alpha=-step_size)

                if momentum > 0.0:
                    buffer = state['momentum_buffer']

                    if group['adjusted_momentum']:
                        buffer.mul_(momentum).sub_(grad)
                        p.add_(buffer, alpha=step_size * momentum)
                    else:
                        buffer.mul_(momentum).add_(grad, alpha=-step_size)
                        p.add_(buffer, alpha=momentum)

            if self.projection_fn is not None:
                self.projection_fn()

        return loss
