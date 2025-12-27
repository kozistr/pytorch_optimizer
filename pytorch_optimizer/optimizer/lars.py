from typing import List, Optional

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, Parameters, ParamGroup
from pytorch_optimizer.optimizer.foreach_utils import (
    foreach_add_,
    foreach_copy_,
    foreach_mul_,
)


class LARS(BaseOptimizer):
    """Layer-wise Adaptive Rate Scaling (no rate scaling or weight decay for parameters <= 1D).

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        weight_decay (float): Weight decay (L2 penalty).
        momentum (float): Momentum.
        dampening (float): Dampening for momentum.
        trust_coefficient (float): Trust coefficient.
        nesterov (bool): Enables Nesterov momentum.
        foreach (Optional[bool]): Whether to use foreach (multi-tensor) operations for speed.
            None means auto-detect based on device (True for CUDA, False otherwise).
        maximize (bool): Maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        dampening: float = 0.0,
        trust_coefficient: float = 1e-3,
        nesterov: bool = False,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(momentum, 'momentum', 0.0, 1.0)
        self.validate_range(dampening, 'dampening', 0.0, 1.0)
        self.validate_non_negative(trust_coefficient, 'trust_coefficient')

        self.foreach = foreach
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'weight_decay': weight_decay,
            'momentum': momentum,
            'dampening': dampening,
            'trust_coefficient': trust_coefficient,
            'nesterov': nesterov,
            'foreach': foreach,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Lars'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if group['momentum'] > 0.0:
                state = self.state[p]

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = grad.clone()

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        """Check if foreach can be used for this group.

        Foreach is disabled when using features that require per-parameter handling:
        - Nesterov momentum (requires per-parameter gradient modification)
        """
        if group.get('foreach') is False:
            return False

        if group.get('nesterov'):
            return False

        return self.can_use_foreach(group, group.get('foreach'))

    def _step_foreach(
        self,
        group: ParamGroup,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        momentum_buffers: List[torch.Tensor],
    ) -> None:
        """Foreach-optimized step for a parameter group."""
        momentum = group['momentum']
        dampening = group['dampening']
        weight_decay = group['weight_decay']
        trust_coeff = group['trust_coefficient']
        lr = group['lr']

        if self.maximize:
            torch._foreach_neg_(grads)

        scaled_grads = []
        for p, grad in zip(params, grads):
            if p.ndim > 1:
                param_norm = torch.linalg.norm(p)
                update_norm = torch.linalg.norm(grad)

                trust_ratio = trust_coeff * param_norm / update_norm if param_norm > 0.0 and update_norm > 0.0 else 1.0

                scaled_grad = grad.add(p, alpha=weight_decay).mul_(trust_ratio)
            else:
                scaled_grad = grad.clone()

            scaled_grads.append(scaled_grad)

        if momentum > 0.0:
            foreach_mul_(momentum_buffers, momentum, foreach=True)
            foreach_add_(momentum_buffers, scaled_grads, alpha=1.0 - dampening, foreach=True)
            foreach_copy_(scaled_grads, momentum_buffers, foreach=True)

        foreach_add_(params, scaled_grads, alpha=-lr, foreach=True)

    def _step_per_param(self, group: ParamGroup) -> None:
        """Per-parameter step (original implementation)."""
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad

            self.maximize_gradient(grad, maximize=self.maximize)

            state = self.state[p]

            if p.ndim > 1:
                param_norm = torch.linalg.norm(p)
                update_norm = torch.linalg.norm(grad)

                one = torch.ones_like(param_norm)

                trust_ratio = torch.where(
                    param_norm > 0.0,
                    torch.where(update_norm > 0.0, (group['trust_coefficient'] * param_norm / update_norm), one),
                    one,
                )

                grad.add_(p, alpha=group['weight_decay'])
                grad.mul_(trust_ratio)

            if group['momentum'] > 0.0:
                mb = state['momentum_buffer']
                mb.mul_(group['momentum']).add_(grad, alpha=1.0 - group['dampening'])

                if group['nesterov']:
                    grad.add_(mb, alpha=group['momentum'])
                else:
                    grad.copy_(mb)

            p.add_(grad, alpha=-group['lr'])

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            if self._can_use_foreach(group) and group['momentum'] > 0.0:
                params, grads, state_dict = self.collect_trainable_params(
                    group, self.state, state_keys=['momentum_buffer']
                )
                if params:
                    self._step_foreach(group, params, grads, state_dict['momentum_buffer'])
            else:
                self._step_per_param(group)

        return loss
