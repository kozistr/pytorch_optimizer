# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT


class MADGRAD(BaseOptimizer):
    """Momentumized adaptive dual averaged gradient descent.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        momentum: Interpolation factor toward the previous parameter values. `0` disables momentum.
        eps: Term added to the denominator to improve numerical stability.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        maximize: Maximize the objective instead of minimizing it.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        eps: float = 1e-6,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(momentum, 'momentum', 0.0, 1.0)
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'momentum': momentum,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'MADGRAD'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if group['momentum'] > 0.0 and grad.is_sparse:
                raise NoSparseGradientError(str(self), note='momentum > 0.0')

            if group['weight_decay'] > 0.0 and not group['weight_decouple'] and grad.is_sparse:
                raise NoSparseGradientError(str(self), note='weight_decay')

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]

            if 'grad_sum_sq' not in state:
                state['grad_sum_sq'] = torch.zeros_like(p)
                state['s'] = torch.zeros_like(p)

                if group['momentum'] > 0.0:
                    state['x0'] = p.clone()

    @staticmethod
    def compute_rms(grad_sum_sq: torch.Tensor, eps: float) -> torch.Tensor:
        """Compute the cube root accumulator, treating zero denominators as inactive coordinates."""
        rms = grad_sum_sq.pow(1.0 / 3.0).add_(eps)
        if eps == 0.0:
            rms[rms == 0] = float('inf')

        return rms

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if 'k' not in self.state:
            self.state['k'] = torch.tensor([0], dtype=torch.long, requires_grad=False)

        for group in self.param_groups:
            self.init_group(group)

            weight_decay, momentum, eps = group['weight_decay'], group['momentum'], group['eps']
            lr: float = group['lr'] + eps if group['lr'] != 0.0 else 0.0

            _lambda = lr * math.pow(self.state['k'] + 1, 0.5)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                grad_sum_sq, s = state['grad_sum_sq'], state['s']
                if weight_decay > 0.0 and not group['weight_decouple']:
                    grad.add_(p, alpha=weight_decay)

                if grad.is_sparse:
                    grad = grad.coalesce()

                    p_masked = p.sparse_mask(grad)
                    grad_sum_sq_masked = grad_sum_sq.sparse_mask(grad)
                    s_masked = s.sparse_mask(grad)

                    rms_masked_values = self.compute_rms(grad_sum_sq_masked._values(), eps)
                    x0_masked_values = p_masked._values().addcdiv(s_masked._values(), rms_masked_values, value=1)

                    grad_sq = grad * grad
                    grad_sum_sq.add_(grad_sq, alpha=_lambda)
                    grad_sum_sq_masked.add_(grad_sq, alpha=_lambda)

                    rms_masked_values = self.compute_rms(grad_sum_sq_masked._values(), eps)

                    s.add_(grad, alpha=_lambda)
                    s_masked._values().add_(grad._values(), alpha=_lambda)

                    p_kp1_masked_values = x0_masked_values.addcdiv(s_masked._values(), rms_masked_values, value=-1)

                    p_masked._values().add_(p_kp1_masked_values, alpha=-1)
                    p.data.add_(p_masked, alpha=-1)
                else:
                    if momentum == 0.0:
                        rms = self.compute_rms(grad_sum_sq, eps)
                        x0 = p.addcdiv(s, rms, value=1)
                    else:
                        x0 = state['x0']

                    grad_sum_sq.addcmul_(grad, grad, value=_lambda)
                    rms = self.compute_rms(grad_sum_sq, eps)

                    s.add_(grad, alpha=_lambda)

                    p_old: torch.Tensor | None = None
                    if weight_decay > 0.0 and group['weight_decouple']:
                        p_old = p.clone()

                    if momentum == 0.0:
                        p.copy_(x0.addcdiv(s, rms, value=-1))
                    else:
                        z = x0.addcdiv(s, rms, value=-1)
                        p.lerp_(z, weight=1.0 - momentum)

                    if weight_decay > 0.0 and group['weight_decouple']:
                        p.add_(p_old, alpha=-lr * weight_decay)

        self.state['k'].add_(1)

        return loss
