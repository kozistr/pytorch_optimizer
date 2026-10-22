# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Callable

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import (
    compile_foreach_step,
    foreach_add_,
    foreach_addcmul_,
    group_tensors_by_device_and_dtype,
)


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
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.
            Float16 updates remain eager when compilation is requested.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        eps: float = 1e-6,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(momentum, 'momentum', 0.0, 1.0)
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize
        self.foreach = foreach
        self._compiled_step_foreach: Callable | None = None

        defaults: Defaults = {
            'lr': lr,
            'foreach': foreach,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'momentum': momentum,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'MADGRAD'

    def _compile_foreach(self, compile_kwargs: dict | None = None) -> None:
        self._compiled_step_foreach = compile_foreach_step(self._step_foreach, compile_kwargs)

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
            rms.masked_fill_(rms == 0, float('inf'))

        return rms

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        state: dict[str, list[torch.Tensor]],
        lr: float | torch.Tensor,
        step_scale: float | torch.Tensor,
    ) -> None:
        momentum, eps = group['momentum'], group['eps']
        grad_sum_sqs, sums = state['grad_sum_sq'], state['s']

        if self.maximize:
            torch._foreach_neg_(grads)

        if not group['weight_decouple']:
            self.apply_weight_decay_foreach(params, grads, lr, group['weight_decay'], False, False)

        if momentum == 0.0:
            rms = [self.compute_rms(moment, eps) for moment in grad_sum_sqs]
            x0s = [param.addcdiv(total, denominator) for param, total, denominator in zip(params, sums, rms)]
        else:
            x0s = state['x0']

        foreach_addcmul_(grad_sum_sqs, grads, grads, value=step_scale)
        rms = [self.compute_rms(moment, eps) for moment in grad_sum_sqs]
        foreach_add_(sums, grads, alpha=step_scale)

        old_params = [p.clone() for p in params] if group['weight_decouple'] and group['weight_decay'] > 0.0 else []

        updates = torch._foreach_addcdiv(x0s, sums, rms, value=-1.0)
        if momentum == 0.0:
            torch._foreach_copy_(params, updates)
        else:
            torch._foreach_lerp_(params, updates, weight=1.0 - momentum)

        if old_params:
            foreach_add_(params, old_params, alpha=-lr * group['weight_decay'])

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

            if self.can_use_foreach(group, group.get('foreach', self.foreach)):
                keys = ['grad_sum_sq', 's'] + (['x0'] if momentum > 0.0 else [])
                params, grads, state = self.collect_trainable_params(group, self.state, state_keys=keys)

                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    update = (
                        self._compiled_step_foreach
                        if self._compiled_step_foreach is not None and batch['params'][0].dtype != torch.float16
                        else self._step_foreach
                    )

                    update(group, batch['params'], batch['grads'], batch, lr, _lambda)

                continue

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
