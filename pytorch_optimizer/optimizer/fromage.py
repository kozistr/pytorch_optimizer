"""Copyright (C) 2020 Jeremy Bernstein, Arash Vahdat, Yisong Yue & Ming-Yu Liu.  All rights reserved.

Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/).
"""

import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import foreach_add_, group_tensors_by_device_and_dtype


class Fromage(BaseOptimizer):
    """Gradient descent scaled by the ratio of parameter and gradient norms.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        p_bound: Restricts the optimization to a bounded set. For example, a value of 2.0 restricts parameter norms
            to lie within 2x their initial norms, which helps regularize the model class.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-2,
        p_bound: float | None = None,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)

        self.p_bound = p_bound
        self.maximize = maximize
        self.foreach = foreach

        defaults: Defaults = {'lr': lr, 'foreach': foreach}

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Fromage'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            state = self.state[p]

            if len(state) == 0 and self.p_bound is not None:
                state['max'] = p.norm().mul_(self.p_bound)

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        bounds: list[torch.Tensor],
        pre_factor: float | torch.Tensor,
    ) -> None:
        if self.maximize:
            torch._foreach_neg_(grads)

        p_norms = torch.stack(torch._foreach_norm(params))
        g_norms = torch.stack(torch._foreach_norm(grads))
        rescale = (p_norms > 0.0) & (g_norms > 0.0)
        ratios = torch.where(rescale, p_norms, 1.0) / torch.where(rescale, g_norms, 1.0)
        updates = torch._foreach_mul(grads, ratios.unbind())
        foreach_add_(params, updates, alpha=-group['lr'])
        torch._foreach_div_(params, pre_factor)

        if self.p_bound is not None:
            norms = torch.stack(torch._foreach_norm(params))
            maxima = torch.stack(bounds)
            bounded = norms > maxima
            torch._foreach_mul_(params, torch.where(bounded, maxima, 1.0).unbind())
            torch._foreach_div_(params, torch.where(bounded, norms, 1.0).unbind())

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            pre_factor = (
                (1 + group['lr'] ** 2).sqrt()
                if isinstance(group['lr'], torch.Tensor)
                else math.sqrt(1 + group['lr'] ** 2)
            )

            if self.can_use_foreach(group, group.get('foreach', self.foreach)):
                keys = ['max'] if self.p_bound is not None else []
                params, grads, state = self.collect_trainable_params(group, self.state, state_keys=keys)
                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    self._step_foreach(group, batch['params'], batch['grads'], batch.get('max', []), pre_factor)
                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                p, grad = self.view_as_real(p, grad)

                p_norm, g_norm = p.norm(), grad.norm()

                if p_norm > 0.0 and g_norm > 0.0:
                    p.add_(grad * (p_norm / g_norm), alpha=-group['lr'])
                else:
                    p.add_(grad, alpha=-group['lr'])

                p.div_(pre_factor)

                if self.p_bound is not None:
                    p_norm = p.norm()
                    if p_norm > state['max']:
                        p.mul_(state['max']).div_(p_norm)

        return loss
