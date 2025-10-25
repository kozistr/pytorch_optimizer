"""Copyright (C) 2020 Jeremy Bernstein, Arash Vahdat, Yisong Yue & Ming-Yu Liu.  All rights reserved.

Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/).
"""

import math
from typing import Optional

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, Parameters, ParamGroup


class Fromage(BaseOptimizer):
    r"""On the distance between two neural networks and the stability of learning.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param p_bound: Optional[float]. Restricts the optimisation to a bounded set. A value of 2.0 restricts parameter
        norms to lie within 2x their initial norms. This regularises the model class.
    :param maximize: bool. maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self, params: Parameters, lr: float = 1e-2, p_bound: Optional[float] = None, maximize: bool = False, **kwargs
    ):
        self.validate_learning_rate(lr)

        self.p_bound = p_bound
        self.maximize = maximize

        defaults: Defaults = {'lr': lr}

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Fromage'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            state = self.state[p]

            if len(state) == 0 and self.p_bound is not None:
                state['max'] = p.norm().mul_(self.p_bound)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' not in group:
                self.init_group(group)
                group['step'] = 1
            else:
                group['step'] += 1

            pre_factor: float = math.sqrt(1 + group['lr'] ** 2)

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
