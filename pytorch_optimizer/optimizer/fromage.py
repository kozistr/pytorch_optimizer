"""Copyright (C) 2020 Jeremy Bernstein, Arash Vahdat, Yisong Yue & Ming-Yu Liu.  All rights reserved.

Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/).
"""

import math
from typing import Optional

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import CLOSURE, DEFAULTS, LOSS, PARAMETERS


class Fromage(BaseOptimizer):
    r"""On the distance between two neural networks and the stability of learning.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param p_bound: Optional[float]. Restricts the optimisation to a bounded set. A value of 2.0 restricts parameter
        norms to lie within 2x their initial norms. This regularises the model class.
    """

    def __init__(self, params: PARAMETERS, lr: float = 1e-2, p_bound: Optional[float] = None, **kwargs):
        self.validate_learning_rate(lr)

        self.p_bound = p_bound

        defaults: DEFAULTS = {'lr': lr}
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Fromage'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                if self.p_bound is not None:
                    state['max'] = p.norm().mul_(self.p_bound)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            pre_factor: float = math.sqrt(1 + group['lr'] ** 2)
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                if len(state) == 0 and self.p_bound is not None:
                    state['max'] = p.norm().mul_(self.p_bound)

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
