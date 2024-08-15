from typing import List, Optional

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import CLOSURE, DEFAULTS, LOSS, PARAMETERS


class SRMM(BaseOptimizer):
    """Stochastic regularized majorization-minimization with weakly convex and multi-convex surrogates.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param beta: float. adaptivity weight.
    :param memory_length: Optional[int]. internal memory length for moving average. None for no refreshing.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 0.01,
        beta: float = 0.5,
        memory_length: Optional[int] = 100,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(beta, 'beta', 0.0, 1.0, range_type='[]')

        defaults: DEFAULTS = {'lr': lr, 'beta': beta, 'memory_length': memory_length}
        super().__init__(params, defaults)

        self.base_lrs: List[float] = [group['lr'] for group in self.param_groups]

    def __str__(self) -> str:
        return 'SRMM'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['mov_avg_grad'] = torch.zeros_like(p)
                state['mov_avg_param'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            w_t: float = (
                (group['step'] % (group['memory_length'] if group['memory_length'] is not None else 1)) + 1
            ) ** -group['beta']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                if len(state) == 0:
                    state['mov_avg_grad'] = torch.zeros_like(p)
                    state['mov_avg_param'] = torch.zeros_like(p)

                mov_avg_grad, mov_avg_param = state['mov_avg_grad'], state['mov_avg_param']

                mov_avg_grad.mul_(1.0 - w_t).add_(grad, alpha=w_t)
                mov_avg_param.mul_(1.0 - w_t).add_(p, alpha=w_t)

                mov_avg_param.add_(mov_avg_grad, alpha=-group['lr'])

                p.copy_(mov_avg_param)

        return loss
