from typing import List, Optional

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import CLOSURE, DEFAULTS, GROUP, LOSS, PARAMETERS


class SRMM(BaseOptimizer):
    """Stochastic regularized majorization-minimization with weakly convex and multi-convex surrogates.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param beta: float. adaptivity weight.
    :param memory_length: Optional[int]. internal memory length for moving average. None for no refreshing.
    :param maximize: bool. maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 0.01,
        beta: float = 0.5,
        memory_length: Optional[int] = 100,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(beta, 'beta', 0.0, 1.0, range_type='[]')

        self.maximize = maximize

        defaults: DEFAULTS = {'lr': lr, 'beta': beta, 'memory_length': memory_length}

        super().__init__(params, defaults)

        self.base_lrs: List[float] = [group['lr'] for group in self.param_groups]

    def __str__(self) -> str:
        return 'SRMM'

    def init_group(self, group: GROUP, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]

            if len(state) == 0:
                state['mov_avg_grad'] = torch.zeros_like(grad)
                state['mov_avg_param'] = torch.zeros_like(grad)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' not in group:
                self.init_group(group)
                group['step'] = 1
            else:
                group['step'] += 1

            w_t: float = (
                (group['step'] % (group['memory_length'] if group['memory_length'] is not None else 1)) + 1
            ) ** -group['beta']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                mov_avg_grad, mov_avg_param = state['mov_avg_grad'], state['mov_avg_param']

                mov_avg_grad.mul_(1.0 - w_t).add_(grad, alpha=w_t)
                mov_avg_param.mul_(1.0 - w_t).add_(p, alpha=w_t)

                mov_avg_param.add_(mov_avg_grad, alpha=-group['lr'])

                p.copy_(mov_avg_param)

        return loss
