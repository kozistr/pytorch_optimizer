import math
from typing import Literal, Optional

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import CLOSURE, DEFAULTS, GROUP, LOSS, PARAMETERS

VARIANTS = Literal['uni', 'inc', 'exp']


class A2Grad(BaseOptimizer):
    r"""Optimal Adaptive and Accelerated Stochastic Gradient Descent.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: Optional[float]. learning rate. no needed.
    :param beta: float. beta.
    :param lips: float. Lipschitz constant.
    :param rho: float. represents the degree of weighting decrease, a constant smoothing factor between 0 and 1.
    :param variant: str. variant of A2Grad optimizer. 'uni', 'inc', 'exp'.
    :param maximize: bool. maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: Optional[float] = None,
        beta: float = 10.0,
        lips: float = 10.0,
        rho: float = 0.5,
        variant: VARIANTS = 'uni',
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(lips, 'lips')
        self.validate_non_negative(rho, 'rho')
        self.validate_options(variant, 'variant', ['uni', 'inc', 'exp'])

        self.variant = variant
        self.maximize = maximize

        defaults: DEFAULTS = {'beta': beta, 'lips': lips}
        if variant == 'exp':
            defaults.update({'rho': rho})

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'A2Grad'

    def init_group(self, group: GROUP, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            state = self.state[p]

            if len(state) == 0:
                state['alpha_k'] = 1.0
                state['v_k'] = torch.zeros((1,), dtype=grad.dtype, device=grad.device)
                state['avg_grad'] = grad.clone()
                state['x_k'] = p.clone()
                if self.variant == 'exp':
                    state['v_kk'] = torch.zeros((1,), dtype=grad.dtype, device=grad.device)

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

            gamma_k: float = 2.0 * group['lips'] / (group['step'] + 1)
            alpha_k_1: float = 2.0 / (group['step'] + 3)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                avg_grad, v_k, x_k, v_kk = state['avg_grad'], state['v_k'], state['x_k'], state.get('v_kk', None)
                p, grad, avg_grad, v_k, x_k, v_kk = self.view_as_real(p, grad, avg_grad, v_k, x_k, v_kk)

                avg_grad.add_(grad - avg_grad, alpha=group['step'] + 1)

                delta_k = grad.clone()
                delta_k.add_(avg_grad, alpha=-1.0)

                delta_k_sq = delta_k.pow(2).sum()

                if self.variant in ('uni', 'inc'):
                    if self.variant == 'inc':
                        v_k.mul_((group['step'] / (group['step'] + 1)) ** 2)
                    v_k.add_(delta_k_sq)
                else:
                    v_kk.mul_(group['rho']).add_(delta_k_sq, alpha=1.0 - group['rho'])
                    torch.max(v_kk, v_k, out=v_k)

                h_k = v_k.sqrt()
                if self.variant != 'uni':
                    h_k.mul_(math.sqrt(group['step'] + 1))

                coefficient = -1.0 / (gamma_k + group['beta'] * h_k.item())

                x_k.add_(grad, alpha=coefficient)

                p.mul_(1.0 - alpha_k_1).add_(x_k, alpha=alpha_k_1)
                p.add_(grad, alpha=(1.0 - alpha_k_1) * state['alpha_k'] * coefficient)

                state['alpha_k'] = alpha_k_1

        return loss
