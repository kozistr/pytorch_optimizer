import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, Parameters, ParamGroup


class Amos(BaseOptimizer):
    """An Adam-style Optimizer with Adaptive Weight Decay towards Model-Oriented Scale.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        beta (float): A float slightly less than 1. Recommended to set `1 - beta` approximately the same magnitude
            as the learning rate, similar to beta2 in Adam.
        momentum (float): Exponential decay rate for optional moving average of updates.
        extra_l2 (float): Additional L2 regularization.
        c_coef (float): Coefficient for decay_factor_c.
        d_coef (float): Coefficient for decay_factor_d.
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the objective with respect to the parameters, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        beta: float = 0.999,
        momentum: float = 0.0,
        extra_l2: float = 0.0,
        c_coef: float = 0.25,
        d_coef: float = 0.25,
        eps: float = 1e-18,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0, range_type='[)')
        self.validate_range(beta, 'beta', 0.0, 1.0, range_type='[)')
        self.validate_non_negative(extra_l2, 'extra_l2')
        self.validate_non_negative(eps, 'eps')

        self.c_coef = c_coef
        self.d_coef = d_coef
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'beta': beta,
            'momentum': momentum,
            'extra_l2': extra_l2,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Amos'

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

            if len(state) == 0:
                state['exp_avg_sq'] = torch.zeros((1,), dtype=p.dtype, device=p.device)
                state['decay'] = torch.zeros((1,), dtype=p.dtype, device=p.device)
                if group['momentum'] > 0.0:
                    state['exp_avg'] = torch.zeros_like(p)

    @staticmethod
    def get_scale(p: torch.Tensor) -> float:
        r"""Get expected scale for model weights."""
        if len(p.shape) == 1:
            return 0.5
        if len(p.shape) == 2:
            return math.sqrt(2 / p.size(1))
        return math.sqrt(1 / p.size(1))

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            momentum, beta = group['momentum'], group['beta']

            lr_sq: float = math.sqrt(group['lr'])
            bias_correction: float = self.debias(beta, group['step'])

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                g2 = grad.pow(2).mean()
                init_lr: float = group['lr'] * self.get_scale(p)

                exp_avg_sq = state['exp_avg_sq']
                exp_avg_sq.mul_(beta).add_(g2, alpha=1.0 - beta)

                r_v_hat = bias_correction / (exp_avg_sq + group['eps'])

                b = state['decay']
                decay_factor_c = torch.rsqrt(1.0 + self.c_coef * lr_sq * b)
                decay_factor_d = torch.reciprocal(1.0 + self.d_coef * math.sqrt(init_lr) * b)

                gamma = decay_factor_c * (group['lr'] ** 2) * r_v_hat * g2

                update = p.clone()
                update.mul_((gamma - group['extra_l2']) / 2.0)
                update.add_(r_v_hat.sqrt() * grad, alpha=init_lr)
                update.mul_(decay_factor_d)

                b.mul_(1.0 + gamma).add_(gamma)

                if momentum > 0.0:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(momentum).add_(update, alpha=1.0 - momentum)

                    update.copy_(exp_avg)

                p.add_(-update)

        return loss
