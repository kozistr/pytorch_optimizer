import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import CLOSURE, DEFAULTS, LOSS, PARAMETERS


class Amos(BaseOptimizer):
    r"""An Adam-style Optimizer with Adaptive Weight Decay towards Model-Oriented Scale.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param beta: float. A float slightly < 1. We recommend setting `1 - beta` to the same order of magnitude
        as the learning rate. similarity with beta2 in Adam.
    :param momentum: float. Exponential decay rate for optional moving average of updates.
    :param extra_l2: float. Additional L2 regularization.
    :param c_coef: float. Coefficient for decay_factor_c.
    :param d_coef: float. Coefficient for decay_factor_d.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        beta: float = 0.999,
        momentum: float = 0.0,
        extra_l2: float = 0.0,
        c_coef: float = 0.25,
        d_coef: float = 0.25,
        eps: float = 1e-18,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0, range_type='[)')
        self.validate_range(beta, 'beta', 0.0, 1.0, range_type='[)')
        self.validate_non_negative(extra_l2, 'extra_l2')
        self.validate_non_negative(eps, 'eps')

        self.c_coef = c_coef
        self.d_coef = d_coef

        defaults: DEFAULTS = {
            'lr': lr,
            'beta': beta,
            'momentum': momentum,
            'extra_l2': extra_l2,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Amos'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg_sq'] = torch.zeros((1,), dtype=p.dtype, device=p.device)
                state['decay'] = torch.zeros((1,), dtype=p.dtype, device=p.device)
                if group['momentum'] > 0.0:
                    state['exp_avg'] = torch.zeros_like(p)

    @staticmethod
    def get_scale(p: torch.Tensor) -> float:
        r"""Get expected scale for model weights."""
        if len(p.shape) == 1:  # expected 'bias'
            return 0.5
        if len(p.shape) == 2:  # expected Embedding, Linear, ...
            return math.sqrt(2 / p.size(1))
        return math.sqrt(1 / p.size(1))

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

            momentum, beta = group['momentum'], group['beta']

            lr_sq: float = math.sqrt(group['lr'])
            bias_correction: float = self.debias(beta, group['step'])

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
