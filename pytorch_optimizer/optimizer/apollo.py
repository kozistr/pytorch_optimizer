from typing import Optional

import numpy as np
import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import CLOSURE, DEFAULTS, LOSS, PARAMETERS


class Apollo(Optimizer, BaseOptimizer):
    r"""An Adaptive Parameter-wise Diagonal Quasi-Newton Method for Nonconvex Stochastic Optimization.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param init_lr: Optional[float]. initial learning rate (default lr / 1000).
    :param beta: float. coefficient used for computing running averages of gradient.
    :param rebound: str. rectified bound for diagonal hessian. (constant, belief).
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decay_type: str. type of weight decay. (l2, decoupled, stable).
    :param warmup_steps: int. number of warmup steps.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        init_lr: Optional[float] = None,
        beta: float = 0.9,
        rebound: str = 'constant',
        weight_decay: float = 0.0,
        weight_decay_type: str = 'l2',
        warmup_steps: int = 500,
        eps: float = 1e-4,
    ):
        self.lr = lr
        self.beta = beta
        self.rebound = rebound
        self.weight_decay = weight_decay
        self.weight_decay_type = weight_decay_type
        self.warmup_steps = warmup_steps
        self.eps = eps

        self.validate_parameters()

        self.init_lr: float = init_lr if init_lr is not None else lr / 1000.0

        defaults: DEFAULTS = {
            'lr': lr,
            'init_lr': self.init_lr,
            'beta': beta,
            'weight_decay': weight_decay,
            'eps': eps,
        }
        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_beta(self.beta)
        self.validate_rebound(self.rebound)
        self.validate_weight_decay(self.weight_decay)
        self.validate_weight_decay_type(self.weight_decay_type)
        self.validate_epsilon(self.eps)

    def __str__(self) -> str:
        return 'Apollo'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['step'] = 0
                state['exp_avg_grad'] = torch.zeros_like(p)
                state['approx_hessian'] = torch.zeros_like(p)
                state['update'] = torch.zeros_like(p)

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

            current_lr: float = (
                group['lr']
                if group['step'] >= self.warmup_steps
                else (self.lr - group['init_lr']) * group['step'] / self.warmup_steps + group['init_lr']
            )

            weight_decay, eps = group['weight_decay'], group['eps']

            bias_correction: float = 1.0 - group['beta'] ** group['step']
            alpha: float = (1.0 - group['beta']) / bias_correction

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg_grad'] = torch.zeros_like(p)
                    state['approx_hessian'] = torch.zeros_like(p)
                    state['update'] = torch.zeros_like(p)

                exp_avg_grad, b, d_p = state['exp_avg_grad'], state['approx_hessian'], state['update']

                if weight_decay > 0.0 and self.weight_decay_type == 'l2':
                    grad.add_(p, alpha=weight_decay)

                delta_grad = grad - exp_avg_grad
                if self.rebound == 'belief':
                    rebound = delta_grad.norm(p=np.inf)
                else:
                    rebound = 1e-2
                    eps /= rebound

                exp_avg_grad.add_(delta_grad, alpha=alpha)

                de_nom = d_p.norm(p=4).add(eps)
                d_p.div_(de_nom)

                v_sq = d_p.mul(d_p)
                delta = delta_grad.div_(de_nom).mul_(d_p).sum().mul(-alpha) - b.mul(v_sq).sum()

                b.addcmul_(v_sq, delta)

                de_nom = b.abs().clamp_(min=rebound)
                if self.rebound == 'belief':
                    de_nom.add_(eps / alpha)

                d_p.copy_(exp_avg_grad.div(de_nom))

                if weight_decay > 0.0 and self.weight_decay_type != 'l2':
                    if self.weight_decay_type == 'stable':
                        weight_decay /= de_nom.mean().item()

                    d_p.add_(p, alpha=weight_decay)

                p.add_(d_p, alpha=-current_lr)

        return loss
