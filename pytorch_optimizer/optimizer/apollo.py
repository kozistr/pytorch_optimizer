import math
from typing import Literal, Optional

import numpy as np
import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.optimizer.galore_utils import GaLoreProjector

SCALE_TYPE = Literal['channel', 'tensor']


class ApolloDQN(BaseOptimizer):
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
        lr: float = 1e-2,
        init_lr: Optional[float] = 1e-5,
        beta: float = 0.9,
        rebound: str = 'constant',
        weight_decay: float = 0.0,
        weight_decay_type: str = 'l2',
        warmup_steps: int = 500,
        eps: float = 1e-4,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(beta, 'beta', 0.0, 1.0, range_type='[]')
        self.validate_options(rebound, 'rebound', ['constant', 'belief'])
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_options(weight_decay_type, 'weight_decay_type', ['l2', 'decoupled', 'stable'])
        self.validate_non_negative(eps, 'eps')

        self.lr = lr
        self.warmup_steps = warmup_steps
        self.init_lr: float = init_lr if init_lr is not None else lr / 1000.0

        defaults: DEFAULTS = {
            'lr': lr,
            'init_lr': self.init_lr,
            'beta': beta,
            'rebound': rebound,
            'weight_decay': weight_decay,
            'weight_decay_type': weight_decay_type,
            'eps': eps,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'ApolloDQN'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

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

            bias_correction: float = self.debias(group['beta'], group['step'])
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

                if weight_decay > 0.0 and group['weight_decay_type'] == 'l2':
                    grad.add_(p, alpha=weight_decay)

                exp_avg_grad, b, d_p = state['exp_avg_grad'], state['approx_hessian'], state['update']

                delta_grad = grad - exp_avg_grad
                if group['rebound'] == 'belief':
                    rebound = delta_grad.norm(p=np.inf)
                else:
                    rebound = 1e-2
                    eps /= rebound

                exp_avg_grad.add_(delta_grad, alpha=alpha)

                de_nom = d_p.norm(p=4).add_(eps)
                d_p.div_(de_nom)

                v_sq = d_p.mul(d_p)
                delta = delta_grad.div_(de_nom).mul_(d_p).sum().mul(-alpha) - b.mul(v_sq).sum()

                b.addcmul_(v_sq, delta)

                de_nom = b.abs().clamp_(min=rebound)
                if group['rebound'] == 'belief':
                    de_nom.add_(eps / alpha)

                d_p.copy_(exp_avg_grad.div(de_nom))

                if weight_decay > 0.0 and group['weight_decay_type'] != 'l2':
                    if group['weight_decay_type'] == 'stable':
                        weight_decay /= de_nom.mean().item()

                    d_p.add_(p, alpha=weight_decay)

                p.add_(d_p, alpha=-current_lr)

        return loss


class APOLLO(BaseOptimizer):
    r"""SGD-like Memory, AdamW-level Performance.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param correct_bias: bool. Whether to correct bias in Adam.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-2,
        betas: BETAS = (0.9, 0.999),
        scale_type: SCALE_TYPE = 'tensor',
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        correct_bias: bool = True,
        eps: float = 1e-6,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'scale_type': scale_type,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'correct_bias': correct_bias,
            'eps': eps,
            **kwargs,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'APOLLO'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

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

            beta1, beta2 = group['betas']

            step_size: float = group['lr']
            if group['correct_bias']:
                bias_correction1: float = self.debias(beta1, group['step'])
                bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))
                step_size *= bias_correction2_sq / bias_correction1

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                if 'rank' in group and p.dim() > 1:
                    if 'projector' not in state:
                        state['projector'] = GaLoreProjector(
                            rank=group['rank'],
                            update_proj_gap=group['update_proj_gap'],
                            scale=group['scale'],
                            projection_type=group['projection_type'],
                        )

                    grad = state['projector'].project(grad, group['step'], from_random_matrix=True)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                de_nom = exp_avg_sq.sqrt().add_(group['eps'])

                norm_grad = exp_avg / de_nom
                if 'rank' in group and p.dim() > 1:
                    if group['scale_type'] == 'channel':
                        norm_dim: int = 0 if norm_grad.shape[0] < norm_grad.shape[1] else 1
                        scaling_factor = torch.norm(norm_grad, dim=norm_dim) / (torch.norm(grad, dim=norm_dim) + 1e-8)
                        if norm_dim == 1:
                            scaling_factor = scaling_factor.unsqueeze(1)
                    else:
                        scaling_factor = torch.norm(norm_grad) / (torch.norm(grad) + 1e-8)

                    scaling_grad = grad * scaling_factor

                    scaling_grad_norm = torch.norm(scaling_grad)
                    if 'scaling_grad' in state:
                        limiter = (
                            max(
                                scaling_grad_norm / (state['scaling_grad'] + 1e-8),
                                1.01,
                            )
                            / 1.01
                        )

                        scaling_grad.div_(limiter)
                        scaling_grad_norm.div_(limiter)

                    state['scaling_grad'] = scaling_grad_norm

                    norm_grad = scaling_grad * np.sqrt(group['scale'])
                    norm_grad = state['projector'].project_back(norm_grad)

                p.add_(norm_grad, alpha=-step_size)

                self.apply_weight_decay(
                    p,
                    grad,
                    lr=step_size,
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

        return loss
