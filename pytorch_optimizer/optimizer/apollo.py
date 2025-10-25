import math
from typing import Literal, Optional

import numpy as np
import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, Parameters, ParamGroup
from pytorch_optimizer.optimizer.galore_utils import GaLoreProjector

SCALE_TYPE = Literal['channel', 'tensor']


class ApolloDQN(BaseOptimizer):
    """An Adaptive Parameter-wise Diagonal Quasi-Newton Method for Nonconvex Stochastic Optimization.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        init_lr (Optional[float]): Initial learning rate (default lr / 1000).
        beta (float): Coefficient used for computing running averages of gradient.
        rebound (str): Rectified bound for diagonal Hessian. Options: 'constant', 'belief'.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decay_type (str): Type of weight decay. Options: 'l2', 'decoupled', 'stable'.
        warmup_steps (int): Number of warmup steps.
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the objective with respect to the parameters, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-2,
        init_lr: Optional[float] = 1e-5,
        beta: float = 0.9,
        rebound: str = 'constant',
        weight_decay: float = 0.0,
        weight_decay_type: str = 'l2',
        warmup_steps: int = 500,
        eps: float = 1e-4,
        maximize: bool = False,
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
        self.maximize = maximize

        defaults: Defaults = {
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

    def init_group(self, group: ParamGroup, **kwargs) -> None:
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

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                exp_avg_grad, b, d_p = state['exp_avg_grad'], state['approx_hessian'], state['update']

                p, grad, exp_avg_grad, b, d_p = self.view_as_real(p, grad, exp_avg_grad, b, d_p)

                if weight_decay > 0.0 and group['weight_decay_type'] == 'l2':
                    grad.add_(p, alpha=weight_decay)

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
    """SGD-like Memory, AdamW-level Performance.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        betas: Coefficients used for computing running averages of gradient and the squared Hessian trace.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): Whether to use decoupled weight decay as in AdamW.
        fixed_decay (bool): Apply fixed weight decay instead of adaptive.
        correct_bias (bool): Whether to correct bias in Adam.
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the objective with respect to the parameters, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-2,
        betas: Betas = (0.9, 0.999),
        scale_type: SCALE_TYPE = 'tensor',
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        correct_bias: bool = True,
        eps: float = 1e-6,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
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

    def init_group(self, group: ParamGroup, **kwargs) -> None:
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

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                p, grad, exp_avg, exp_avg_sq = self.view_as_real(p, grad, exp_avg, exp_avg_sq)

                if 'rank' in group and p.dim() > 1:
                    if 'projector' not in state:
                        state['projector'] = GaLoreProjector(
                            rank=group['rank'],
                            update_proj_gap=group['update_proj_gap'],
                            scale=group['scale'],
                            projection_type=group['projection_type'],
                        )

                    grad = state['projector'].project(grad, group['step'], from_random_matrix=True)

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
