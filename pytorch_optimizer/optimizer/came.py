import math
from typing import Tuple

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class CAME(BaseOptimizer):
    r"""Confidence-guided Adaptive Memory Efficient Optimization.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param clip_threshold: float. threshold of root-mean-square of final gradient update.
    :param ams_bound: bool. whether to use the AMSBound variant.
    :param eps1: float. term added to the denominator to improve numerical stability.
    :param eps2: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 2e-4,
        betas: BETAS = (0.9, 0.999, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        clip_threshold: float = 1.0,
        ams_bound: bool = False,
        eps1: float = 1e-30,
        eps2: float = 1e-16,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps1, 'eps1')
        self.validate_non_negative(eps2, 'eps2')

        self.clip_threshold = clip_threshold
        self.eps1 = eps1
        self.eps2 = eps2

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'ams_bound': ams_bound,
            'eps1': eps1,
            'eps2': eps2,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'CAME'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                grad = p.grad

                grad_shape: Tuple[int, ...] = grad.shape
                factored: bool = self.get_options(grad_shape)

                state['exp_avg'] = torch.zeros_like(p)

                if factored:
                    state['exp_avg_sq_row'] = torch.zeros(grad_shape[:-1], dtype=grad.dtype, device=grad.device)
                    state['exp_avg_sq_col'] = torch.zeros(
                        grad_shape[:-2] + grad_shape[-1:], dtype=grad.dtype, device=grad.device
                    )
                    state['exp_avg_res_row'] = torch.zeros(grad_shape[:-1], dtype=grad.dtype, device=grad.device)
                    state['exp_avg_res_col'] = torch.zeros(
                        grad_shape[:-2] + grad_shape[-1:], dtype=grad.dtype, device=grad.device
                    )
                else:
                    state['exp_avg_sq'] = torch.zeros_like(grad)

                if group['ams_bound']:
                    state['exp_avg_sq_hat'] = torch.zeros_like(grad)

                state['RMS'] = 0.0

    @staticmethod
    def get_options(shape: Tuple[int, ...]) -> bool:
        r"""Get `factored`."""
        return len(shape) >= 2

    @staticmethod
    def get_rms(x: torch.Tensor) -> float:
        r"""Get RMS."""
        return x.norm(2) / math.sqrt(x.numel())

    @staticmethod
    def approximate_sq_grad(
        exp_avg_sq_row: torch.Tensor,
        exp_avg_sq_col: torch.Tensor,
        output: torch.Tensor,
    ):
        r"""Get approximation of EMA of squared gradient."""
        r_factor: torch.Tensor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor: torch.Tensor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        torch.mul(r_factor, c_factor, out=output)

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

            beta1, beta2, beta3 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                grad_shape: Tuple[int, ...] = grad.shape
                factored: bool = self.get_options(grad_shape)

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                    if factored:
                        state['exp_avg_sq_row'] = torch.zeros(grad_shape[:-1], dtype=grad.dtype, device=grad.device)
                        state['exp_avg_sq_col'] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:], dtype=grad.dtype, device=grad.device
                        )
                        state['exp_avg_res_row'] = torch.zeros(grad_shape[:-1], dtype=grad.dtype, device=grad.device)
                        state['exp_avg_res_col'] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:], dtype=grad.dtype, device=grad.device
                        )
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(grad)

                    if group['ams_bound']:
                        state['exp_avg_sq_hat'] = torch.zeros_like(grad)

                    state['RMS'] = 0.0

                state['RMS'] = self.get_rms(p)

                update = torch.mul(grad, grad).add_(self.eps1)

                if factored:
                    exp_avg_sq_row, exp_avg_sq_col = state['exp_avg_sq_row'], state['exp_avg_sq_col']

                    exp_avg_sq_row.mul_(beta2).add_(update.mean(dim=-1), alpha=1.0 - beta2)
                    exp_avg_sq_col.mul_(beta2).add_(update.mean(dim=-2), alpha=1.0 - beta2)

                    self.approximate_sq_grad(exp_avg_sq_row, exp_avg_sq_col, update)
                else:
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2).add_(update, alpha=1.0 - beta2)
                    torch.rsqrt(exp_avg_sq, out=update)

                if group['ams_bound']:
                    exp_avg_sq_hat = state['exp_avg_sq_hat']
                    torch.max(exp_avg_sq_hat, 1 / update, out=exp_avg_sq_hat)
                    torch.rsqrt(exp_avg_sq_hat / beta2, out=update)

                update.mul_(grad)

                update.div_((self.get_rms(update) / self.clip_threshold).clamp_(min=1.0))

                exp_avg = state['exp_avg']
                exp_avg.mul_(beta1).add_(update, alpha=1.0 - beta1)

                res = update - exp_avg
                res.pow_(2).add_(self.eps2)

                if factored:
                    exp_avg_res_row, exp_avg_res_col = state['exp_avg_res_row'], state['exp_avg_res_col']

                    exp_avg_res_row.mul_(beta3).add_(res.mean(dim=-1), alpha=1.0 - beta3)
                    exp_avg_res_col.mul_(beta3).add_(res.mean(dim=-2), alpha=1.0 - beta3)

                    self.approximate_sq_grad(exp_avg_res_row, exp_avg_res_col, update)
                    update.mul_(exp_avg)
                else:
                    update = exp_avg

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                update.mul_(group['lr'])

                p.add_(-update)

        return loss
