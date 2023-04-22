import math
from typing import Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class AdaFactor(Optimizer, BaseOptimizer):
    r"""Adaptive Learning Rates with Sublinear Memory Cost.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param decay_rate: float. coefficient used to compute running averages of square gradient.
    :param weight_decay: float. weight decay (L2 penalty).
    :param clip_threshold: float. threshold of root-mean-square of final gradient update.
    :param scale_parameter: bool. if true, learning rate is scaled by root-mean-square of parameter.
    :param relative_step: bool. if true, time-dependent learning rate is computed instead of external learning rate.
    :param warmup_init: bool. time-dependent learning rate computation depends on
        whether warm-up initialization is being used.
    :param eps1: float. term added to the denominator to improve numerical stability.
    :param eps2: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: Optional[float] = 1e-3,
        betas: BETAS = (0.9, 0.999),
        decay_rate: float = -0.8,
        weight_decay: float = 0.0,
        clip_threshold: float = 1.0,
        scale_parameter: bool = True,
        relative_step: bool = True,
        warmup_init: bool = False,
        eps1: float = 1e-30,
        eps2: float = 1e-3,
    ):
        self.lr = lr
        self.betas = betas
        self.decay_rate = decay_rate
        self.weight_decay = weight_decay
        self.clip_threshold = clip_threshold
        self.scale_parameter = scale_parameter
        self.relative_step = relative_step
        self.warmup_init = warmup_init
        self.eps1 = eps1
        self.eps2 = eps2

        self.validate_parameters()

        defaults: DEFAULTS = {'lr': lr, 'weight_decay': weight_decay}
        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_betas(self.betas)
        self.validate_weight_decay(self.weight_decay)
        self.validate_epsilon(self.eps1)
        self.validate_epsilon(self.eps2)

    def __str__(self) -> str:
        return 'AdaFactor'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                grad = p.grad

                grad_shape: Tuple[int, ...] = grad.shape
                factored: bool = self.get_options(grad_shape)

                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)

                if factored:
                    state['exp_avg_sq_row'] = torch.zeros(grad_shape[:-1]).to(grad)
                    state['exp_avg_sq_col'] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
                else:
                    state['exp_avg_sq'] = torch.zeros_like(grad)

                state['RMS'] = 0.0

    def get_lr(self, lr: float, step: int, rms: float) -> float:
        relative_step_size: float = lr
        if self.relative_step:
            min_step: float = 1e-6 * step if self.warmup_init else 1e-2
            relative_step_size = min(min_step, 1.0 / math.sqrt(step))

        param_scale: float = 1.0 if self.scale_parameter else max(self.eps2, rms)

        return param_scale * relative_step_size

    @staticmethod
    def get_options(shape: Tuple[int, ...]) -> bool:
        r"""Get `factored`."""
        return len(shape) >= 2

    @staticmethod
    def get_rms(x: torch.Tensor) -> float:
        r"""Get RMS."""
        return x.norm(2) / (x.numel() ** 0.5)

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
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)

                    if factored:
                        state['exp_avg_sq_row'] = torch.zeros(grad_shape[:-1], dtype=grad.dtype, device=grad.device)
                        state['exp_avg_sq_col'] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:], dtype=grad.dtype, device=grad.device
                        )
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(grad)

                    state['RMS'] = 0.0

                state['step'] += 1

                state['RMS'] = self.get_rms(p)
                lr = self.get_lr(group['lr'], state['step'], state['RMS'])

                beta2_t: float = 1.0 - math.pow(state['step'], self.decay_rate)
                update = torch.mul(grad, grad).add_(self.eps1)

                if factored:
                    exp_avg_sq_row, exp_avg_sq_col = state['exp_avg_sq_row'], state['exp_avg_sq_col']

                    exp_avg_sq_row.mul_(beta2_t).add_(update.mean(dim=-1), alpha=1.0 - beta2_t)
                    exp_avg_sq_col.mul_(beta2_t).add_(update.mean(dim=-2), alpha=1.0 - beta2_t)

                    self.approximate_sq_grad(exp_avg_sq_row, exp_avg_sq_col, output=update)
                else:
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2_t).add_(update, alpha=1.0 - beta2_t)
                    torch.rsqrt(exp_avg_sq, out=update)

                update.mul_(grad)

                # TODO: implement AMSGrad

                update.div_((self.get_rms(update) / self.clip_threshold).clamp_(min=1.0)).mul_(lr)

                exp_avg = state['exp_avg']
                exp_avg.mul_(self.betas[0]).add_(update, alpha=1.0 - self.betas[0])

                update = exp_avg

                if group['weight_decay'] > 0.0:
                    p.add_(p, alpha=-lr * group['weight_decay'])

                p.add_(-update)

        return loss
