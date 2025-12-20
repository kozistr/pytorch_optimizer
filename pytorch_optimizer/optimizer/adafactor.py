import math
from typing import Optional, Tuple

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, Parameters, ParamGroup


class AdaFactor(BaseOptimizer):
    """Adaptive Learning Rates with Sublinear Memory Cost with some tweaks.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        betas (Betas): Coefficients used for computing running averages of gradient and the squared Hessian trace.
            If beta1 is None, first momentum will be skipped.
        decay_rate (float): Coefficient used to compute running averages of squared gradient.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): The optimizer uses decoupled weight decay as in AdamW.
        fixed_decay (bool): Fix weight decay.
        clip_threshold (float): Threshold of root-mean-square of final gradient update.
        ams_bound (bool): Whether to use the AMSBound variant.
        scale_parameter (bool): If True, the learning rate is scaled by root-mean-square of parameter.
        relative_step (bool): If True, time-dependent learning rate is computed instead of external learning rate.
        warmup_init (bool): Time-dependent learning rate computation depends on whether warm-up initialization is
            being used.
        eps1 (float): Term added to the denominator to improve numerical stability.
        eps2 (float): Term added to the denominator to improve numerical stability.
        momentum_dtype (torch.dtype): Type of momentum variable. In the ViT paper, it was observed that storing
            momentum in half-precision (bfloat16 type) does not affect training dynamics and reduces optimizer
            overhead from 2-fold to 1.5-fold.
        maximize (bool): Maximize the objective with respect to the parameters, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: Optional[float] = 1e-3,
        betas: Betas = (0.9, 0.999),
        decay_rate: float = -0.8,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        clip_threshold: float = 1.0,
        ams_bound: bool = False,
        scale_parameter: bool = True,
        relative_step: bool = True,
        warmup_init: bool = False,
        eps1: float = 1e-30,
        eps2: float = 1e-3,
        momentum_dtype: torch.dtype = torch.bfloat16,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps1, 'eps1')
        self.validate_non_negative(eps2, 'eps2')

        self.decay_rate = decay_rate
        self.clip_threshold = clip_threshold
        self.eps1 = eps1
        self.eps2 = eps2
        self.momentum_dtype = momentum_dtype
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'ams_bound': ams_bound,
            'scale_parameter': scale_parameter,
            'relative_step': relative_step,
            'warmup_init': warmup_init,
            'eps1': eps1,
            'eps2': eps2,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdaFactor'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        beta1: float = kwargs.get('beta1', 0.9)

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]

            grad_shape: Tuple[int, ...] = grad.shape
            factored: bool = self.get_options(grad_shape)

            if len(state) == 0:
                if beta1 is not None:
                    state['exp_avg'] = torch.zeros_like(p, dtype=self.momentum_dtype)

                if factored:
                    state['exp_avg_sq_row'] = torch.zeros(grad_shape[:-1], dtype=grad.dtype, device=grad.device)
                    state['exp_avg_sq_col'] = torch.zeros(
                        grad_shape[:-2] + grad_shape[-1:], dtype=grad.dtype, device=grad.device
                    )
                else:
                    state['exp_avg_sq'] = torch.zeros_like(grad)

                if group['ams_bound']:
                    state['exp_avg_sq_hat'] = torch.zeros_like(grad)

                state['RMS'] = 0.0

    def get_lr(
        self, lr: float, step: int, rms: float, relative_step: bool, warmup_init: bool, scale_parameter: bool
    ) -> float:
        r"""Get AdaFactor learning rate."""
        relative_step_size: float = lr
        if relative_step:
            min_step: float = 1e-6 * step if warmup_init else 1e-2
            relative_step_size = min(min_step, 1.0 / math.sqrt(step))

        param_scale: float = 1.0 if scale_parameter else max(self.eps2, rms)

        return param_scale * relative_step_size

    @staticmethod
    def get_options(shape: Tuple[int, ...]) -> bool:
        r"""Get `factored`."""
        return len(shape) >= 2

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']

            self.init_group(group, beta1=beta1)
            group['step'] += 1

            beta2_t: float = 1.0 - math.pow(group['step'], self.decay_rate)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                grad_shape: Tuple[int, ...] = grad.shape
                factored: bool = self.get_options(grad_shape)

                state['RMS'] = self.get_rms(p)

                lr: float = self.get_lr(
                    lr=group['lr'],
                    step=group['step'],
                    rms=state['RMS'],
                    relative_step=group['relative_step'],
                    warmup_init=group['warmup_init'],
                    scale_parameter=group['scale_parameter'],
                )

                update = torch.mul(grad, grad).add_(self.eps1)

                if factored:
                    exp_avg_sq_row, exp_avg_sq_col = state['exp_avg_sq_row'], state['exp_avg_sq_col']

                    exp_avg_sq_row.mul_(beta2_t).add_(update.mean(dim=-1), alpha=1.0 - beta2_t)
                    exp_avg_sq_col.mul_(beta2_t).add_(update.mean(dim=-2), alpha=1.0 - beta2_t)

                    self.approximate_sq_grad(exp_avg_sq_row, exp_avg_sq_col, update)
                else:
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2_t).add_(update, alpha=1.0 - beta2_t)
                    exp_avg_sq.clamp_(max=beta2)

                    torch.rsqrt(exp_avg_sq, out=update)

                if group['ams_bound']:
                    exp_avg_sq_hat = state['exp_avg_sq_hat']
                    torch.max(exp_avg_sq_hat, 1 / update, out=exp_avg_sq_hat)
                    torch.rsqrt(exp_avg_sq_hat / beta2_t, out=update)

                update.mul_(grad)

                update.div_((self.get_rms(update) / self.clip_threshold).clamp_(min=1.0)).mul_(lr)

                if beta1 is not None:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(beta1).add_(update, alpha=1.0 - beta1)

                    update = exp_avg.clone()
                    if group.get('cautious'):
                        self.apply_cautious(update, grad)

                self.apply_weight_decay(
                    p=p,
                    grad=None,
                    lr=lr,
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                p.add_(-update)

        return loss
