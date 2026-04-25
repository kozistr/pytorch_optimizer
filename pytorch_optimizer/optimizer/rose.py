from typing import Union

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils import copy_stochastic


class ROSE(BaseOptimizer):
    """Range-Of-Slice Equilibration optimizer.

    Args:
        params (ParamsT): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        weight_decay (float): Weight decay (L2 penalty).
        wd_schedule (Union[float, bool]): Schedule-Coupled Weight Decay. If `False`, standard decoupled weight decay is
            used. If `True`, `lr_ref` is the first available among `group['max_lr']`, `group['initial_lr']`, and the
            learning-rate passed at construction time. If a float is provided, it is used directly as `lr_ref`.
        weight_decouple (bool): The optimizer uses decoupled weight decay as in AdamW.
        fixed_decay (bool): Fix weight decay.
        centralize (bool): Gradient Centralization. Removes shared offsets from gradient slices before the range
            computation. This can improve generalization and training stability. Biases and other 1D parameters are not
            centralized.
        stabilize (bool): Coefficient-of-Variation Trust Gating. Computes a trust factor from the coefficient of
            variation of the per-slice range tensor, and then interpolates between the local range and a smoother
            global mean denominator. This can smooth noisy gradients.
        bf16_sr (bool): Stochastic Rounding for BFloat16.
        maximize (bool): Maximize the objective with respect to the params, instead of minimizing.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        wd_schedule: Union[bool, float] = False,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        centralize: bool = True,
        stabilize: bool = True,
        bf16_sr: bool = True,
        compute_dtype: torch.dtype = torch.float64,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(weight_decay, 'weight_decay')

        self.maximize = maximize

        if bf16_sr and compute_dtype not in (torch.float32, torch.float64, None):
            raise ValueError(f'bf16_sr=True has no useful effect when compute_dtype is {compute_dtype}.')

        defaults: Defaults = {
            'lr': lr,
            'weight_decay': weight_decay,
            'wd_schedule': wd_schedule,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'centralize': centralize,
            'stabilize': stabilize,
            'bf16_sr': bf16_sr,
            'compute_dtype': compute_dtype,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'ROSE'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            lr = group['lr']
            weight_decay, wd_schedule = group['weight_decay'], group['wd_schedule']
            compute_dtype = group['compute_dtype']

            if weight_decay and wd_schedule:
                wd_lr = lr / (
                    wd_schedule if isinstance(wd_schedule, float) else group.get('max_lr', group.get('initial_lr'))
                )
            else:
                wd_lr = lr

            for p in group['params']:
                if p.grad is None:
                    continue

                use_bf16_sr = group['bf16_sr'] and p.dtype is torch.bfloat16
                fp32 = use_bf16_sr and not compute_dtype

                grad = p.grad.to(dtype=torch.float32 if fp32 else compute_dtype)
                param = p.to(dtype=torch.float32 if fp32 else compute_dtype)

                self.apply_weight_decay(
                    p,
                    grad,
                    lr=wd_lr,
                    weight_decay=weight_decay,
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                if grad.ndim == 0:
                    param.add_(grad.sign(), alpha=-lr)
                elif grad.ndim == 1:
                    g_min, g_max = grad.aminmax()
                    de_nom = g_max.abs_().sub_(g_min)

                    de_nom.masked_fill_(de_nom == 0.0, 1.0)
                    param.addcdiv_(grad, de_nom, value=-lr)
                else:
                    active_axes = tuple(range(1, grad.ndim))

                    if group['centralize']:
                        if grad is not p.grad:
                            grad.sub_(grad.mean(dim=active_axes, keepdim=True))
                        else:
                            grad = grad.sub(grad.mean(dim=active_axes, keepdim=True))

                    raw_scale = (
                        grad.amax(dim=active_axes, keepdim=True).abs_().sub_(grad.amin(dim=active_axes, keepdim=True))
                    )

                    if group['stabilize']:
                        std, mean = torch.std_mean(raw_scale, correction=0)

                        trust = mean.div(std.add_(mean).masked_fill_(mean == 0.0, 1.0))

                        de_nom = mean.lerp(raw_scale, trust)
                    else:
                        de_nom = raw_scale

                    de_nom.masked_fill_(de_nom == 0.0, 1.0)
                    param.addcdiv_(grad, de_nom, value=-lr)

                if use_bf16_sr:
                    param = param.to(dtype=torch.float32)

                    copy_stochastic(p, param)
                elif param is not p:
                    p.copy_(param)

        return loss
