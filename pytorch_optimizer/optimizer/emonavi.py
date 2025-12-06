import math
from typing import Dict, Optional, Union

import torch
from torch.nn.functional import softsign

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, Parameters, ParamGroup


def update_ema(state: Dict, loss: Union[float, torch.Tensor]) -> Dict[str, float]:
    """Update the EMA dictionary for the `short`, `medium`, and `long` terms."""
    if isinstance(loss, torch.Tensor):
        loss = loss.item()

    ema = state.setdefault('ema', {})
    ema['short'] = 0.3 * loss + 0.7 * ema.get('short', loss)
    ema['medium'] = 0.05 * loss + 0.95 * ema.get('medium', loss)
    ema['long'] = 0.01 * loss + 0.99 * ema.get('long', loss)

    return ema


def compute_scalar(ema: Dict[str, float]) -> float:
    """Compute the difference scalar."""
    scale_base_l = max(ema['long'], 1e-5)
    scale_base_m = max(ema['medium'], 1e-5)

    diff = ema['long'] - ema['short']
    diff_l = diff / scale_base_l
    diff_m = diff / scale_base_m

    if abs(diff_l) < 0.05:
        return math.tanh(diff_l)

    return math.tanh(diff_m) if abs(diff_m) * scale_base_m < abs(diff_l) * scale_base_l else math.tanh(diff_l)


def get_coef(scalar: float) -> float:
    """Get the scalar coefficient."""
    abs_scaler = abs(scalar)
    if abs_scaler > 0.625:
        return 1.0 - abs_scaler
    if abs_scaler > 0.125:
        return 1.0 + scalar
    return 1.0


def get_scalar_ratio(scalar: float, use_shadow: bool) -> float:
    """Get the scalar ratio."""
    if not use_shadow:
        return 0.0

    scalar = abs(scalar)
    if scalar > 0.75:
        return 0.75
    if scalar > 0.25:
        return -0.1
    return 0.0


class EmoNavi(BaseOptimizer):
    """An emotion-driven optimizer that feels loss and navigates accordingly.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        lr_max (float): maximum learning rate.
        lr_min (float): minimum learning rate.
        betas (Betas): Coefficients used for computing running averages of gradient and the squared Hessian trace.
        use_shadow (bool): Whether to use shadowing or not.
        shadow_weight (float): The weight of the shadow.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): The optimizer uses decoupled weight decay as in AdamW.
        fixed_decay (bool): Fix weight decay.
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the objective with respect to the parameters, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        lr_max: float = 1e-3,
        lr_min: float = 1e-8,
        betas: Betas = (0.9, 0.999),
        use_shadow: bool = False,
        shadow_weight: float = 0.05,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_learning_rate(lr_max)
        self.validate_learning_rate(lr_min)
        self.validate_betas(betas)
        self.validate_range(shadow_weight, 'shadow_weight', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.use_shadow = use_shadow
        self.maximize = maximize

        self.lr = lr
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.k: float = 0.2
        self.prev_loss: Optional[float] = None

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'use_shadow': use_shadow,
            'shadow_weight': shadow_weight,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'EmoNavi'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]

            if len(state) == 0:
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

                if group['use_shadow']:
                    state['shadow'] = p.clone()

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss = 0.0
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

            ema = update_ema(self.state, loss)
            scalar = compute_scalar(ema)
            coef = get_coef(scalar)
            ratio = get_scalar_ratio(scalar, group['use_shadow'])

            if self.prev_loss is not None:
                delta = self.prev_loss - loss
                target_delta = max(1e-8, 0.01 * max(loss, 1e-8))

                self.lr *= math.exp(self.k * (delta - target_delta) / (abs(target_delta) + group['eps']))

            step_size: float = max(self.lr_min, min(self.lr_max, self.lr * coef))

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=step_size,
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                if group['use_shadow']:
                    shadow = state['shadow']

                    if ratio > 0.0:
                        p.lerp_(shadow, weight=coef)
                        shadow.lerp_(p, weight=group['shadow_weight'])
                    else:
                        leap_ratio: float = 0.1 if ratio < 0.0 else 0.1 * coef
                        shadow.lerp_(p, weight=leap_ratio)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                de_nom = exp_avg_sq.sqrt().add_(group['eps'])

                p.addcdiv_(exp_avg, de_nom, value=-step_size)

        self.prev_loss = loss

        return loss


class EmoLynx(BaseOptimizer):
    """EmoLynx optimizer.

    Lynx was developed with inspiration from Lion and Tiger, which we deeply respect for their lightweight and
    intelligent design. It also integrates EmoNAVI to enhance its capabilities.

    Args:
        params (Parameters): Iterable of parameters to optimize, or dicts defining parameter groups.
        lr (float): Learning rate.
        lr_max (float): maximum learning rate.
        lr_min (float): minimum learning rate.
        betas (Betas): Coefficients used for computing running averages of gradient and the squared hessian trace.
        use_shadow (bool): Whether to use shadow feature.
        shadow_weight (float): The weight of the shadow.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): The optimizer uses decoupled weight decay as in AdamW.
        fixed_decay (bool): Fix weight decay.
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        lr_max: float = 1e-3,
        lr_min: float = 1e-8,
        betas: Betas = (0.9, 0.99),
        use_shadow: bool = False,
        shadow_weight: float = 0.05,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_range(shadow_weight, 'shadow_weight', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        self.lr = lr
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.k: float = 0.2
        self.prev_loss: Optional[float] = None

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'use_shadow': use_shadow,
            'shadow_weight': shadow_weight,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'EmoLynx'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]

            if len(state) == 0:
                if group['use_shadow']:
                    state['shadow'] = p.clone()
                state['exp_avg'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss = 0.0
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

            ema = update_ema(self.state, loss)
            scalar = compute_scalar(ema)
            coef = get_coef(scalar)
            ratio = get_scalar_ratio(scalar, group['use_shadow'])

            if self.prev_loss is not None:
                delta = self.prev_loss - loss
                target_delta = max(1e-8, 0.01 * max(loss, 1e-8))

                self.lr *= math.exp(self.k * (delta - target_delta) / (abs(target_delta) + group['eps']))

            step_size: float = max(self.lr_min, min(self.lr_max, self.lr * coef))

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=step_size,
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                if group['use_shadow']:
                    shadow = state['shadow']
                    if ratio > 0.0:
                        p.lerp_(shadow, weight=ratio)
                        shadow.lerp_(p, weight=group['shadow_weight'])
                    else:
                        leap_ratio = 0.1 if ratio < 0 else 0.1 * coef
                        shadow.lerp_(p, weight=leap_ratio)

                exp_avg = state['exp_avg']

                blended_grad = grad.mul(1.0 - beta1).add_(exp_avg, alpha=beta1).sign_()
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

                p.add_(blended_grad, alpha=-step_size)

        self.prev_loss = loss

        return loss


class EmoFact(BaseOptimizer):
    """EmoFact optimizer.

    EmoFact is inspired by AdaFactor and its VRAM-friendly design is something everyone loves.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        lr_max (float): maximum learning rate.
        lr_min (float): minimum learning rate.
        betas (Betas): Coefficients used for computing running averages of gradient and the squared Hessian trace.
        use_shadow (bool): Whether to use shadow weights or not.
        shadow_weight (float): The weight of the shadow.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): The optimizer uses decoupled weight decay as in AdamW.
        fixed_decay (bool): Fix weight decay.
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the objective with respect to the parameters, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        lr_max: float = 1e-3,
        lr_min: float = 1e-8,
        betas: Betas = (0.9, 0.999),
        use_shadow: bool = False,
        shadow_weight: float = 0.05,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_learning_rate(lr_max)
        self.validate_learning_rate(lr_min)
        self.validate_betas(betas)
        self.validate_range(shadow_weight, 'shadow_weight', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        self.lr = lr
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.k: float = 0.2
        self.prev_loss: Optional[float] = None

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'use_shadow': use_shadow,
            'shadow_weight': shadow_weight,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'EmoFact'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]

            if len(state) == 0:
                if group['use_shadow']:
                    state['shadow'] = p.clone()

                shape = p.size()

                if len(shape) >= 2:
                    r_shape = [shape[0]] + [1] * (len(shape) - 1)
                    state['exp_avg_r'] = torch.zeros(r_shape, dtype=p.dtype, device=p.device)

                    c_shape = [1, *list(shape[1:])]
                    state['exp_avg_c'] = torch.zeros(c_shape, dtype=p.dtype, device=p.device)
                else:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss = 0.0
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

            ema = update_ema(self.state, loss)
            scalar = compute_scalar(ema)
            coef = get_coef(scalar)
            ratio = get_scalar_ratio(scalar, group['use_shadow'])

            if self.prev_loss is not None:
                delta = self.prev_loss - loss
                target_delta = max(1e-8, 0.01 * max(loss, 1e-8))

                self.lr *= math.exp(self.k * (delta - target_delta) / (abs(target_delta) + group['eps']))

            step_size: float = max(self.lr_min, min(self.lr_max, self.lr * coef))

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=step_size,
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                if group['use_shadow']:
                    shadow = state['shadow']
                    if ratio > 0.0:
                        p.lerp_(shadow, weight=ratio)
                        shadow.lerp_(p, weight=group['shadow_weight'])
                    else:
                        leap_ratio = 0.1 if ratio < 0 else 0.1 * coef
                        shadow.lerp_(p, weight=leap_ratio)

                if grad.dim() >= 2:
                    exp_avg_r, exp_avg_c = state['exp_avg_r'], state['exp_avg_c']

                    grad_p2 = grad.pow(2)
                    r_sq = (
                        torch.mean(grad_p2, dim=tuple(range(1, grad.dim())), keepdim=True).add_(group['eps']).sqrt_()
                    )
                    c_sq = torch.mean(grad_p2, dim=0, keepdim=True).add_(group['eps']).sqrt_()

                    exp_avg_r.mul_(beta1).add_(r_sq, alpha=1.0 - beta1)
                    exp_avg_c.mul_(beta1).add_(c_sq, alpha=1.0 - beta1)

                    de_nom = (exp_avg_r * exp_avg_c).sqrt_().add_(group['eps'])

                    update = grad / de_nom
                else:
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    de_nom = exp_avg_sq.sqrt().add_(group['eps'])

                    update = exp_avg / de_nom

                p.add_(update, alpha=-step_size)

        self.prev_loss = loss

        return loss


class EmoNeco(BaseOptimizer):
    """EmoNeco optimizer.

    EmoNeco was developed with inspiration from Lion, Tiger, Cautious, softsign, and EmoLynx which we deeply respect
    for their lightweight and intelligent design.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        betas (Betas): Coefficients used for computing running averages of gradient and the squared Hessian trace.
        use_shadow (bool): Whether to use shadow weights or not.
        shadow_weight (float): The weight of the shadow.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): The optimizer uses decoupled weight decay as in AdamW.
        fixed_decay (bool): Fix weight decay.
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the objective with respect to the parameters, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.99),
        use_shadow: bool = False,
        shadow_weight: float = 0.05,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_range(shadow_weight, 'shadow_weight', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'use_shadow': use_shadow,
            'shadow_weight': shadow_weight,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'EmoNeco'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]

            if len(state) == 0:
                if group['use_shadow']:
                    state['shadow'] = p.clone()
                state['exp_avg'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss = 0.0
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

            ema = update_ema(self.state, loss)
            scalar = compute_scalar(ema)
            ratio = get_scalar_ratio(scalar, use_shadow=group['use_shadow'])

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                if group['use_shadow'] and ratio > 0.0:
                    shadow = state['shadow']

                    p.lerp_(shadow, weight=ratio)
                    shadow.lerp_(p, weight=group['shadow_weight'])

                exp_avg = state['exp_avg']

                blended_grad = grad.mul(1.0 - beta1).add_(exp_avg, alpha=beta1)
                grad_norm = torch.norm(grad.float()).add_(group['eps'])

                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

                scalar = abs(scalar)

                if 0.2 < scalar <= 0.5:
                    update = softsign(blended_grad).mul_(grad_norm * (1.0 - scalar))
                else:
                    direction = blended_grad.sign()

                    update = direction.clone()
                    update[direction != grad.sign()] = 1.0 - scalar

                p.add_(update, alpha=-group['lr'])

        return loss


class EmoZeal(BaseOptimizer):
    """EmoZeal optimizer.

    EmoZeal is inspired by Adafactor, and EmoFact, and its VRAM-friendly design is something everyone loves.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        betas (Betas): Coefficients used for computing running averages of gradient and the squared Hessian trace.
        use_shadow (bool): Whether to use shadow feature.
        shadow_weight (float): The weight of the shadow.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): The optimizer uses decoupled weight decay as in AdamW.
        fixed_decay (bool): Fix weight decay.
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the objective with respect to the parameters, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        use_shadow: bool = False,
        shadow_weight: float = 0.05,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_range(shadow_weight, 'shadow_weight', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        self.alpha_prev: float = 1.0

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'use_shadow': use_shadow,
            'shadow_weight': shadow_weight,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'EmoZeal'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]

            if len(state) == 0:
                state['shadow'] = p.clone()
                state['exp_avg'] = torch.zeros_like(p)

                shape = p.size()

                if len(shape) >= 2:
                    r_shape = [shape[0]] + [1] * (len(shape) - 1)
                    state['exp_avg_r'] = torch.zeros(r_shape, dtype=p.dtype, device=p.device)

                    c_shape = [1, *list(shape[1:])]
                    state['exp_avg_c'] = torch.zeros(c_shape, dtype=p.dtype, device=p.device)
                else:
                    state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss = 0.0
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

            ema = update_ema(self.state, loss)
            scalar = compute_scalar(ema)
            ratio = get_scalar_ratio(scalar, use_shadow=group['use_shadow'])

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                if group['use_shadow'] and ratio > 0.0:
                    shadow = state['shadow']

                    p.lerp_(shadow, weight=ratio)
                    shadow.lerp_(p, weight=group['shadow_weight'])

                scalar = abs(scalar)

                if grad.dim() >= 2:
                    exp_avg = state['exp_avg']

                    blended_grad = grad.mul(1.0 - beta1).add_(exp_avg, alpha=beta1)
                    grad_norm = torch.norm(grad.float()).add_(group['eps'])

                    exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

                    if scalar > 0.1:
                        if scalar > 0.6:
                            direction = blended_grad.sign()

                            update = direction.clone()
                            update[direction != grad.sign()] = 1.0 - scalar
                        else:
                            update = softsign(blended_grad)
                            update.mul_(grad_norm * (1.0 - scalar))

                        p.add_(update, alpha=-group['lr'])

                    exp_avg_r, exp_avg_c = state['exp_avg_r'], state['exp_avg_c']

                    grad_p2 = grad.pow(2)
                    r_sq = (
                        torch.mean(grad_p2, dim=tuple(range(1, grad.dim())), keepdim=True).add_(group['eps']).sqrt_()
                    )
                    c_sq = torch.mean(grad_p2, dim=0, keepdim=True).add_(group['eps']).sqrt_()

                    exp_avg_r.mul_(beta1).add_(r_sq, alpha=1.0 - beta1)
                    exp_avg_c.mul_(beta1).add_(c_sq, alpha=1.0 - beta1)

                    de_nom = (exp_avg_r * exp_avg_c).sqrt_().add_(group['eps'])

                    update = grad / de_nom
                else:
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    de_nom = exp_avg_sq.sqrt().add_(group['eps'])

                    update = exp_avg / de_nom

                p.add_(update, alpha=-group['lr'] * (1.0 - scalar))

        return loss
