import math
from typing import Dict, Union

import torch
from torch.nn.functional import softsign

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import BETAS, CLOSURE, DEFAULTS, GROUP, LOSS, PARAMETERS


def update_ema(state: Dict, loss: Union[float, torch.Tensor]) -> Dict[str, float]:
    r"""Update the EMA dictionary for the `short` and `long` terms."""
    if isinstance(loss, torch.Tensor):
        loss = loss.item()

    ema = state.setdefault('ema', {})
    ema['short'] = 0.3 * loss + 0.7 * ema.get('short', loss)
    ema['long'] = 0.01 * loss + 0.99 * ema.get('long', loss)

    return ema


def compute_scalar(ema: Dict[str, float]) -> float:
    r"""Compute the difference scalar."""
    diff: float = ema['short'] - ema['long']
    return math.tanh(5.0 * diff)


def get_scalar_ratio(scalar: float, use_shadow: bool = True) -> float:
    r"""Get the scalar ratio."""
    if not use_shadow:
        return 0.0

    scalar = abs(scalar)
    if scalar > 0.6:
        return 0.6 + (scalar - 0.6) / 0.4 * 0.4
    if scalar > 0.1:
        return 0.1 + (scalar - 0.1) / 0.5 * 0.5
    return 0.0


class EmoNavi(BaseOptimizer):
    r"""An emotion-driven optimizer that feels loss and navigates accordingly.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param shadow_weight: float. the weight of the shadow.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param maximize: bool. maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
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

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'shadow_weight': shadow_weight,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'EmoNavi'

    def init_group(self, group: GROUP, **kwargs) -> None:
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
                state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
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

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                ema = update_ema(state, loss)
                scalar = compute_scalar(ema)
                ratio = get_scalar_ratio(scalar)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                if ratio > 0.0:
                    shadow = state['shadow']

                    p.lerp_(shadow, weight=ratio)
                    shadow.lerp_(p, weight=group['shadow_weight'])

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                de_nom = exp_avg_sq.sqrt().add_(group['eps'])

                p.addcdiv_(exp_avg, de_nom, value=-group['lr'])

        return loss


class EmoLynx(BaseOptimizer):
    r"""EmoLynx optimizer.

    Lynx was developed with inspiration from Lion and Tiger, which we deeply respect for their lightweight and
    intelligent design. It also integrates EmoNAVI to enhance its capabilities.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param use_shadow: bool. whether to use shadow feature.
    :param shadow_weight: float. the weight of the shadow.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param maximize: bool. maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.99),
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

        defaults: DEFAULTS = {
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

    def init_group(self, group: GROUP, **kwargs) -> None:
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
    def step(self, closure: CLOSURE = None) -> LOSS:
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

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                ema = update_ema(state, loss)
                scalar = compute_scalar(ema)
                ratio = get_scalar_ratio(scalar, use_shadow=group['use_shadow'])

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

                blended_grad = grad.mul(1.0 - beta1).add_(exp_avg, alpha=beta1).sign_()
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

                p.add_(blended_grad, alpha=-group['lr'])

        return loss


class EmoFact(BaseOptimizer):
    r"""EmoFact optimizer.

    EmoFact is inspired by AdaFactor and its VRAM-friendly design is something everyone loves.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param use_shadow: bool. whether to use shadow weights or not.
    :param shadow_weight: float. the weight of the shadow.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param maximize: bool. maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
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

        defaults: DEFAULTS = {
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

    def init_group(self, group: GROUP, **kwargs) -> None:
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
    def step(self, closure: CLOSURE = None) -> LOSS:
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

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                ema = update_ema(state, loss)
                scalar = compute_scalar(ema)
                ratio = get_scalar_ratio(scalar, use_shadow=group['use_shadow'])

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

                p.add_(update, alpha=-group['lr'])

        return loss


class EmoNeco(BaseOptimizer):
    r"""EmoNeco optimizer.

    EmoNeco was developed with inspiration from Lion, Tiger, Cautious, softsign, and EmoLynx which we deeply respect
    for their lightweight and intelligent design.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param shadow_weight: float. the weight of the shadow.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param maximize: bool. maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.99),
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

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'shadow_weight': shadow_weight,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'EmoNeco'

    def init_group(self, group: GROUP, **kwargs) -> None:
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

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
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

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                ema = update_ema(state, loss)
                scalar = compute_scalar(ema)
                ratio = get_scalar_ratio(scalar)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                if ratio > 0.0:
                    shadow = state['shadow']

                    p.lerp_(shadow, weight=ratio)
                    shadow.lerp_(p, weight=group['shadow_weight'])

                exp_avg = state['exp_avg']

                blended_grad = grad.mul(1.0 - beta1).add_(exp_avg, alpha=beta1)
                grad_norm = torch.norm(grad.float()).add_(group['eps'])

                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

                if 0.3 < scalar <= 0.5:
                    update = softsign(blended_grad).mul_(grad_norm)
                elif scalar < -0.3:
                    update = softsign(blended_grad)
                else:
                    direction = blended_grad.sign()

                    update = direction.clone()
                    update[direction != grad.sign()] = 0.0

                p.add_(update, alpha=-group['lr'])

        return loss


class EmoZeal(BaseOptimizer):
    r"""EmoZeal optimizer.

    EmoZeal is inspired by Adafactor, and EmoFact, and its VRAM-friendly design is something everyone loves.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param use_shadow: bool. whether to use shadow feature.
    :param shadow_weight: float. the weight of the shadow.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param maximize: bool. maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
        use_shadow: bool = True,
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

        defaults: DEFAULTS = {
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

    def init_group(self, group: GROUP, **kwargs) -> None:
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
    def step(self, closure: CLOSURE = None) -> LOSS:
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

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                ema = update_ema(state, loss)
                scalar = compute_scalar(ema)
                ratio = get_scalar_ratio(scalar, use_shadow=group['use_shadow'])

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

                if grad.dim() >= 2:
                    exp_avg = state['exp_avg']

                    blended_grad = grad.mul(1.0 - beta1).add_(exp_avg, alpha=beta1)
                    grad_norm = torch.norm(grad.float()).add_(group['eps'])

                    exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

                    if 0.3 < scalar <= 0.5:
                        update = softsign(blended_grad).mul_(grad_norm)
                    elif scalar < -0.3:
                        update = softsign(blended_grad)
                    else:
                        direction = blended_grad.sign()

                        update = direction.clone()
                        update[direction != grad.sign()] = 0.0

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

                p.add_(update, alpha=-group['lr'])

        return loss
