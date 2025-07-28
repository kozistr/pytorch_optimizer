import math
from typing import Dict, Union

import torch

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


def get_scalar_ratio(scalar: float) -> float:
    r"""Get the scalar ratio."""
    if scalar > 0.6:
        return 0.7 + 0.2 * scalar
    if scalar < -0.6:
        return 0.1
    if abs(scalar) > 0.3:
        return 0.3
    return 0.0


class EmoNavi(BaseOptimizer):
    r"""An emotion-driven optimizer that feels loss and navigates accordingly.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
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
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
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
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = 0.0

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

                shadow, exp_avg, exp_avg_sq = state['shadow'], state['exp_avg'], state['exp_avg_sq']

                if ratio > 0.0:
                    p.lerp_(shadow, weight=ratio)
                    shadow.lerp_(p, weight=0.05)

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
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
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
                state['shadow'] = p.clone()
                state['exp_avg'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = 0.0

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

                shadow, exp_avg = state['shadow'], state['exp_avg']

                if ratio > 0.0:
                    p.lerp_(shadow, weight=ratio)
                    shadow.lerp_(p, weight=0.05)

                blended_grad = grad.mul(1.0 - beta1).add_(exp_avg, alpha=beta1)
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

                p.add_(blended_grad.sign_(), alpha=-group['lr'])

        return loss


class EmoFact(BaseOptimizer):
    r"""EmoFact optimizer.

    EmoFact is inspired by AdaFactor and its VRAM-friendly design is something everyone loves.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
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
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
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
                state['shadow'] = p.clone()

                shape = p.size()

                if len(shape) >= 2:
                    r_shape = [s if i == 0 else 1 for i, s in enumerate(shape)]
                    state['exp_avg_r'] = torch.zeros(r_shape, dtype=p.dtype, device=p.device)

                    c_shape = [1 if i == 0 else s for i, s in enumerate(shape)]
                    state['exp_avg_c'] = torch.zeros(c_shape, dtype=p.dtype, device=p.device)
                else:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = 0.0

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
                    shadow.lerp_(p, weight=0.05)

                if grad.dim() >= 2:
                    exp_avg_r, exp_avg_c = state['exp_avg_r'], state['exp_avg_c']

                    grad_p2 = grad.pow(2)
                    r_sq = torch.mean(grad_p2, dim=tuple(range(1, grad.dim())), keepdim=True).add_(group['eps'])
                    c_sq = torch.mean(grad_p2, dim=0, keepdim=True).add_(group['eps'])

                    exp_avg_r.mul_(beta1).add_(r_sq.sqrt(), alpha=1.0 - beta1)
                    exp_avg_c.mul_(beta1).add_(c_sq.sqrt(), alpha=1.0 - beta1)

                    de_nom = torch.sqrt(exp_avg_r * exp_avg_c).add_(group['eps'])

                    update = grad / de_nom
                else:
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    de_nom = exp_avg_sq.sqrt().add_(group['eps'])

                    update = exp_avg / de_nom

                p.add_(update, alpha=-group['lr'])

        return loss
