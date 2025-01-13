import math
from typing import Optional

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class Prodigy(BaseOptimizer):
    r"""An Expeditiously Adaptive Parameter-Free Learner.

        Leave LR set to 1 unless you encounter instability.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. betas.
    :param beta3: float. coefficients for computing the Prodigy step-size using running averages. If set to None,
        uses the value of square root of beta2.
    :param d0: float. initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
    :param d_coef: float. Coefficient in the expression for the estimate of d.
    :param growth_rate: float. prevent the D estimate from growing faster than this multiplicative rate.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. use AdamW style weight decay.
    :param fixed_decay: bool. fix weight decay.
    :param bias_correction: bool. turn on Adam's bias correction.
    :param safeguard_warmup: bool. remove lr from the denominator of D estimate to avoid issues during warm-up stage.
    :param eps: float. term added to the denominator to improve numerical stability. when eps is None, use atan2 rather
        than epsilon and division for parameter updates.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1.0,
        betas: BETAS = (0.9, 0.999),
        beta3: Optional[float] = None,
        d0: float = 1e-6,
        d_coef: float = 1.0,
        growth_rate: float = float('inf'),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        bias_correction: bool = False,
        safeguard_warmup: bool = False,
        eps: Optional[float] = 1e-8,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas((*betas, beta3))
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'beta3': beta3,
            'd': d0,
            'd0': d0,
            'd_max': d0,
            'd_coef': d_coef,
            'growth_rate': growth_rate,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'bias_correction': bias_correction,
            'safeguard_warmup': safeguard_warmup,
            'step': 1,
            'eps': eps,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Prodigy'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 1
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                state['s'] = torch.zeros_like(p)
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        device = group['params'][0].device

        d_de_nom = torch.tensor([0.0], device=device)

        beta1, beta2 = group['betas']
        beta3 = group['beta3'] if group['beta3'] is not None else math.sqrt(beta2)

        bias_correction1: float = self.debias(beta1, group['step'])
        bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))
        bias_correction: float = (bias_correction1 / bias_correction2_sq) if group['bias_correction'] else 1.0

        d, d0 = group['d'], group['d0']
        d_lr: float = d * group['lr'] / bias_correction

        if 'd_numerator' not in group:
            group['d_numerator'] = torch.tensor([0.0], device=device)
        elif group['d_numerator'].device != device:
            group['d_numerator'] = group['d_numerator'].to(device)  # pragma: no cover

        d_numerator = group['d_numerator']
        d_numerator.mul_(beta3)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                if len(state) == 0:
                    state['s'] = torch.zeros_like(p)
                    state['p0'] = p.clone()
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                p0, exp_avg, exp_avg_sq = state['p0'], state['exp_avg'], state['exp_avg_sq']

                d_numerator.add_(torch.dot(grad.flatten(), (p0 - p).flatten()), alpha=(d / d0) * d_lr)

                exp_avg.mul_(beta1).add_(grad, alpha=d * (1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=d * d * (1.0 - beta2))

                s = state['s']
                s.mul_(beta3).add_(grad, alpha=(d / d0) * (d if group['safeguard_warmup'] else d_lr))

                d_de_nom.add_(s.abs().sum())

        if d_de_nom == 0:
            return loss

        d_hat = (group['d_coef'] * d_numerator / d_de_nom).item()
        if d == group['d0']:
            d = max(d, d_hat)

        d_max = max(group['d_max'], d_hat)
        d = min(d_max, d * group['growth_rate'])

        for group in self.param_groups:
            group['step'] += 1

            group['d_numerator'] = d_numerator
            group['d_de_nom'] = d_de_nom
            group['d'] = d
            group['d_max'] = d_max
            group['d_hat'] = d_hat

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                self.apply_weight_decay(
                    p,
                    p.grad,
                    lr=d_lr,
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                de_nom = exp_avg_sq.sqrt()

                if group['eps'] is not None:
                    de_nom.add_(d * group['eps'])
                    p.addcdiv_(exp_avg, de_nom, value=-d_lr)
                else:
                    update = exp_avg.clone().atan2_(de_nom)
                    p.add_(update, alpha=-d_lr)

        return loss
