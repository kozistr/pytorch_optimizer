import math

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT


class Prodigy(BaseOptimizer):
    """Adam with distance adaptive step sizes.

    Leave LR set to 1 unless you encounter instability.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for the gradient mean and squared gradients.
        beta3: Decay rate for the distance estimate. `None` uses the square root of `beta2`.
        d0: Initial estimate of the distance to the optimum.
        d_coef: Coefficient in the expression for the estimate of d.
        growth_rate: Maximum multiplicative growth of the distance estimate per step.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        bias_correction: Apply bias correction to the moment estimates.
        safeguard_warmup: Exclude the learning rate from the distance estimate denominator during warmup.
        eps: Term added to the denominator to improve numerical stability. when eps is None, use atan2 rather than
            epsilon and division for parameter updates.
        maximize: Maximize the objective instead of minimizing it.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1.0,
        betas: Betas = (0.9, 0.999),
        beta3: float | None = None,
        d0: float = 1e-6,
        d_coef: float = 1.0,
        growth_rate: float = float('inf'),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        bias_correction: bool = False,
        safeguard_warmup: bool = False,
        eps: float | None = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas((*betas, beta3))
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
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

            state = self.state[p]

            if len(state) == 0:
                state['s'] = torch.zeros_like(p)
                state['p0'] = p.clone()
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        device = group['params'][0].device

        d_de_nom = torch.tensor([0.0], device=device)

        beta1, beta2 = group['betas']
        beta3: float = group['beta3'] if group['beta3'] is not None else math.sqrt(beta2)

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
            self.init_group(group)

            group['step'] += 1

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

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
