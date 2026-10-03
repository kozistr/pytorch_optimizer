import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT


class AdamG(BaseOptimizer):
    """Parameter free adaptive updates with gradient dependent scaling.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for scaled gradient momentum, squared gradients, and the numerator scale.
        p: The p value in the numerator function `s(x) = p * x^q`.
        q: The q value in the numerator function `s(x) = p * x^q`.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        fixed_decay: Apply decoupled weight decay without scaling it by the learning rate.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1.0,
        betas: Betas = (0.95, 0.999, 0.95),
        p: float = 0.2,
        q: float = 0.24,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_positive(p, 'p')
        self.validate_positive(q, 'q')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.p = p
        self.q = q
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdamG'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            state = self.state[p]

            if len(state) == 0:
                state['m'] = torch.zeros_like(p)
                state['v'] = torch.zeros_like(p)
                state['r'] = torch.zeros_like(p)

    def s(self, p: torch.Tensor) -> torch.Tensor:
        """Compute the numerator scaling function `p * x ** q`."""
        return p.pow(self.q).mul_(self.p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            beta1, beta2, beta3 = group['betas']

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2: float = self.debias(beta2, group['step'])

            step_size: float = min(group['lr'], 1.0 / math.sqrt(group['step']))

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                m, v, r = state['m'], state['v'], state['r']

                p, grad, m, v, r = self.view_as_real(p, grad, m, v, r)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                r.lerp_(self.s(v), weight=1.0 - beta3)
                m.mul_(beta1).addcmul_(r, grad, value=1.0 - beta1)

                update = (m / bias_correction1) / (v / bias_correction2).sqrt_().add_(group['eps'])

                p.add_(update, alpha=-step_size)

        return loss
