import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, Parameters, ParamGroup


class PNM(BaseOptimizer):
    """Positive-Negative Momentum.

    Args:
        params (Parameters): Iterable of the parameters to optimize.
        lr (float): Learning rate.
        betas (Betas): Coefficients used for computing running averages of gradient and the squared Hessian trace.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): Use weight decoupling, as in AdamW.
        fixed_decay (bool): Fix weight decay.
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the objective instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        betas: Betas = (0.9, 1.0),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas, beta_range_type='[]')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'PNM'

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
                state['pos_momentum'] = torch.zeros_like(p)
                state['neg_momentum'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            beta1, beta2 = group['betas']

            beta1_p2: float = beta1 ** 2  # fmt: skip
            noise_norm: float = math.sqrt((1 + beta2) ** 2 + beta2 ** 2)  # fmt: skip

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                if group['step'] % 2 == 1:
                    pos_momentum, neg_momentum = state['pos_momentum'], state['neg_momentum']
                else:
                    neg_momentum, pos_momentum = state['pos_momentum'], state['neg_momentum']

                p, grad, pos_momentum, neg_momentum = self.view_as_real(p, grad, pos_momentum, neg_momentum)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                pos_momentum.mul_(beta1_p2).add_(grad, alpha=1.0 - beta1_p2)

                delta_p = pos_momentum.mul(1.0 + beta2).add_(neg_momentum, alpha=-beta2).mul_(1.0 / noise_norm)

                p.add_(delta_p, alpha=-group['lr'])

        return loss
