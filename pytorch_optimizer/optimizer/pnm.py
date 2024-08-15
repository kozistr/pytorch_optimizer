import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class PNM(BaseOptimizer):
    r"""Positive-Negative Momentum Optimizers.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. use weight_decouple.
    :param fixed_decay: bool. fix weight decay.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 1.0),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        eps: float = 1e-8,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'PNM'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['pos_momentum'] = torch.zeros_like(p)
                state['neg_momentum'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            beta1, beta2 = group['betas']

            noise_norm: float = math.sqrt((1 + beta2) ** 2 + beta2 ** 2)  # fmt: skip

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

                self.apply_weight_decay(
                    p=p,
                    grad=p.grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                if group['step'] % 2 == 1:
                    pos_momentum, neg_momentum = state['pos_momentum'], state['neg_momentum']
                else:
                    neg_momentum, pos_momentum = state['pos_momentum'], state['neg_momentum']

                pos_momentum.mul_(beta1 ** 2).add_(grad, alpha=1.0 - beta1 ** 2)  # fmt: skip

                delta_p = pos_momentum.mul(1 + beta2).add_(neg_momentum, alpha=-beta2).mul_(1.0 / noise_norm)
                p.add_(delta_p, alpha=-group['lr'])

        return loss
