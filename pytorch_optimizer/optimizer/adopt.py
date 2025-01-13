import math
from typing import Callable, Optional

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class ADOPT(BaseOptimizer):
    r"""Modified Adam Can Converge with Any β2 with the Optimal Rate.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param cautious: bool. whether to use the Cautious variant.
    :param stable_adamw: bool. whether to use stable AdamW variant.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.9999),
        clip_lambda: Optional[Callable[[float], float]] = lambda step: math.pow(step, 0.25),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        cautious: bool = False,
        stable_adamw: bool = False,
        eps: float = 1e-6,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.clip_lambda = clip_lambda
        self.cautious = cautious
        self.stable_adamw = stable_adamw

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
        return 'ADOPT'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

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
            lr: float = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=lr,
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                if group['step'] == 1:
                    exp_avg_sq.addcmul_(grad, grad.conj())
                    continue

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1.0 - beta2)

                de_nom = exp_avg_sq.sqrt().clamp_(min=group['eps'])

                normed_grad = grad.div(de_nom)
                if self.clip_lambda is not None:
                    clip = self.clip_lambda(group['step'])
                    normed_grad.clamp_(-clip, clip)

                exp_avg.lerp_(normed_grad, weight=1.0 - beta1)

                if self.cautious:
                    update = exp_avg.clone()
                    self.apply_cautious(update, normed_grad)
                else:
                    update = exp_avg

                if self.stable_adamw:
                    lr /= self.get_stable_adamw_rms(grad, exp_avg_sq)

                p.add_(update, alpha=-lr)

        return loss
