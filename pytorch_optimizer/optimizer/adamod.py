import math

import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class AdaMod(Optimizer, BaseOptimizer):
    r"""An Adaptive and Momental Bound Method for Stochastic Learning.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
        beta3 is for smoothing coefficient for adaptive learning rates.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param adam_debias: bool. Only correct the denominator to avoid inflating step sizes early in training.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.99, 0.9999),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        adam_debias: bool = False,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.eps = eps

        self.validate_parameters()

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'adam_debias': adam_debias,
            'eps': eps,
        }
        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_betas(self.betas)
        self.validate_weight_decay(self.weight_decay)
        self.validate_epsilon(self.eps)

    def __str__(self) -> str:
        return 'AdaMod'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['exp_avg_lr'] = torch.zeros_like(p)

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

            beta1, beta2, beta3 = group['betas']

            bias_correction1 = 1.0 - beta1 ** group['step']
            bias_correction2_sq = math.sqrt(1.0 - beta2 ** group['step'])

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
                    state['exp_avg_lr'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                de_nom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = torch.full_like(
                    de_nom,
                    fill_value=group['lr']
                    * bias_correction2_sq
                    / (bias_correction1 if not group['adam_debias'] else 1.0),
                )
                step_size.div_(de_nom)

                if group['weight_decouple']:
                    p.mul_(1.0 - group['weight_decay'] * group['lr'])
                elif group['weight_decay'] > 0.0:
                    grad.add_(p, alpha=group['weight_decay'])

                exp_avg_lr = state['exp_avg_lr']
                exp_avg_lr.mul_(beta3).add_(step_size, alpha=1.0 - beta3)

                torch.min(step_size, exp_avg_lr, out=step_size)
                step_size.mul_(exp_avg)

                p.add_(-step_size)

        return loss
