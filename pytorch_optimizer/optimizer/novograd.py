import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class NovoGrad(BaseOptimizer):
    r"""Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param grad_averaging: bool. multiply ck (1 - momentum).
    :param adam_debias: bool. Only correct the denominator to avoid inflating step sizes early in training.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.95, 0.98),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        grad_averaging: bool = False,
        adam_debias: bool = False,
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
            'grad_averaging': grad_averaging,
            'adam_debias': adam_debias,
            'eps': eps,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'NovoGrad'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                grad = p.grad

                g_2 = grad.pow(2).sum()  # fmt: skip

                state['moments'] = grad.div(g_2.sqrt() + group['eps']) + group['weight_decay'] * p
                state['grads_ema'] = g_2

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

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

            step_size: float = group['lr'] * bias_correction2_sq
            if not group['adam_debias']:
                step_size /= bias_correction1

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                grad_p2 = grad.pow(2).sum()

                if len(state) == 0:
                    state['moments'] = grad.div(grad_p2.sqrt() + group['eps']) + group['weight_decay'] * p
                    state['grads_ema'] = grad_p2

                grads_ema = state['grads_ema']
                grads_ema.mul_(beta2).add_(grad_p2, alpha=1.0 - beta2)

                de_nom = grads_ema.sqrt().add_(group['eps'])
                grad.div_(de_nom)

                self.apply_weight_decay(
                    p=p,
                    grad=p.grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                if group['grad_averaging']:
                    grad.mul_(1.0 - beta1)

                moments = state['moments']
                moments.mul_(beta1).add_(grad)

                p.add_(moments, alpha=-step_size)

        return loss
