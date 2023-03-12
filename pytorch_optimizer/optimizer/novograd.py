import math

import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class NovoGrad(Optimizer, BaseOptimizer):
    r"""Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param grad_averaging: bool. multiply ck (1 - momentum).
    :param adamd_debias_term: bool. Only correct the denominator to avoid inflating step sizes early in training.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.95, 0.98),
        weight_decay: float = 0.0,
        grad_averaging: bool = False,
        adamd_debias_term: bool = False,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.grad_averaging = grad_averaging
        self.adamd_debias_term = adamd_debias_term
        self.eps = eps

        self.validate_parameters()

        defaults: DEFAULTS = {'lr': lr, 'betas': betas, 'weight_decay': weight_decay, 'eps': eps}
        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_betas(self.betas)
        self.validate_weight_decay(self.weight_decay)
        self.validate_epsilon(self.eps)

    def __str__(self) -> str:
        return 'NovoGrad'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                grad = p.grad
                g_2 = grad ** 2  # fmt: skip

                state['step'] = 0
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
            weight_decay = group['weight_decay']

            bias_correction1 = 1.0 - beta1 ** group['step']
            bias_correction2_sq = math.sqrt(1.0 - beta2 ** group['step'])

            step_size: float = group['lr'] * bias_correction2_sq
            if not self.adamd_debias_term:
                step_size /= bias_correction1

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                g_2 = grad ** 2  # fmt: skip

                if len(state) == 0:
                    state['moments'] = grad.div(g_2.sqrt() + group['eps']) + weight_decay * p
                    state['grads_ema'] = g_2

                moments, grads_ema = state['moments'], state['grads_ema']

                grads_ema.mul_(beta2).add_(g_2, alpha=1.0 - beta2)

                de_nom = grads_ema.sqrt().add_(group['eps'])
                grad.div_(de_nom)

                if weight_decay > 0.0:
                    grad.add_(p, alpha=weight_decay)

                if self.grad_averaging:
                    grad.mul_(1.0 - beta1)

                moments.mul_(beta1).add_(grad)

                p.add_(moments, alpha=-step_size)

        return loss
