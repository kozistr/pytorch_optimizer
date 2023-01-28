import math

import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError, ZeroParameterSizeError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.optimizer.gc import centralize_gradient


class Adai(Optimizer, BaseOptimizer):
    r"""Disentangling the Effects of Adaptive Learning Rate and Momentum

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param dampening: float. dampening for momentum. where dampening < 1,
        it will show some adaptive-moment behavior.
    :param use_gc: bool. use gradient centralization.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.1, 0.99),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        dampening: float = 1.0,
        use_gc: bool = False,
        eps: float = 1e-3,
    ):
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.weight_decouple = weight_decouple
        self.dampening = dampening
        self.use_gc = use_gc
        self.eps = eps

        self.validate_parameters()

        defaults: DEFAULTS = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            dampening=dampening,
            eps=eps,
        )
        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_betas(self.betas)
        self.validate_weight_decay(self.weight_decay)
        self.validate_epsilon(self.eps)

    @property
    def __name__(self) -> str:
        return 'Adai'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['beta1_prod'] = torch.ones_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        param_size: int = 0
        exp_avg_sq_hat_sum: float = 0.0

        for group in self.param_groups:
            _, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(self.__name__)

                param_size += p.numel()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['beta1_prod'] = torch.ones_like(p)

                state['step'] += 1

                exp_avg_sq = state['exp_avg_sq']

                if self.use_gc:
                    grad = centralize_gradient(grad, gc_conv_only=False)

                bias_correction2 = 1.0 - beta2 ** state['step']

                if group['weight_decay'] > 0.0:
                    if self.weight_decouple:
                        p.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        grad.add_(p, alpha=group['weight_decay'])

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                exp_avg_sq_hat_sum += exp_avg_sq.sum() / bias_correction2

        if param_size == 0:
            raise ZeroParameterSizeError()

        exp_avg_sq_hat_mean = exp_avg_sq_hat_sum / param_size

        for group in self.param_groups:
            beta0, beta2 = group['betas']
            beta0_dp = math.pow(beta0, 1.0 - group['dampening'])
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1_prod = state['beta1_prod']

                bias_correction2 = 1.0 - beta2 ** state['step']

                exp_avg_sq_hat = exp_avg_sq / bias_correction2
                beta1 = (
                    1.0 - (exp_avg_sq_hat / exp_avg_sq_hat_mean).pow(1.0 / (3.0 - 2.0 * group['dampening'])).mul(beta0)
                ).clamp(0.0, 1.0 - group['eps'])
                beta3 = (1.0 - beta1).pow(group['dampening'])

                beta1_prod.mul_(beta1)
                bias_correction1 = 1.0 - beta1_prod

                exp_avg.mul_(beta1).addcmul_(beta3, grad)
                exp_avg_hat = exp_avg / bias_correction1 * beta0_dp

                p.add_(exp_avg_hat, alpha=-group['lr'])

        return loss
