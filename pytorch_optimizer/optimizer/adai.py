import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError, ZeroParameterSizeError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.optimizer.gc import centralize_gradient


class Adai(BaseOptimizer):
    r"""Disentangling the Effects of Adaptive Learning Rate and Momentum.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param stable_weight_decay: bool. perform stable weight decay.
    :param dampening: float. dampening for momentum. where dampening < 1, it will show some adaptive-moment behavior.
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
        fixed_decay: bool = False,
        stable_weight_decay: bool = False,
        dampening: float = 1.0,
        use_gc: bool = False,
        eps: float = 1e-3,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.use_gc = use_gc

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'stable_weight_decay': stable_weight_decay,
            'dampening': dampening,
            'eps': eps,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
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
                    raise NoSparseGradientError(str(self))

                param_size += p.numel()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['beta1_prod'] = torch.ones_like(p)

                state['step'] += 1

                if self.use_gc:
                    centralize_gradient(grad, gc_conv_only=False)

                bias_correction2: float = self.debias(beta2, state['step'])

                if not group['stable_weight_decay'] and group['weight_decay'] > 0.0:
                    self.apply_weight_decay(
                        p=p,
                        grad=grad,
                        lr=group['lr'],
                        weight_decay=group['weight_decay'],
                        weight_decouple=group['weight_decouple'],
                        fixed_decay=group['fixed_decay'],
                    )

                exp_avg_sq = state['exp_avg_sq']
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                exp_avg_sq_hat_sum += exp_avg_sq.sum() / bias_correction2

        if param_size == 0:
            raise ZeroParameterSizeError()

        exp_avg_sq_hat_mean = exp_avg_sq_hat_sum / param_size

        for group in self.param_groups:
            beta0, beta2 = group['betas']
            beta0_dp: float = math.pow(beta0, 1.0 - group['dampening'])
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]

                if group['stable_weight_decay'] and group['weight_decay'] > 0.0:
                    self.apply_weight_decay(
                        p=p,
                        grad=grad,
                        lr=group['lr'],
                        weight_decay=group['weight_decay'],
                        weight_decouple=group['weight_decouple'],
                        fixed_decay=group['fixed_decay'],
                    )

                bias_correction2: float = self.debias(beta2, state['step'])

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                exp_avg_sq_hat = exp_avg_sq / bias_correction2
                beta1 = (
                    1.0
                    - (exp_avg_sq_hat / exp_avg_sq_hat_mean).pow_(1.0 / (3.0 - 2.0 * group['dampening'])).mul_(beta0)
                ).clamp_(0.0, 1.0 - group['eps'])
                beta3 = (1.0 - beta1).pow_(group['dampening'])

                beta1_prod = state['beta1_prod']
                beta1_prod.mul_(beta1)

                exp_avg.mul_(beta1).addcmul_(beta3, grad)
                exp_avg_hat = exp_avg.div(1.0 - beta1_prod).mul_(beta0_dp)

                p.add_(exp_avg_hat, alpha=-group['lr'])

        return loss
