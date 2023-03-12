import math

import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError, ZeroParameterSizeError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class AdamS(Optimizer, BaseOptimizer):
    r"""Adam with stable weight decay.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param amsgrad: bool. whether to use the AMSGrad variant of this algorithm from the paper.
    :param adamd_debias_term: bool. Only correct the denominator to avoid inflating step sizes early in training.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
        weight_decay: float = 1e-4,
        amsgrad: bool = False,
        adamd_debias_term: bool = False,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
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
        return 'AdamS'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        param_size: int = 0
        exp_avg_sq_hat_sum: float = 0.0

        for group in self.param_groups:
            beta1, beta2 = group['betas']
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
                    if self.amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                bias_correction2 = 1.0 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if self.amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    exp_avg_sq_hat = max_exp_avg_sq
                else:
                    exp_avg_sq_hat = exp_avg_sq

                exp_avg_sq_hat_sum += exp_avg_sq_hat.sum() / bias_correction2

        if param_size == 0:
            raise ZeroParameterSizeError()

        exp_avg_sq_hat_mean = math.sqrt(exp_avg_sq_hat_sum / param_size) + self.eps

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                if group['weight_decay'] > 0.0:
                    p.mul_(1.0 - group['lr'] * group['weight_decay'] / exp_avg_sq_hat_mean)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                bias_correction1 = 1.0 - beta1 ** state['step']
                bias_correction2 = 1.0 - beta2 ** state['step']

                exp_avg_sq_hat = state['max_exp_avg_sq'] if self.amsgrad else exp_avg_sq
                exp_avg_sq_hat.div_(bias_correction2)

                de_nom = exp_avg_sq_hat.sqrt().add(group['eps'])

                step_size = group['lr'] if self.adamd_debias_term else group['lr'] / bias_correction1
                p.addcdiv_(exp_avg, de_nom, value=-step_size)

        return loss
