import math

import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class AdaNorm(Optimizer, BaseOptimizer):
    r"""Symbolic Discovery of Optimization Algorithms.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param r: float. EMA factor.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param adamd_debias_term: bool. Only correct the denominator to avoid inflating step sizes early in training.
    :param amsgrad: bool. whether to use the AMSGrad variant of this algorithm from the paper.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.99),
        r: float = 0.95,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        adamd_debias_term: bool = False,
        amsgrad: bool = False,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.betas = betas
        self.r = r
        self.weight_decay = weight_decay
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.adamd_debias_term = adamd_debias_term
        self.amsgrad = amsgrad
        self.eps = eps

        self.validate_parameters()

        defaults: DEFAULTS = {'lr': lr, 'betas': betas, 'weight_decay': weight_decay}
        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_betas(self.betas)
        self.validate_weight_decay(self.weight_decay)
        self.validate_epsilon(self.eps)

    def __str__(self) -> str:
        return 'AdaNorm'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_var'] = torch.zeros_like(p)
                state['exp_grad_norm'] = torch.zeros((1,), dtype=p.dtype, device=p.device)
                if self.amsgrad:
                    state['max_exp_avg_var'] = torch.zeros_like(p)

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

            bias_correction1 = 1 - beta1 ** group['step']
            bias_correction2_sq = math.sqrt(1 - beta2 ** group['step'])

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_var'] = torch.zeros_like(p)
                    state['exp_grad_norm'] = torch.zeros((1,), dtype=p.dtype, device=p.device)
                    if self.amsgrad:
                        state['max_exp_avg_var'] = torch.zeros_like(p)

                if self.weight_decouple:
                    p.mul_(1.0 - group['weight_decay'] * (1 if self.fixed_decay else group['lr']))
                elif group['weight_decay'] > 0.0:
                    grad.add_(p, alpha=group['weight_decay'])

                # gradient norm correction (GNC)
                grad_norm = torch.linalg.norm(grad)

                exp_grad_norm = state['exp_grad_norm']
                exp_grad_norm.mul_(self.r).add_(grad_norm, alpha=1.0 - self.r)

                s_grad = grad
                if exp_grad_norm > grad_norm:
                    s_grad *= exp_grad_norm / grad_norm

                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']
                exp_avg.mul_(beta1).add_(s_grad, alpha=1.0 - beta1)
                exp_avg_var.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if self.amsgrad:
                    max_exp_avg_var = state['max_exp_avg_var']
                    torch.max(max_exp_avg_var, exp_avg_var, out=max_exp_avg_var)
                    de_nom = max_exp_avg_var.add(self.eps).sqrt()
                else:
                    de_nom = exp_avg_var.add(self.eps).sqrt()

                de_nom.div_(bias_correction2_sq).add_(self.eps)

                step_size = group['lr'] if self.adamd_debias_term else group['lr'] / bias_correction1
                p.addcdiv_(exp_avg, de_nom, value=-step_size)

        return loss
