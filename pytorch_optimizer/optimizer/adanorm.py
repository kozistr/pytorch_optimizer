import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class AdaNorm(BaseOptimizer):
    r"""Symbolic Discovery of Optimization Algorithms.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param r: float. EMA factor. between 0.9 ~ 0.99 is preferred.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param ams_bound: bool. whether to use the AMSBound variant.
    :param adam_debias: bool. Only correct the denominator to avoid inflating step sizes early in training.
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
        ams_bound: bool = False,
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
            'r': r,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'ams_bound': ams_bound,
            'adam_debias': adam_debias,
            'eps': eps,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdaNorm'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_var'] = torch.zeros_like(p)
                state['exp_grad_norm'] = torch.zeros((1,), dtype=p.dtype, device=p.device)
                if group['ams_bound']:
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

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

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
                    if group['ams_bound']:
                        state['max_exp_avg_var'] = torch.zeros_like(p)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                s_grad = self.get_adanorm_gradient(
                    grad=grad,
                    adanorm=True,
                    exp_grad_norm=state['exp_grad_norm'],
                    r=group['r'],
                )

                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']
                exp_avg.mul_(beta1).add_(s_grad, alpha=1.0 - beta1)
                exp_avg_var.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                de_nom = self.apply_ams_bound(
                    ams_bound=group['ams_bound'],
                    exp_avg_sq=exp_avg_var,
                    max_exp_avg_sq=state.get('max_exp_avg_var', None),
                    eps=group['eps'],
                )
                de_nom.div_(bias_correction2_sq)

                step_size: float = self.apply_adam_debias(
                    adam_debias=group['adam_debias'],
                    step_size=group['lr'],
                    bias_correction1=bias_correction1,
                )

                p.addcdiv_(exp_avg, de_nom, value=-step_size)

        return loss
