import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class LaProp(BaseOptimizer):
    r"""Separating Momentum and Adaptivity in Adam.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param centered: bool.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param ams_bound: bool. whether to use the AMSBound variant.
    :param cautious: bool. whether to use the Cautious variant.
    :param eps: float. epsilon value.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 4e-4,
        betas: BETAS = (0.9, 0.999),
        centered: bool = False,
        steps_before_using_centered: int = 10,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        ams_bound: bool = False,
        cautious: bool = False,
        eps: float = 1e-15,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.cautious = cautious
        self.steps_before_using_centered: int = steps_before_using_centered

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'centered': centered,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'ams_bound': ams_bound,
            'eps': eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'LaProp'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['exp_avg_lr_1'] = 0.0
                state['exp_avg_lr_2'] = 0.0

                if group['centered']:
                    state['exp_mean_avg_beta2'] = torch.zeros_like(p)
                if group['ams_bound']:
                    state['max_exp_avg_sq'] = torch.zeros_like(p)

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
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['exp_avg_lr_1'] = 0.0
                    state['exp_avg_lr_2'] = 0.0

                    if group['centered']:
                        state['exp_mean_avg_beta2'] = torch.zeros_like(p)
                    if group['ams_bound']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                state['exp_avg_lr_1'] = state['exp_avg_lr_1'] * beta1 + (1.0 - beta1) * group['lr']
                state['exp_avg_lr_2'] = state['exp_avg_lr_2'] * beta2 + (1.0 - beta2)

                bias_correction1: float = state['exp_avg_lr_1'] / group['lr'] if group['lr'] != 0.0 else 1.0
                step_size: float = 1.0 / bias_correction1

                de_nom = exp_avg_sq
                if group['centered']:
                    exp_mean_avg_beta2 = state['exp_mean_avg_beta2']
                    exp_mean_avg_beta2.mul_(beta2).add_(grad, alpha=1.0 - beta2)
                    if group['step'] > self.steps_before_using_centered:
                        de_nom -= exp_mean_avg_beta2.pow(2)

                de_nom = self.apply_ams_bound(
                    ams_bound=group['ams_bound'],
                    exp_avg_sq=exp_avg_sq,
                    max_exp_avg_sq=state.get('max_exp_avg_sq', None),
                    eps=group['eps'],
                )
                de_nom.div_(bias_correction2_sq)

                exp_avg.mul_(beta1).addcdiv_(grad, de_nom, value=(1.0 - beta1) * group['lr'])

                if self.cautious:
                    update = exp_avg.clone()
                    self.apply_cautious(update, grad)
                else:
                    update = exp_avg

                p.add_(update, alpha=-step_size)

                self.apply_weight_decay(
                    p=p,
                    grad=p.grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

        return loss
