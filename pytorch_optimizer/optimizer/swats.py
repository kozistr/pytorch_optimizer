import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class SWATS(BaseOptimizer):
    r"""Improving Generalization Performance by Switching from Adam to SGD.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param ams_bound: bool. whether to use the ams_bound variant of this algorithm from the paper.
    :param nesterov: bool. enables Nesterov momentum.
    :param r: float. EMA factor. between 0.9 ~ 0.99 is preferred.
    :param adanorm: bool. whether to use the AdaNorm variant.
    :param adam_debias: bool. Only correct the denominator to avoid inflating step sizes early in training.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        ams_bound: bool = False,
        nesterov: bool = False,
        r: float = 0.95,
        adanorm: bool = False,
        adam_debias: bool = False,
        eps: float = 1e-6,
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
            'ams_bound': ams_bound,
            'nesterov': nesterov,
            'adanorm': adanorm,
            'adam_debias': adam_debias,
            'phase': 'adam',
            'eps': eps,
        }
        if adanorm:
            defaults.update({'r': r})

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SWATS'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['exp_avg2'] = torch.zeros((1,), dtype=p.dtype, device=p.device)
                if group['ams_bound']:
                    state['max_exp_avg_sq'] = torch.zeros_like(p)
                if group['adanorm']:
                    state['exp_grad_norm'] = torch.zeros((1,), dtype=p.dtype, device=p.device)

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
            bias_correction2: float = self.debias(beta2, group['step'])

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
                    state['exp_avg2'] = torch.zeros((1,), dtype=grad.dtype, device=grad.device)
                    if group['ams_bound']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p)
                    if group['adanorm']:
                        state['exp_grad_norm'] = torch.zeros((1,), dtype=grad.dtype, device=grad.device)

                self.apply_weight_decay(
                    p=p,
                    grad=p.grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                if group['phase'] == 'sgd':
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(grad)

                    buf = state['momentum_buffer']
                    buf.mul_(beta1).add_(grad)

                    update = buf.clone()
                    update.mul_(1.0 - beta1)

                    if group['nesterov']:
                        update.add_(buf, alpha=beta1)

                    p.add_(update, alpha=-group['lr'])

                    continue

                s_grad = self.get_adanorm_gradient(
                    grad=grad,
                    adanorm=group['adanorm'],
                    exp_grad_norm=state.get('exp_grad_norm', None),
                    r=group.get('r', None),
                )

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg.mul_(beta1).add_(s_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                de_nom = self.apply_ams_bound(
                    ams_bound=group['ams_bound'],
                    exp_avg_sq=exp_avg_sq,
                    max_exp_avg_sq=state.get('max_exp_avg_sq', None),
                    eps=group['eps'],
                )

                step_size: float = self.apply_adam_debias(
                    adam_debias=group['adam_debias'],
                    step_size=group['lr'] * math.sqrt(bias_correction2),
                    bias_correction1=bias_correction1,
                )

                perturb = exp_avg.clone()
                perturb.div_(de_nom).mul_(-step_size)

                p.add_(perturb)

                perturb_view = perturb.view(-1)
                pg = perturb_view.dot(grad.view(-1))

                if pg != 0:
                    scaling = perturb_view.dot(perturb_view).div_(-pg)

                    exp_avg2 = state['exp_avg2']
                    exp_avg2.mul_(beta2).add_(scaling, alpha=1.0 - beta2)

                    corrected_exp_avg = exp_avg2 / bias_correction2

                    if (
                        group['step'] > 1
                        and corrected_exp_avg > 0.0
                        and corrected_exp_avg.allclose(scaling, rtol=group['eps'])
                    ):
                        group['phase'] = 'sgd'
                        group['lr'] = corrected_exp_avg.item()

        return loss
