import math

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.optimizer.gc import centralize_gradient
from pytorch_optimizer.optimizer.utils import projection


class AdamP(BaseOptimizer):
    r"""Slowing Down the Slowdown for Momentum Optimizers on Scale-invariant Weights.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param fixed_decay: bool. fix weight decay.
    :param delta: float. threshold that determines whether a set of parameters is scale invariant or not.
    :param wd_ratio: float. relative weight decay applied on scale-invariant parameters compared to that applied
        on scale-variant parameters.
    :param use_gc: bool. use gradient centralization.
    :param cautious: bool. whether to use the Cautious variant.
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
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        delta: float = 0.1,
        wd_ratio: float = 0.1,
        use_gc: bool = False,
        cautious: bool = False,
        nesterov: bool = False,
        r: float = 0.95,
        adanorm: bool = False,
        adam_debias: bool = False,
        eps: float = 1e-8,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(wd_ratio, 'wd_ratio', 0.0, 1.0)
        self.validate_non_negative(eps, 'eps')

        self.use_gc = use_gc
        self.cautious = cautious

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'delta': delta,
            'wd_ratio': wd_ratio,
            'nesterov': nesterov,
            'adanorm': adanorm,
            'adam_debias': adam_debias,
            'eps': eps,
        }
        if adanorm:
            defaults.update({'r': r})

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdamP'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
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
                    if group['adanorm']:
                        state['exp_grad_norm'] = torch.zeros((1,), dtype=grad.dtype, device=grad.device)

                if self.use_gc:
                    centralize_gradient(grad, gc_conv_only=False)

                s_grad = self.get_adanorm_gradient(
                    grad=grad,
                    adanorm=group['adanorm'],
                    exp_grad_norm=state.get('exp_grad_norm', None),
                    r=group.get('r', None),
                )

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg.mul_(beta1).add_(s_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                inv_de_nom = exp_avg_sq.rsqrt().add_(group['eps']).mul_(bias_correction2_sq)

                perturb = exp_avg.clone()
                if group['nesterov']:
                    perturb.mul_(beta1).addcmul_(grad, inv_de_nom, value=1.0 - beta1)
                else:
                    perturb.mul_(inv_de_nom)

                wd_ratio: float = 1.0
                if len(p.shape) > 1:
                    perturb, wd_ratio = projection(
                        p,
                        grad,
                        perturb,
                        group['delta'],
                        group['wd_ratio'],
                        group['eps'],
                    )

                self.apply_weight_decay(
                    p=p,
                    grad=None,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                    ratio=wd_ratio,
                )

                step_size: float = self.apply_adam_debias(
                    adam_debias=group['adam_debias'],
                    step_size=group['lr'],
                    bias_correction1=bias_correction1,
                )

                if self.cautious:
                    self.apply_cautious(perturb, grad)

                p.add_(perturb, alpha=-step_size)

        return loss
