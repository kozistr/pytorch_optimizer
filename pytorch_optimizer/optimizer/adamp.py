import math

import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.optimizer.gc import centralize_gradient
from pytorch_optimizer.optimizer.utils import projection


class AdamP(Optimizer, BaseOptimizer):
    r"""Slowing Down the Slowdown for Momentum Optimizers on Scale-invariant Weights.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param delta: float. threshold that determines whether a set of parameters is scale invariant or not.
    :param wd_ratio: float. relative weight decay applied on scale-invariant parameters compared to that applied
        on scale-variant parameters.
    :param use_gc: bool. use gradient centralization.
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
        delta: float = 0.1,
        wd_ratio: float = 0.1,
        use_gc: bool = False,
        nesterov: bool = False,
        r: float = 0.95,
        adanorm: bool = False,
        adam_debias: bool = False,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.eps = eps
        self.wd_ratio = wd_ratio
        self.use_gc = use_gc

        self.validate_parameters()

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
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

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_betas(self.betas)
        self.validate_weight_decay(self.weight_decay)
        self.validate_weight_decay_ratio(self.wd_ratio)
        self.validate_epsilon(self.eps)

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
            bias_correction1 = 1.0 - beta1 ** group['step']
            bias_correction2_sq = math.sqrt(1.0 - beta2 ** group['step'])

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

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                if self.use_gc:
                    grad = centralize_gradient(grad, gc_conv_only=False)

                s_grad = grad
                if group['adanorm']:
                    grad_norm = torch.linalg.norm(grad)

                    exp_grad_norm = state['exp_grad_norm']
                    exp_grad_norm.mul_(group['r']).add_(grad_norm, alpha=1.0 - group['r'])

                    if exp_grad_norm > grad_norm:
                        s_grad *= exp_grad_norm / grad_norm

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

                if group['weight_decay'] > 0.0:
                    p.mul_(1.0 - group['lr'] * group['weight_decay'] * wd_ratio)

                step_size: float = group['lr'] if group['adam_debias'] else group['lr'] / bias_correction1
                p.add_(perturb, alpha=-step_size)

        return loss
