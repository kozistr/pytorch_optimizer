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
    :param adamd_debias_term: bool. Only correct the denominator to avoid inflating step sizes early in training.
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
        adamd_debias_term: bool = False,
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
            'adamd_debias_term': adamd_debias_term,
            'eps': eps,
        }
        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_betas(self.betas)
        self.validate_weight_decay(self.weight_decay)
        self.validate_weight_decay_ratio(self.wd_ratio)
        self.validate_epsilon(self.eps)

    @property
    def __str__(self) -> str:
        return 'AdamP'

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

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(self.__str__)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                bias_correction1 = 1.0 - beta1 ** state['step']
                bias_correction2_sq = math.sqrt(1.0 - beta2 ** state['step'])

                if self.use_gc:
                    grad = centralize_gradient(grad, gc_conv_only=False)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                inv_de_nom = 1.0 / (exp_avg_sq.sqrt() / bias_correction2_sq).add_(group['eps'])

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

                step_size: float = group['lr'] if group['adamd_debias_term'] else group['lr'] / bias_correction1
                p.add_(perturb, alpha=-step_size)

        return loss
