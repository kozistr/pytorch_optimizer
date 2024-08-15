import math
from typing import Union

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.optimizer.gc import centralize_gradient
from pytorch_optimizer.optimizer.utils import get_global_gradient_norm


class Adan(BaseOptimizer):
    r"""Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. decoupled weight decay.
    :param max_grad_norm: float. max gradient norm to clip.
    :param use_gc: bool. use gradient centralization.
    :param r: float. EMA factor. between 0.9 ~ 0.99 is preferred.
    :param adanorm: bool. whether to use the AdaNorm variant.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.98, 0.92, 0.99),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        max_grad_norm: float = 0.0,
        use_gc: bool = False,
        r: float = 0.95,
        adanorm: bool = False,
        eps: float = 1e-8,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(max_grad_norm, 'max_grad_norm')
        self.validate_non_negative(eps, 'eps')

        self.max_grad_norm = max_grad_norm
        self.use_gc = use_gc

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'max_grad_norm': max_grad_norm,
            'adanorm': adanorm,
            'eps': eps,
        }
        if adanorm:
            defaults.update({'r': r})

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Adan'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['exp_avg_diff'] = torch.zeros_like(p)
                state['previous_grad'] = torch.zeros_like(p)
                if group['adanorm']:
                    state['exp_grad_norm'] = torch.zeros((1,), dtype=p.dtype, device=p.device)

    @torch.no_grad()
    def get_global_gradient_norm(self) -> Union[torch.Tensor, float]:
        if self.defaults['max_grad_norm'] == 0.0:
            return 1.0

        global_grad_norm = get_global_gradient_norm(self.param_groups)
        global_grad_norm.sqrt_().add_(self.defaults['eps'])

        return torch.clamp(self.defaults['max_grad_norm'] / global_grad_norm, max=1.0)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        clip_global_grad_norm = self.get_global_gradient_norm()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            beta1, beta2, beta3 = group['betas']

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2: float = self.debias(beta2, group['step'])
            bias_correction3_sq: float = math.sqrt(self.debias(beta3, group['step']))

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
                    state['exp_avg_diff'] = torch.zeros_like(p)
                    state['previous_grad'] = grad.clone().mul_(-clip_global_grad_norm)
                    if group['adanorm']:
                        state['exp_grad_norm'] = torch.zeros((1,), dtype=grad.dtype, device=grad.device)

                grad.mul_(clip_global_grad_norm)

                if self.use_gc:
                    centralize_gradient(grad, gc_conv_only=False)

                grad_diff = state['previous_grad']
                grad_diff.add_(grad)

                s_grad = self.get_adanorm_gradient(
                    grad=grad,
                    adanorm=group['adanorm'],
                    exp_grad_norm=state.get('exp_grad_norm', None),
                    r=group.get('r', None),
                )

                exp_avg, exp_avg_sq, exp_avg_diff = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_diff']

                exp_avg.mul_(beta1).add_(s_grad, alpha=1.0 - beta1)
                exp_avg_diff.mul_(beta2).add_(grad_diff, alpha=1.0 - beta2)

                grad_diff.mul_(beta2).add_(grad)
                exp_avg_sq.mul_(beta3).addcmul_(grad_diff, grad_diff, value=1.0 - beta3)

                de_nom = exp_avg_sq.sqrt().div_(bias_correction3_sq).add_(group['eps'])

                if group['weight_decouple']:
                    p.mul_(1.0 - group['lr'] * group['weight_decay'])

                p.addcdiv_(exp_avg, de_nom, value=-group['lr'] / bias_correction1)
                p.addcdiv_(exp_avg_diff, de_nom, value=-group['lr'] * beta2 / bias_correction2)

                if not group['weight_decouple']:
                    p.div_(1.0 + group['lr'] * group['weight_decay'])

                state['previous_grad'].copy_(-grad)

        return loss
