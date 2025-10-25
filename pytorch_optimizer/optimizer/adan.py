import math
from typing import Union

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, Parameters, ParamGroup
from pytorch_optimizer.optimizer.gradient_centralization import centralize_gradient
from pytorch_optimizer.optimizer.utils import get_global_gradient_norm


class Adan(BaseOptimizer):
    r"""Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. decoupled weight decay.
    :param max_grad_norm: float. max gradient norm to clip.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param maximize: bool. maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        betas: Betas = (0.98, 0.92, 0.99),
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        max_grad_norm: float = 0.0,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(max_grad_norm, 'max_grad_norm')
        self.validate_non_negative(eps, 'eps')

        self.max_grad_norm = max_grad_norm
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'max_grad_norm': max_grad_norm,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Adan'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        clip_global_grad_norm = kwargs.get('clip_global_grad_norm')

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

                if group.get('adanorm'):
                    state['exp_grad_adanorm'] = torch.zeros((1,), dtype=grad.dtype, device=grad.device)

    @torch.no_grad()
    def get_global_gradient_norm(self) -> Union[torch.Tensor, float]:
        if self.defaults['max_grad_norm'] == 0.0:
            return 1.0

        global_grad_norm = get_global_gradient_norm(self.param_groups)
        global_grad_norm.sqrt_().add_(self.defaults['eps'])

        return torch.clamp(self.defaults['max_grad_norm'] / global_grad_norm, max=1.0)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        clip_global_grad_norm = self.get_global_gradient_norm()

        for group in self.param_groups:
            if 'step' not in group:
                self.init_group(group, clip_global_grad_norm=clip_global_grad_norm)
                group['step'] = 1
            else:
                group['step'] += 1

            beta1, beta2, beta3 = group['betas']

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2: float = self.debias(beta2, group['step'])
            bias_correction3_sq: float = math.sqrt(self.debias(beta3, group['step']))

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                exp_avg, exp_avg_sq, exp_avg_diff = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_diff']
                grad_diff = state['previous_grad']

                p, grad, exp_avg, exp_avg_sq, exp_avg_diff, grad_diff = self.view_as_real(
                    p, grad, exp_avg, exp_avg_sq, exp_avg_diff, grad_diff
                )

                grad.mul_(clip_global_grad_norm)

                if group.get('use_gc'):
                    centralize_gradient(grad, gc_conv_only=False)

                grad_diff.add_(grad)

                s_grad = self.get_adanorm_gradient(
                    grad=grad,
                    adanorm=group.get('adanorm', False),
                    exp_grad_norm=state.get('exp_grad_adanorm', None),
                    r=group.get('adanorm_r', None),
                )

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

                grad.neg_()
                state['previous_grad'].copy_(
                    torch.view_as_complex(grad) if torch.is_complex(state['previous_grad']) else grad
                )

        return loss
