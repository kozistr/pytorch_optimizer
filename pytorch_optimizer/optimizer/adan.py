import math
from typing import Union

import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.optimizer.gc import centralize_gradient


class Adan(Optimizer, BaseOptimizer):
    r"""Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. decoupled weight decay.
    :param max_grad_norm: float. max gradient norm to clip.
    :param use_gc: bool. use gradient centralization.
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
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.weight_decouple = weight_decouple
        self.max_grad_norm = max_grad_norm
        self.use_gc = use_gc
        self.eps = eps

        self.validate_parameters()

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'max_grad_norm': max_grad_norm,
            'eps': eps,
        }
        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_betas(self.betas)
        self.validate_weight_decay(self.weight_decay)
        self.validate_epsilon(self.eps)
        self.validate_norm(self.max_grad_norm)

    @property
    def __str__(self) -> str:
        return 'Adan'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_diff'] = torch.zeros_like(p)
                state['exp_avg_nest'] = torch.zeros_like(p)
                state['previous_grad'] = torch.zeros_like(p)

    @torch.no_grad()
    def get_global_gradient_norm(self) -> Union[torch.Tensor, float]:
        if self.defaults['max_grad_norm'] == 0.0:
            return 1.0

        device = self.param_groups[0]['params'][0].device

        global_grad_norm = torch.zeros(1, device=device)
        max_grad_norm = torch.tensor(self.defaults['max_grad_norm'], device=device)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    global_grad_norm.add_(torch.linalg.norm(p.grad).pow(2))

        global_grad_norm = torch.sqrt(global_grad_norm)

        return torch.clamp(max_grad_norm / (global_grad_norm + self.eps), max=1.0)

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
            bias_correction1 = 1.0 - beta1 ** group['step']
            bias_correction2 = 1.0 - beta2 ** group['step']
            bias_correction3_sq = math.sqrt(1.0 - beta3 ** group['step'])

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(self.__str__)

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_diff'] = torch.zeros_like(p)
                    state['exp_avg_nest'] = torch.zeros_like(p)
                    state['previous_grad'] = grad.clone()

                exp_avg, exp_avg_diff, exp_avg_nest = state['exp_avg'], state['exp_avg_diff'], state['exp_avg_nest']

                grad.mul_(clip_global_grad_norm)

                if self.use_gc:
                    grad = centralize_gradient(grad, gc_conv_only=False)

                grad_diff = -state['previous_grad']
                grad_diff.add_(grad)
                state['previous_grad'].copy_(grad)

                update = grad + beta2 * grad_diff

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_diff.mul_(beta2).add_(grad_diff, alpha=1.0 - beta2)
                exp_avg_nest.mul_(beta3).addcmul_(update, update, value=1.0 - beta3)

                de_nom = (exp_avg_nest.sqrt_() / bias_correction3_sq).add_(self.eps)
                perturb = (exp_avg / bias_correction1 + beta2 * exp_avg_diff / bias_correction2).div_(de_nom)

                if group['weight_decouple']:
                    p.mul_(1.0 - group['lr'] * group['weight_decay'])
                    p.add_(perturb, alpha=-group['lr'])
                else:
                    p.add_(perturb, alpha=-group['lr'])
                    p.div_(1.0 + group['lr'] * group['weight_decay'])

        return loss
