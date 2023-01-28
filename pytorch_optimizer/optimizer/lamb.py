from typing import Union

import torch
from torch.optim import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class Lamb(Optimizer, BaseOptimizer):
    r"""Large Batch Optimization for Deep Learning

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param weight_decay: float. weight decay (L2 penalty).
    :param grad_averaging: bool. whether apply (1 - beta2) to grad when calculating running averages of gradient.
    :param max_grad_norm: float. max gradient norm to clip.
    :param adamd_debias_term: bool. Only correct the denominator to avoid inflating step sizes early in training.
    :param pre_norm: bool. perform pre-normalization of all gradients.
    """

    clamp: float = 10.0

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        grad_averaging: bool = True,
        max_grad_norm: float = 1.0,
        adam: bool = False,
        adamd_debias_term: bool = False,
        pre_norm: bool = False,
    ):
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.grad_averaging = grad_averaging
        self.max_grad_norm = max_grad_norm
        self.eps = eps
        self.adam = adam
        self.adamd_debias_term = adamd_debias_term
        self.pre_norm = pre_norm

        self.validate_parameters()

        defaults: DEFAULTS = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            grad_averaging=grad_averaging,
            max_grad_norm=max_grad_norm,
        )

        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_betas(self.betas)
        self.validate_weight_decay(self.weight_decay)
        self.validate_epsilon(self.eps)
        self.validate_norm(self.max_grad_norm)

    @property
    def __name__(self) -> str:
        return 'Lamb'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

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
                    global_grad_norm.add_(p.grad.pow(2).sum())

        global_grad_norm = torch.sqrt(global_grad_norm)

        return torch.clamp(max_grad_norm / (global_grad_norm + self.eps), max=1.0)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        grad_norm = 1.0
        if self.pre_norm:
            grad_norm = self.get_global_gradient_norm()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if self.pre_norm:
                    p.grad /= grad_norm

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(self.__name__)

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                step_size = group['lr']

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(p, alpha=group['weight_decay'])

                adam_norm = adam_step.pow(2).sum().sqrt()
                weight_norm = p.pow(2).sum().sqrt().clamp(0, self.clamp)
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1.0
                else:
                    trust_ratio = weight_norm / adam_norm

                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio

                if self.adam:
                    trust_ratio = 1.0

                p.add_(adam_step, alpha=-step_size * trust_ratio)

        return loss
