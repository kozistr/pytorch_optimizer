import math

import torch
from torch.optim import Optimizer

from pytorch_optimizer.base_optimizer import BaseOptimizer
from pytorch_optimizer.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class Lamb(Optimizer, BaseOptimizer):
    """
    Reference : https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
    Example :
        from pytorch_optimizer import Lamb
        ...
        model = YourModel()
        optimizer = Lamb(model.parameters())
        ...
        for input, output in data:
          optimizer.zero_grad()
          loss = loss_function(output, model(input))
          loss.backward()
          optimizer.step()
    """

    clamp: float = 10.0

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        adam: bool = False,
        adamd_debias_term: bool = False,
        pre_norm: bool = False,
    ):
        """Lamb
        :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups
        :param lr: float. learning rate
        :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace
        :param eps: float. term added to the denominator to improve numerical stability
        :param weight_decay: float. weight decay (L2 penalty)
        :param adamd_debias_term: bool. Only correct the denominator to avoid inflating step sizes early in training
        :param pre_norm: bool. perform pre-normalization of all gradients
        """
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.eps = eps
        self.adam = adam
        self.adamd_debias_term = adamd_debias_term
        self.pre_norm = pre_norm

        self.validate_parameters()

        defaults: DEFAULTS = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_betas(self.betas)
        self.validate_weight_decay(self.weight_decay)
        self.validate_epsilon(self.eps)

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

    def get_gradient_norm(self) -> float:
        norm_sq: float = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                norm_sq += torch.linalg.norm(p.grad).item() ** 2

        norm = math.sqrt(norm_sq)

        return norm

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        grad_norm: float = 1.0
        if self.pre_norm:
            grad_norm = self.get_gradient_norm() + self.eps

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if self.pre_norm:
                    p.grad /= grad_norm

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instead.')

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
