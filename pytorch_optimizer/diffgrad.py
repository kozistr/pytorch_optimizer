import math

import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base_optimizer import BaseOptimizer
from pytorch_optimizer.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class DiffGrad(Optimizer, BaseOptimizer):
    """
    Reference : https://github.com/shivram1987/diffGrad
    Example :
        from pytorch_optimizer import DiffGrad
        ...
        model = YourModel()
        optimizer = DiffGrad(model.parameters())
        ...
        for input, output in data:
          optimizer.zero_grad()
          loss = loss_function(output, model(input))
          loss.backward()
          optimizer.step()
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        adamd_debias_term: bool = False,
    ):
        """DiffGrad optimizer
        :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups
        :param lr: float. learning rate
        :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace
        :param eps: float. term added to the denominator to improve numerical stability
        :param weight_decay: float. weight decay (L2 penalty)
        :param adamd_debias_term: bool. Only correct the denominator to avoid inflating step sizes early in training
        """
        self.lr = lr
        self.eps = eps
        self.betas = betas
        self.weight_decay = weight_decay

        self.validate_parameters()

        defaults: DEFAULTS = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, adamd_debias_term=adamd_debias_term
        )
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
                state['previous_grad'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('diffGrad does not support sparse gradients')

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['previous_grad'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq, previous_grad = state['exp_avg'], state['exp_avg_sq'], state['previous_grad']

                if group['weight_decay'] != 0:
                    grad.add_(p, alpha=group['weight_decay'])

                state['step'] += 1
                beta1, beta2 = group['betas']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                de_nom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1.0 - beta1 ** state['step']
                bias_correction2 = 1.0 - beta2 ** state['step']

                # compute diffGrad coefficient (dfc)
                diff = abs(previous_grad - grad)
                dfc = 1.0 / (1.0 + torch.exp(-diff))
                state['previous_grad'] = grad.clone()

                # update momentum with dfc
                exp_avg1 = exp_avg * dfc

                step_size = group['lr'] * math.sqrt(bias_correction2)
                if not group['adamd_debias_term']:
                    step_size /= bias_correction1

                p.addcdiv_(exp_avg1, de_nom, value=-step_size)

        return loss
