import math

import torch
from torch.optim import Optimizer

from pytorch_optimizer.types import BETAS, CLOSURE, DEFAULTS, PARAMETERS


class Lamb(Optimizer):
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

        self.check_valid_parameters()

        defaults: DEFAULTS = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super().__init__(params, defaults)

    def check_valid_parameters(self):
        if self.lr < 0.0:
            raise ValueError(f'Invalid learning rate : {self.lr}')
        if not 0.0 <= self.betas[0] < 1.0:
            raise ValueError(f'Invalid beta_0 : {self.betas[0]}')
        if not 0.0 <= self.betas[1] < 1.0:
            raise ValueError(f'Invalid beta_1 : {self.betas[1]}')
        if self.weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay : {self.weight_decay}')
        if self.eps < 0.0:
            raise ValueError(f'Invalid eps : {self.eps}')

    def get_gradient_norm(self) -> float:
        norm_sq: float = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                norm_sq += torch.linalg.norm(p.grad).item() ** 2

        norm = math.sqrt(norm_sq)

        return norm

    def step(self, closure: CLOSURE = None) -> float:
        loss = None
        if closure is not None:
            loss = closure()

        grad_norm: float = 1.0
        if self.pre_norm:
            grad_norm = self.get_gradient_norm()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if self.pre_norm:
                    p.grad /= (grad_norm + self.eps)

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('[-] Lamb does not support sparse gradients, consider SparseAdam instead.')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                step_size = group['lr']

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])

                adam_norm = adam_step.pow(2).sum().sqrt()
                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, self.clamp)
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1.0
                else:
                    trust_ratio = weight_norm / adam_norm

                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio

                if self.adam:
                    trust_ratio = 1.0

                p.data.add_(adam_step, alpha=-step_size * trust_ratio)

        return loss
