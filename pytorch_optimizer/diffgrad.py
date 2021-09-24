import math

import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS, STATE


class DiffGrad(Optimizer):
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
    ):
        """
        :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups
        :param lr: float. learning rate.
        :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace
        :param eps: float. term added to the denominator to improve numerical stability
        :param weight_decay: float. weight decay (L2 penalty)
        """
        self.lr = lr
        self.eps = eps
        self.betas = betas
        self.weight_decay = weight_decay

        self.check_valid_parameters()

        defaults: DEFAULTS = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def check_valid_parameters(self):
        if self.lr < 0.0:
            raise ValueError(f'Invalid learning rate : {self.lr}')
        if self.weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay : {self.weight_decay}')
        if not 0.0 <= self.betas[0] < 1.0:
            raise ValueError(f'Invalid beta_0 : {self.betas[0]}')
        if not 0.0 <= self.betas[1] < 1.0:
            raise ValueError(f'Invalid beta_1 : {self.betas[1]}')
        if self.eps < 0.0:
            raise ValueError(f'Invalid eps : {self.eps}')

    def __setstate__(self, state: STATE):
        super().__setstate__(state)

    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('diffGrad does not support sparse gradients')

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['previous_grad'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq, previous_grad = state['exp_avg'], state['exp_avg_sq'], state['previous_grad']

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                state['step'] += 1
                beta1, beta2 = group['betas']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # compute diffGrad coefficient (dfc)
                diff = abs(previous_grad - grad)
                dfc = 1.0 / (1.0 + torch.exp(-diff))
                state['previous_grad'] = grad.clone()

                # update momentum with dfc
                exp_avg1 = exp_avg * dfc

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg1, denom)

        return loss
