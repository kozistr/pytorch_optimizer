import math

import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.types import (
    BETAS,
    CLOSURE,
    DEFAULT_PARAMETERS,
    LOSS,
    PARAMS,
    STATE,
)


class AdaBound(Optimizer):
    """
    Reference : https://github.com/Luolc/AdaBound/blob/master/adabound/adabound.py
    Example :
        from pytorch_optimizer import AdaBound
        ...
        model = YourModel()
        optimizer = AdaBound(model.parameters())
        ...
        for input, output in data:
          optimizer.zero_grad()
          loss = loss_function(output, model(input))
          loss.backward()
          optimizer.step()
    """

    def __init__(
        self,
        params: PARAMS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
        final_lr: float = 0.1,
        gamma: float = 1e-3,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsbound: bool = False,
    ):
        """AdaBound optimizer
        :param params: PARAMS. iterable of parameters to optimize or dicts defining parameter groups
        :param lr: float. learning rate
        :param final_lr: float. final learning rate
        :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace
        :param gamma: float. convergence speed of the bound functions
        :param eps: float. term added to the denominator to improve numerical stability
        :param weight_decay: float. weight decay (L2 penalty)
        :param amsbound: bool. whether to use the AMSBound variant
        """
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        defaults: DEFAULT_PARAMETERS = dict(
            lr=lr,
            betas=betas,
            final_lr=final_lr,
            gamma=gamma,
            eps=eps,
            weight_decay=weight_decay,
            amsbound=amsbound,
        )
        super().__init__(params, defaults)

        self.base_lrs = [group['lr'] for group in self.param_groups]

    def check_valid_parameters(self):
        if 0.0 > self.lr:
            raise ValueError(f'Invalid learning rate : {self.lr}')
        if 0.0 > self.eps:
            raise ValueError(f'Invalid eps : {self.eps}')
        if 0.0 > self.weight_decay:
            raise ValueError(f'Invalid weight_decay : {self.weight_decay}')
        if not 0.0 <= self.betas[0] < 1.0:
            raise ValueError(f'Invalid beta_0 : {self.betas[0]}')
        if not 0.0 <= self.betas[1] < 1.0:
            raise ValueError(f'Invalid beta_1 : {self.betas[1]}')

    def __setstate__(self, state: STATE):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)

    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'AdaBound does not support sparse gradients'
                    )

                amsbound = group['amsbound']

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if amsbound:
                        state['max_exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsbound:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = (
                    group['lr']
                    * math.sqrt(bias_correction2)
                    / bias_correction1
                )

                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (
                    1 - 1 / (group['gamma'] * state['step'] + 1)
                )
                upper_bound = final_lr * (
                    1 + 1 / (group['gamma'] * state['step'])
                )
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(
                    exp_avg
                )

                p.data.add_(-step_size)

        return loss


class AdaBoundW(Optimizer):
    """
    Reference : https://github.com/Luolc/AdaBound
    Example :
        from pytorch_optimizer import AdaBoundW
        ...
        model = YourModel()
        optimizer = AdaBoundW(model.parameters())
        ...
        for input, output in data:
          optimizer.zero_grad()
          loss = loss_function(output, model(input))
          loss.backward()
          optimizer.step()
    """

    def __init__(
        self,
        params: PARAMS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
        final_lr: float = 0.1,
        gamma: float = 1e-3,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsbound: bool = False,
    ):
        """AdaBound optimizer with decoupled weight decay
        :param params: PARAMS. iterable of parameters to optimize or dicts defining parameter groups
        :param lr: float. learning rate
        :param final_lr: float. final learning rate
        :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace
        :param gamma: float. convergence speed of the bound functions
        :param eps: float. term added to the denominator to improve numerical stability
        :param weight_decay: float. weight decay (L2 penalty)
        :param amsbound: bool. whether to use the AMSBound variant
        """
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        defaults: DEFAULT_PARAMETERS = dict(
            lr=lr,
            betas=betas,
            final_lr=final_lr,
            gamma=gamma,
            eps=eps,
            weight_decay=weight_decay,
            amsbound=amsbound,
        )
        super().__init__(params, defaults)

        self.base_lrs = [group['lr'] for group in self.param_groups]

    def check_valid_parameters(self):
        if 0.0 > self.lr:
            raise ValueError(f'Invalid learning rate : {self.lr}')
        if 0.0 > self.eps:
            raise ValueError(f'Invalid eps : {self.eps}')
        if 0.0 > self.weight_decay:
            raise ValueError(f'Invalid weight_decay : {self.weight_decay}')
        if not 0.0 <= self.betas[0] < 1.0:
            raise ValueError(f'Invalid beta_0 : {self.betas[0]}')
        if not 0.0 <= self.betas[1] < 1.0:
            raise ValueError(f'Invalid beta_1 : {self.betas[1]}')

    def __setstate__(self, state: STATE):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)

    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue

                p.mul_(1 - base_lr * group['weight_decay'])

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'AdaBound does not support sparse gradients'
                    )

                amsbound = group['amsbound']

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if amsbound:
                        state['max_exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsbound:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = (
                    group['lr']
                    * math.sqrt(bias_correction2)
                    / bias_correction1
                )

                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (
                    1 - 1 / (group['gamma'] * state['step'] + 1)
                )
                upper_bound = final_lr * (
                    1 + 1 / (group['gamma'] * state['step'])
                )
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(
                    exp_avg
                )

                p.data.add_(-step_size)

        return loss
