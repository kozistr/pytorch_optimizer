import math
from typing import List

import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base_optimizer import BaseOptimizer
from pytorch_optimizer.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class AdaBound(Optimizer, BaseOptimizer):
    """
    Reference : https://github.com/Luolc/AdaBound
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
        params: PARAMETERS,
        lr: float = 1e-3,
        final_lr: float = 1e-1,
        betas: BETAS = (0.9, 0.999),
        gamma: float = 1e-3,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        amsbound: bool = False,
        adamd_debias_term: bool = False,
        eps: float = 1e-8,
    ):
        """AdaBound
        :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups
        :param lr: float. learning rate
        :param final_lr: float. final learning rate
        :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace
        :param gamma: float. convergence speed of the bound functions
        :param weight_decay: float. weight decay (L2 penalty)
        :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW
        :param fixed_decay: bool. fix weight decay
        :param amsbound: bool. whether to use the AMSBound variant
        :param adamd_debias_term: bool. Only correct the denominator to avoid inflating step sizes early in training
        :param eps: float. term added to the denominator to improve numerical stability
        """
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.eps = eps

        self.validate_parameters()

        defaults: DEFAULTS = dict(
            lr=lr,
            betas=betas,
            final_lr=final_lr,
            gamma=gamma,
            weight_decay=weight_decay,
            amsbound=amsbound,
            adamd_debias_term=adamd_debias_term,
            eps=eps,
        )
        super().__init__(params, defaults)

        self.base_lrs: List[float] = [group['lr'] for group in self.param_groups]

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
                if group['amsbound']:
                    state['max_exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdaBound does not support sparse gradients')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if group['amsbound']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                if self.weight_decouple:
                    if not self.fixed_decay:
                        p.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        p.mul_(1.0 - group['weight_decay'])
                else:
                    if group['weight_decay'] != 0:
                        grad.add_(p, alpha=group['weight_decay'])

                beta1, beta2 = group['betas']

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if group['amsbound']:
                    exp_avg_sq = torch.max(state['max_exp_avg_sq'], exp_avg_sq)

                de_nom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1.0 - beta1 ** state['step']
                bias_correction2 = 1.0 - beta2 ** state['step']

                step_size = group['lr'] * math.sqrt(bias_correction2)
                if not group['adamd_debias_term']:
                    step_size /= bias_correction1

                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))

                step_size = torch.full_like(de_nom, step_size)
                step_size.div_(de_nom).clamp_(lower_bound, upper_bound).mul_(exp_avg)

                p.add_(-step_size)

        return loss
