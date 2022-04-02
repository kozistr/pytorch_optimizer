import math

import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base_optimizer import BaseOptimizer
from pytorch_optimizer.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class PNM(Optimizer, BaseOptimizer):
    """
    Reference : https://github.com/zeke-xie/Positive-Negative-Momentum
    Example :
        from pytorch_optimizer import PNM
        ...
        model = YourModel()
        optimizer = PNM(model.parameters())
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
        betas: BETAS = (0.9, 1.0),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        eps: float = 1e-8,
    ):
        """PNM optimizer
        :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups
        :param lr: float. learning rate
        :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace
        :param weight_decay: float. weight decay (L2 penalty)
        :param weight_decouple: bool. use weight_decouple
        :param eps: float. term added to the denominator to improve numerical stability
        """
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.weight_decouple = weight_decouple
        self.eps = eps

        self.validate_parameters()

        defaults: DEFAULTS = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            weight_decouple=weight_decouple,
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
                state['pos_momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['neg_momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)

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
                    raise RuntimeError('PNM does not support sparse gradients')

                if group['weight_decouple']:
                    p.mul_(1.0 - group['lr'] * group['weight_decay'])
                else:
                    grad.add_(p, alpha=group['weight_decay'])

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['pos_momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['neg_momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['step'] += 1
                beta1, beta2 = group['betas']

                if state['step'] % 2 == 1:
                    pos_momentum = state['pos_momentum']
                    neg_momentum = state['neg_momentum']
                else:
                    neg_momentum = state['pos_momentum']
                    pos_momentum = state['neg_momentum']

                pos_momentum.mul_(beta1 ** 2).add_(grad, alpha=1.0 - beta1 ** 2)

                noise_norm = math.sqrt((1 + beta2) ** 2 + beta2 ** 2)
                delta_p = pos_momentum.mul(1 + beta2).add(neg_momentum, alpha=-beta2).mul(1.0 / noise_norm)

                p.add_(delta_p, alpha=-group['lr'])

        return loss
