import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class Lion(Optimizer, BaseOptimizer):
    r"""Symbolic Discovery of Optimization Algorithms.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-4,
        betas: BETAS = (0.9, 0.99),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
    ):
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.weight_decouple = weight_decouple

        self.validate_parameters()

        defaults: DEFAULTS = {'lr': lr, 'betas': betas, 'weight_decay': weight_decay}
        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_betas(self.betas)
        self.validate_weight_decay(self.weight_decay)

    def __str__(self) -> str:
        return 'Lion'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                update = exp_avg = state['exp_avg']

                if weight_decay > 0.0:
                    if self.weight_decouple:
                        p.mul_(1.0 - group['lr'] * weight_decay)
                    else:
                        grad.add_(p, alpha=weight_decay)

                update.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

                p.add_(update.sign(), alpha=-group['lr'])

        return loss
