import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import CLOSURE, DEFAULTS, LOSS, PARAMETERS


class AccSGD(Optimizer, BaseOptimizer):
    r"""Accelerating Stochastic Gradient Descent For Least Squares Regression.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param kappa: float. ratio of long to short step.
    :param xi: float. statistical advantage parameter.
    :param small_const: float. any small constant under 1.
    :param weight_decay: float. weight decay.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        kappa: float = 1000.0,
        xi: float = 10.0,
        small_const: float = 0.7,
        weight_decay: float = 0.0,
    ):
        self.lr = lr
        self.kappa = kappa
        self.xi = xi
        self.small_const = small_const
        self.weight_decay = weight_decay

        self.validate_parameters()

        defaults: DEFAULTS = {
            'lr': lr,
            'kappa': kappa,
            'xi': xi,
            'small_const': small_const,
            'weight_decay': weight_decay,
        }
        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_kappa(self.kappa)
        self.validate_xi(self.xi)
        self.validate_weight_decay(self.weight_decay)
        self.validate_constant(self.small_const, boundary=1.0)

    def __str__(self) -> str:
        return 'AccSGD'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['momentum_buffer'] = p.clone()

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            large_lr: float = group['lr'] * group['kappa'] / group['small_const']
            alpha: float = 1.0 - (group['xi'] * (group['small_const'] ** 2) / group['kappa'])
            beta: float = 1.0 - alpha
            zeta: float = group['small_const'] / (group['small_const'] + beta)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                if len(state) == 0:
                    state['momentum_buffer'] = p.clone()

                if group['weight_decay'] > 0.0:
                    p.add_(p, alpha=group['weight_decay'])

                buf = state['momentum_buffer']
                buf.mul_(1.0 / beta - 1.0).add_(p, alpha=1.0 - large_lr).mul_(beta)

                p.add_(p, alpha=-group['lr']).mul_(zeta).add_(buf, alpha=1.0 - zeta)

        return loss
