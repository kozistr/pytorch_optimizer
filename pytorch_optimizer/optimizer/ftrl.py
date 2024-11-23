import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import CLOSURE, DEFAULTS, LOSS, PARAMETERS


class FTRL(BaseOptimizer):
    r"""Follow The Regularized Leader.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param lr_power: float. controls how the learning rate decreases during training. use zero for a fixed learning
        rate.
    :param beta: float. beta value in the paper.
    :param lambda_1: float. L1 regularization parameter.
    :param lambda_2: float. L2 regularization parameter.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        lr_power: float = -0.5,
        beta: float = 0.0,
        lambda_1: float = 0.0,
        lambda_2: float = 0.0,
        **kwargs
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(beta, 'beta')
        self.validate_non_positive(lr_power, 'lr_power')
        self.validate_non_negative(lambda_1, 'lambda_1')
        self.validate_non_negative(lambda_2, 'lambda_2')

        defaults: DEFAULTS = {'lr': lr, 'lr_power': lr_power, 'beta': beta, 'lambda_1': lambda_1, 'lambda_2': lambda_2}
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'FTRL'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['z'] = torch.zeros_like(p)
                state['n'] = torch.zeros_like(p)

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
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                if len(state) == 0:
                    state['z'] = torch.zeros_like(p)
                    state['n'] = torch.zeros_like(p)

                z, n = state['z'], state['n']

                grad_p2 = grad.pow(2)

                sigma = (n + grad_p2).pow_(-group['lr_power']).sub_(n.pow(-group['lr_power'])).div_(group['lr'])

                z.add_(grad).sub_(sigma.mul(p))
                n.add_(grad_p2)

                update = z.sign().mul_(group['lambda_1']).sub_(z)
                update.div_((group['beta'] + n.sqrt()).div_(group['lr']).add_(group['lambda_2']))

                p.copy_(update)
                p.masked_fill_(z.abs() < group['lambda_1'], 0.0)

        return loss
