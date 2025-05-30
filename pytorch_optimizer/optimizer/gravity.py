import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import CLOSURE, DEFAULTS, GROUP, LOSS, PARAMETERS


class Gravity(BaseOptimizer):
    r"""a Kinematic Approach on Optimization in Deep Learning.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param alpha: float. alpha controls the V initialization.
    :param beta: float. beta will be used to compute running average of V.
    :param maximize: bool. maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-2,
        alpha: float = 0.01,
        beta: float = 0.9,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(alpha, 'alpha', 0.0, 1.0)
        self.validate_range(beta, 'beta', 0.0, 1.0, range_type='[]')

        self.maximize = maximize

        defaults: DEFAULTS = {'lr': lr, 'alpha': alpha, 'beta': beta}

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Gravity'

    def init_group(self, group: GROUP, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            state = self.state[p]

            if len(state) == 0:
                state['v'] = torch.empty_like(p).normal_(mean=0.0, std=group['alpha'] / group['lr'])

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' not in group:
                self.init_group(group)
                group['step'] = 1
            else:
                group['step'] += 1

            beta_t: float = (group['beta'] * group['step'] + 1) / (group['step'] + 2)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                v = state['v']

                p, grad, v = self.view_as_real(p, grad, v)

                m = 1.0 / grad.abs().max()
                zeta = grad / (1.0 + (grad / m) ** 2)

                v.mul_(beta_t).add_(zeta, alpha=1.0 - beta_t)

                p.add_(v, alpha=-group['lr'])

        return loss
