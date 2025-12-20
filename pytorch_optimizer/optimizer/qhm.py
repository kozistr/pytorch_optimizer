import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, Parameters, ParamGroup


class QHM(BaseOptimizer):
    """Quasi-hyperbolic momentum (QHM) optimization algorithm.

    Args:
        params (Parameters): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate.
        momentum (float): momentum factor.
        nu (float): immediate discount factor used to estimate the gradient and its square.
        weight_decay (float): weight decay (L2 penalty).
        weight_decouple (bool): the optimizer uses decoupled weight decay as in AdamW.
        fixed_decay (bool): fix weight decay.
        maximize (bool): maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        momentum: float = 0.0,
        nu: float = 1.0,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_nus(nu)

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'momentum': momentum,
            'nu': nu,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'QHM'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            state = self.state[p]

            if len(state) == 0:
                state['momentum_buffer'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                buf = state['momentum_buffer']

                p, grad, buf = self.view_as_real(p, grad, buf)

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=group['fixed_decay'],
                )

                buf.mul_(group['momentum']).add_(grad, alpha=1.0 - group['momentum'])

                p.add_(buf, alpha=-group['lr'] * group['nu'])
                p.add_(grad, alpha=-group['lr'] * (1.0 - group['nu']))

        return loss
