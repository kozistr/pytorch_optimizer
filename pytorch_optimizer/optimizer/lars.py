import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, Parameters, ParamGroup


class LARS(BaseOptimizer):
    """Layer-wise Adaptive Rate Scaling (no rate scaling or weight decay for parameters <= 1D).

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        weight_decay (float): Weight decay (L2 penalty).
        momentum (float): Momentum.
        dampening (float): Dampening for momentum.
        trust_coefficient (float): Trust coefficient.
        nesterov (bool): Enables Nesterov momentum.
        maximize (bool): Maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        dampening: float = 0.0,
        trust_coefficient: float = 1e-3,
        nesterov: bool = False,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(momentum, 'momentum', 0.0, 1.0)
        self.validate_range(dampening, 'dampening', 0.0, 1.0)
        self.validate_non_negative(trust_coefficient, 'trust_coefficient')

        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'weight_decay': weight_decay,
            'momentum': momentum,
            'dampening': dampening,
            'trust_coefficient': trust_coefficient,
            'nesterov': nesterov,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Lars'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if group['momentum'] > 0.0:
                state = self.state[p]

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = grad.clone()

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' not in group:
                self.init_group(group)
                group['step'] = 1
            else:
                group['step'] += 1

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                if p.ndim > 1:
                    param_norm = torch.linalg.norm(p)
                    update_norm = torch.linalg.norm(grad)

                    one = torch.ones_like(param_norm)

                    trust_ratio = torch.where(
                        param_norm > 0.0,
                        torch.where(update_norm > 0.0, (group['trust_coefficient'] * param_norm / update_norm), one),
                        one,
                    )

                    grad.add_(p, alpha=group['weight_decay'])
                    grad.mul_(trust_ratio)

                if group['momentum'] > 0.0:
                    mb = state['momentum_buffer']
                    mb.mul_(group['momentum']).add_(grad, alpha=1.0 - group['dampening'])

                    if group['nesterov']:
                        grad.add_(mb, alpha=group['momentum'])
                    else:
                        grad.copy_(mb)

                p.add_(grad, alpha=-group['lr'])

        return loss
