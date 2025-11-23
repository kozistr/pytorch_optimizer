from typing import Literal, Optional

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, Parameters, ParamGroup

Mode = Literal['g', 'm', 'c']


class BCOS(BaseOptimizer):
    """Stochastic Approximation with Block Coordinate Optimal Stepsizes.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        beta (float): smoothing factor in computing the momentum and EMA estimators.
        beta2 (Optional[float]):
        mode (Mode): algorithmic mode of BCOS, must be one of the three choices.
            'g': use gradient as search direction and EMA estimator for its 2nd moment (equivalent to RMSprop).
            'm': use momentum as search direction and EMA estimator for its 2nd moment (using same beta).
            'c': use momentum as search direction and conditional estimator for its 2nd moment.
        simple_cond (bool): whether use simple alternative in BCOS-c variant.
        weight_decay (float): weight decay regularization strength.
        weight_decouple (bool): The optimizer uses decoupled weight decay as in AdamW.
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the objective with respect to the parameters, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        beta: float = 0.9,
        beta2: Optional[float] = None,
        mode: Mode = 'c',
        simple_cond: bool = False,
        weight_decay: float = 0.1,
        weight_decouple: bool = True,
        eps: float = 1e-6,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(beta, 'beta', 0.0, 1.0)
        self.validate_options(mode, 'mode', ['g', 'm', 'c'])
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        self.mode = mode
        self.simple_cond = simple_cond
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'beta': beta,
            'beta2': beta2,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'eps': eps,
            **kwargs,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'BCOS'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]

            if self.mode in ('m', 'c') and 'm' not in state:
                state['m'] = grad.clone()

            if self.mode in ('g', 'm') and 'v' not in state:
                state['v'] = grad.square()

    def compute_v(self, grad: torch.Tensor, m: torch.Tensor, beta: float, beta2: Optional[float]) -> torch.Tensor:
        g2 = grad.square()

        if self.simple_cond:
            beta_v: float = 1.0 - (1.0 - beta) ** 2 if beta2 is None else beta2
            return beta_v * m.square() + (1.0 - beta_v) * g2

        return (
            (3.0 * beta ** 2 - 2.0 * beta ** 3) * m.square()
            + (1.0 - beta) ** 2 * g2
            + 2.0 * beta * (1.0 - beta) ** 2 * m * grad
        )  # fmt: skip

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

            beta, beta2 = group['beta'], group['beta2']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                self.apply_weight_decay(
                    p=p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=False,
                )

                v: Optional[torch.Tensor] = None

                if self.mode in ('m', 'c'):
                    m = state['m']
                    if self.mode == 'c':
                        v: torch.Tensor = self.compute_v(grad, m, beta, beta2)

                    m.mul_(beta).add_(grad, alpha=1.0 - beta)
                    d = m
                else:
                    d = grad

                if self.mode in ('g', 'm'):
                    beta_v: float = beta if beta2 is None else beta2

                    v = state['v']
                    v.mul_(beta_v).add_(d.square(), alpha=1.0 - beta_v)

                update = v.sqrt().add_(group['eps'])

                p.addcdiv_(d, update, alpha=-group['lr'])

        return loss
