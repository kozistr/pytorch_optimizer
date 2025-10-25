from collections import deque
from typing import Callable, Optional

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, Parameters, ParamGroup


class AdaShift(BaseOptimizer):
    """Decorrelation and Convergence of Adaptive Learning Rate Methods.

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        betas (Betas): Coefficients used for computing running averages of gradient and the squared Hessian trace.
        keep_num (int): Number of gradients used to compute first moment estimation.
        reduce_func (Optional[Callable]): Function applied to squared gradients to reduce correlation.
            If None, no function is applied.
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the objective with respect to the parameters, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        keep_num: int = 10,
        reduce_func: Optional[Callable] = torch.max,
        eps: float = 1e-10,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_positive(keep_num, 'keep_num')
        self.validate_non_negative(eps, 'eps')

        self.reduce_func: Callable = reduce_func if reduce_func is not None else lambda x: x
        self.maximize = maximize

        defaults: Defaults = {'lr': lr, 'betas': betas, 'keep_num': keep_num, 'eps': eps, **kwargs}

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'AdaShift'

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

            if len(state) == 0:
                state['grad_queue'] = deque([grad.clone()], maxlen=group['keep_num'])
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

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

            beta1, beta2 = group['betas']

            exp_weight_sum: int = sum(beta1 ** i for i in range(group['keep_num']))  # fmt: skip
            first_grad_weight: float = beta1 ** (group['keep_num'] - 1) / exp_weight_sum
            last_grad_weight: float = 1.0 / exp_weight_sum

            bias_correction: float = self.debias(beta2, group['step'] - group['keep_num'])

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                grad_queue = state['grad_queue']
                grad_queue.append(grad.clone())

                if len(grad_queue) != group['keep_num']:
                    continue

                offset_grad = grad_queue[0]

                exp_avg = state['exp_avg']
                exp_avg.sub_(offset_grad, alpha=first_grad_weight).mul_(beta1).add_(grad, alpha=last_grad_weight)

                reduced_grad_sq = self.reduce_func(offset_grad.pow_(2))

                exp_avg_sq = state['exp_avg_sq']
                exp_avg_sq.mul_(beta2).add_(reduced_grad_sq, alpha=1.0 - beta2)

                update = exp_avg.clone()
                if group.get('cautious'):
                    self.apply_cautious(update, grad)

                update.div_(exp_avg_sq.div(bias_correction).sqrt_().add_(group['eps']))

                p.add_(update, alpha=-group['lr'])

        return loss
