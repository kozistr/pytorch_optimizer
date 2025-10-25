from typing import List

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, Parameters, ParamGroup


def channel_view(x: torch.Tensor) -> torch.Tensor:
    r"""Do channel view."""
    return x.view(x.size()[0], -1)


def neuron_norm(x: torch.Tensor) -> torch.Tensor:
    r"""Get norm of the tensor."""
    if x.dim() <= 1:
        return x.abs()

    view_shape: List[int] = [x.shape[0]] + [1] * (x.dim() - 1)

    return channel_view(x).norm(dim=1).view(*view_shape)


def neuron_mean(x: torch.Tensor) -> torch.Tensor:
    r"""Get mean of the tensor."""
    if x.dim() <= 1:
        raise ValueError('[-] neuron_mean not defined on 1D tensors.')

    view_shape: List[int] = [x.shape[0]] + [1] * (x.dim() - 1)

    return channel_view(x).mean(dim=1).view(*view_shape)


class Nero(BaseOptimizer):
    """Learning by Turning: Neural Architecture Aware Optimisation.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param beta: float. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param constraints: bool.
    :param eps: float. term added to the denominator to improve numerical stability.
    :param maximize: bool. maximize the objective with respect to the params, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 0.01,
        beta: float = 0.999,
        constraints: bool = True,
        eps: float = 1e-8,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(beta, 'beta', 0.0, 1.0, range_type='[]')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {'lr': lr, 'beta': beta, 'constraints': constraints, 'eps': eps}

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Nero'

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
                if group['constraints'] and p.dim() > 1:
                    p.sub_(neuron_mean(p))
                    p.div_(neuron_norm(p).add_(group['eps']))

                state['exp_avg_sq'] = torch.zeros_like(neuron_norm(p))

                state['scale'] = neuron_norm(p).mean()
                if state['scale'] == 0.0:
                    state['scale'] = 0.01

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

            bias_correction: float = self.debias(group['beta'], group['step'])

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                grad_norm = neuron_norm(grad)

                exp_avg_sq = state['exp_avg_sq']
                exp_avg_sq.mul_(group['beta']).addcmul_(grad_norm, grad_norm, value=1.0 - group['beta'])

                grad_normed = grad / ((exp_avg_sq / bias_correction).sqrt_().add_(group['eps']))
                torch.nan_to_num(grad_normed, nan=0.0, out=grad_normed)

                p.add_(grad_normed, alpha=-group['lr'] * state['scale'])

                if group['constraints'] and p.dim() > 1:
                    p.sub_(neuron_mean(p))
                    p.div_(neuron_norm(p).add_(group['eps']))

        return loss
