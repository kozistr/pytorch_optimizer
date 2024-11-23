from typing import List

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.optimizer.utils import channel_view


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
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 0.01,
        beta: float = 0.999,
        constraints: bool = True,
        eps: float = 1e-8,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(beta, 'beta', 0.0, 1.0, range_type='[]')
        self.validate_non_negative(eps, 'eps')

        defaults: DEFAULTS = {'lr': lr, 'beta': beta, 'constraints': constraints, 'eps': eps}
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Nero'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                if group['constraints'] and p.dim() > 1:
                    p.sub_(neuron_mean(p))
                    p.div_(neuron_norm(p) + group['eps'])

                state = self.state[p]

                state['step'] = 0
                state['exp_avg_sq'] = torch.zeros_like(neuron_norm(p))
                state['scale'] = neuron_norm(p).mean()

                if state['scale'] == 0.0:
                    state['scale'] = 0.01

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
                    if group['constraints'] and p.dim() > 1:
                        p.sub_(neuron_mean(p))
                        p.div_(neuron_norm(p) + group['eps'])

                    state['step'] = 0
                    state['exp_avg_sq'] = torch.zeros_like(neuron_norm(p))
                    state['scale'] = neuron_norm(p).mean()
                    if state['scale'] == 0.0:
                        state['scale'] = 0.01

                state['step'] += 1

                grad_norm = neuron_norm(grad)

                exp_avg_sq = state['exp_avg_sq']
                exp_avg_sq.mul_(group['beta']).addcmul_(grad_norm, grad_norm, value=1.0 - group['beta'])

                bias_correction: float = 1.0 - group['beta'] ** state['step']

                grad_normed = grad / ((exp_avg_sq / bias_correction).sqrt_().add_(group['eps']))
                torch.nan_to_num(grad_normed, nan=0.0, out=grad_normed)

                p.sub_(grad_normed, alpha=group['lr'] * state['scale'])

                if group['constraints'] and p.dim() > 1:
                    p.sub_(neuron_mean(p))
                    p.div_(neuron_norm(p) + group['eps'])

        return loss
