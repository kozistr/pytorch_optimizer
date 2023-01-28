import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.optimizer.utils import matrix_power


class Shampoo(Optimizer, BaseOptimizer):
    r"""Preconditioned Stochastic Tensor Optimization

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param momentum: float. momentum.
    :param weight_decay: float. weight decay (L2 penalty).
    :param update_freq: int. update frequency to compute inverse.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        update_freq: int = 1,
        eps: float = 1e-4,
    ):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.update_freq = update_freq
        self.eps = eps

        self.validate_parameters()

        defaults: DEFAULTS = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            update_freq=update_freq,
            eps=eps,
        )
        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_momentum(self.momentum)
        self.validate_weight_decay(self.weight_decay)
        self.validate_update_frequency(self.update_freq)
        self.validate_epsilon(self.eps)

    @property
    def __name__(self) -> str:
        return 'Shampoo'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['step'] = 0

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(self.__name__)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0

                    if momentum > 0.0:
                        state['momentum_buffer'] = grad.clone()

                    # pre-condition matrices
                    for dim_id, dim in enumerate(grad.size()):
                        state[f'pre_cond_{dim_id}'] = group['eps'] * torch.eye(dim, out=grad.new(dim, dim))
                        state[f'inv_pre_cond_{dim_id}'] = grad.new(dim, dim).zero_()

                if momentum > 0.0:
                    grad.mul_(1.0 - momentum).add_(state['momentum_buffer'], alpha=momentum)

                if weight_decay > 0.0:
                    grad.add_(p, alpha=weight_decay)

                order: int = grad.ndimension()
                original_size: int = grad.size()
                for dim_id, dim in enumerate(grad.size()):
                    pre_cond = state[f'pre_cond_{dim_id}']
                    inv_pre_cond = state[f'inv_pre_cond_{dim_id}']

                    grad = grad.transpose_(0, dim_id).contiguous()
                    transposed_size = grad.size()

                    grad = grad.view(dim, -1)

                    grad_t = grad.t()
                    pre_cond.add_(grad @ grad_t)
                    if state['step'] % group['update_freq'] == 0:
                        inv_pre_cond.copy_(matrix_power(pre_cond, -1 / order))

                    if dim_id == order - 1:
                        grad = grad_t @ inv_pre_cond
                        grad = grad.view(original_size)
                    else:
                        grad = inv_pre_cond @ grad
                        grad = grad.view(transposed_size)

                state['step'] += 1
                state['momentum_buffer'] = grad

                p.add_(grad, alpha=-group['lr'])

        return loss
