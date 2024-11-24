import torch

from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import CLOSURE, DEFAULTS, LOSS, PARAMETERS


@torch.no_grad()
def reduce_max_except_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    r"""Perform reduce-max along all dimensions except the given dim.

    :param x: torch.Tensor. tensor to reduce-max.
    :param dim: int. dimension to exclude.
    """
    rank: int = len(x.shape)
    if rank == 0:
        return x

    if dim >= rank:
        raise ValueError(f'[-] given dim is bigger than rank. {dim} >= {rank}')

    for d in range(rank):
        if d != dim:
            x = x.max(dim=d, keepdim=True).values
    return x


class SM3(BaseOptimizer):
    r"""Memory-Efficient Adaptive Optimization.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param momentum: float. coefficient used to scale prior updates before adding. This drastically increases
        memory usage if `momentum > 0.0`. This is ignored if the parameter's gradient is sparse.
    :param beta: float. coefficient used for exponential moving averages.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-1,
        momentum: float = 0.0,
        beta: float = 0.0,
        eps: float = 1e-30,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0)
        self.validate_range(beta, 'beta', 0.0, 1.0, range_type='[]')
        self.validate_non_negative(eps, 'eps')

        defaults: DEFAULTS = {'lr': lr, 'momentum': momentum, 'beta': beta, 'eps': eps}
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SM3'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['step'] = 0
                state['momentum_buffer'] = torch.zeros_like(p)

    @staticmethod
    def make_sparse(grad: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        if grad._indices().dim() == 0 or values.dim() == 0:
            return grad.new().resize_as_(grad)
        return grad.new(grad._indices(), values, grad.size())

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum, beta = group['momentum'], group['beta']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                shape = grad.shape
                rank: int = len(shape)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p)

                    if grad.is_sparse:
                        state['accumulator_0'] = torch.zeros(shape[0], dtype=grad.dtype, device=grad.device)
                    elif rank == 0:
                        state['accumulator_0'] = torch.zeros_like(p)
                    else:
                        for i in range(rank):
                            state[f'accumulator_{i}'] = torch.zeros(
                                [1] * i + [shape[i]] + [1] * (rank - 1 - i), dtype=grad.dtype, device=grad.device
                            )

                state['step'] += 1

                if grad.is_sparse:
                    grad = grad.coalesce()

                    acc = state['accumulator_0']
                    update_values = torch.gather(acc, 0, grad._indices()[0])
                    if beta > 0.0:
                        update_values.mul_(beta)
                    update_values.addcmul_(grad._values(), grad._values(), value=1.0 - beta)

                    nu_max = reduce_max_except_dim(self.make_sparse(grad, update_values).to_dense(), 0).squeeze_()

                    if beta > 0.0:
                        torch.max(acc, nu_max, out=acc)
                    else:
                        acc.copy_(nu_max)

                    update_values.add_(group['eps']).rsqrt_().mul_(grad._values())

                    update = self.make_sparse(grad, update_values)
                else:
                    update = state['accumulator_0'].clone()
                    for i in range(1, rank):
                        update = torch.min(update, state[f'accumulator_{i}'])

                    if beta > 0.0:
                        update.mul_(beta)
                    update.addcmul_(grad, grad, value=1.0 - beta)

                    for i in range(rank):
                        acc = state[f'accumulator_{i}']
                        nu_max = reduce_max_except_dim(update, i)
                        if beta > 0.0:
                            torch.max(acc, nu_max, out=acc)
                        else:
                            acc.copy_(nu_max)

                    update.add_(group['eps']).rsqrt_().mul_(grad)

                    if momentum > 0.0:
                        m = state['momentum_buffer']
                        m.mul_(momentum).add_(update, alpha=1.0 - momentum)
                        update = m

                p.add_(update, alpha=-group['lr'])

        return loss
