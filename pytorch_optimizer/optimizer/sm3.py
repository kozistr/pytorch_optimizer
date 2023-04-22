import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import CLOSURE, DEFAULTS, LOSS, PARAMETERS


class SM3(Optimizer, BaseOptimizer):
    r"""Memory-Efficient Adaptive Optimization.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param momentum: float.
    :param beta: float.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-1,
        momentum: float = 0.0,
        beta: float = 0.0,
        eps: float = 1e-30,
    ):
        self.lr = lr
        self.momentum = momentum
        self.beta = beta
        self.eps = eps

        self.validate_parameters()

        defaults: DEFAULTS = {'lr': lr, 'momentum': momentum, 'beta': beta}
        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_momentum(self.momentum)
        self.validate_beta(self.beta)
        self.validate_epsilon(self.eps)

    def __str__(self) -> str:
        return 'SM3'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['momentum_buffer'] = 0.0

    @staticmethod
    def max_reduce_except_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
        r"""Perform reduce-max along all dimensions except the given dim."""
        rank: int = len(x.shape)
        if rank == 0:
            return x

        if dim >= rank:
            raise ValueError(f'[-] given dim is bigger than rank. {dim} >= {rank}')

        for d in range(rank):
            if d != dim:
                x = x.max(dim=d, keepdim=True).values
        return x

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
                    state['momentum_buffer'] = 0.0

                    if grad.is_sparse:
                        state['accumulator_0'] = torch.zeros(shape[0])
                    elif rank == 0:
                        state['accumulator_0'] = torch.zeros(shape)
                    else:
                        for i in range(rank):
                            state[f'accumulator_{i}'] = torch.zeros([1] * i + [shape[i]] + [1] * (rank - 1 - i))

                state['step'] += 1

                if grad.is_sparse:
                    grad = grad.coalesce()

                    acc = state['accumulator_0']
                    update_values = torch.gather(acc, 0, grad._indices()[0])
                    if beta > 0.0:
                        update_values.mul_(beta)
                    update_values.addcmul_(grad._values(), grad._values(), value=1.0 - beta)

                    sparse_update_values = self.make_sparse(grad, update_values)

                    nu_max = self.max_reduce_except_dim(sparse_update_values.to_dense(), 0).squeeze()
                    if beta > 0.0:
                        torch.max(acc, nu_max, acc)
                    else:
                        acc.copy_(nu_max)

                    update_values.add_(self.eps).rsqrt_().mul_(grad._values())

                    update = self.make_sparse(grad, update_values)
                else:
                    acc_list = (
                        [state[f'accumulator_{i}'] for i in range(rank)] if rank > 1 else [state['accumulator_0']]
                    )

                    update = acc_list[0].clone()
                    for i in range(1, rank):
                        update = torch.min(update, acc_list[i])

                    if beta > 0.0:
                        update.mul_(beta)
                    update.addcmul_(grad, grad, value=1.0 - beta)

                    for i, acc in enumerate(acc_list):
                        print(i, len(update.shape))
                        nu_max = self.max_reduce_except_dim(update, i)
                        if beta > 0.0:
                            torch.max(acc, nu_max, out=acc)
                        else:
                            acc.copy_(nu_max)

                    update.add_(self.eps).rsqrt_().mul_(grad)

                    if momentum > 0.0:
                        m = state['momentum_buffer']

                        update.mul_(1.0 - momentum).add_(m, alpha=momentum)
                        state['momentum_buffer'] = update.detach()

                p.sub_(update, alpha=group['lr'])

        return loss
