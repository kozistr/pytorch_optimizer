import torch

from pytorch_optimizer.base.exception import NoComplexParameterError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, Parameters, ParamGroup


@torch.no_grad()
def reduce_max_except_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Perform reduce-max along all dimensions except the given dim.

    Args:
        x (torch.Tensor): Tensor to reduce-max.
        dim (int): Dimension to exclude.
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

    Args:
        params (Parameters): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        momentum (float): Coefficient used to scale prior updates before adding. This drastically increases
        memory usage if momentum > 0.0. This is ignored if the parameter's gradient is sparse.
        beta (float): Coefficient used for exponential moving averages.
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the objective with respect to the parameters, instead of minimizing.
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 1e-1,
        momentum: float = 0.0,
        beta: float = 0.0,
        eps: float = 1e-30,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0)
        self.validate_range(beta, 'beta', 0.0, 1.0, range_type='[]')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {'lr': lr, 'momentum': momentum, 'beta': beta, 'eps': eps}

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SM3'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            grad = p.grad

            shape = grad.shape
            rank: int = len(shape)

            state = self.state[p]

            if len(state) == 0:
                state['momentum_buffer'] = torch.zeros_like(grad)

                if grad.is_sparse:
                    state['accumulator_0'] = torch.zeros(shape[0], dtype=grad.dtype, device=grad.device)
                elif rank == 0:
                    state['accumulator_0'] = torch.zeros_like(grad)
                else:
                    for i in range(rank):
                        state[f'accumulator_{i}'] = torch.zeros(
                            [1] * i + [shape[i]] + [1] * (rank - 1 - i), dtype=grad.dtype, device=grad.device
                        )

    @staticmethod
    def make_sparse(grad: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        if grad._indices().dim() == 0 or values.dim() == 0:
            return grad.new().resize_as_(grad)
        return grad.new(grad._indices(), values, grad.size())

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

            momentum, beta = group['momentum'], group['beta']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                shape = grad.shape
                rank: int = len(shape)

                state = self.state[p]

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
