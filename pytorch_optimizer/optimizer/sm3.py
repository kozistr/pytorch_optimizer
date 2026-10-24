import torch

from pytorch_optimizer.base.exception import NoComplexParameterError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import foreach_add_, group_tensors_by_device_and_dtype


@torch.no_grad()
def reduce_max_except_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute the maximum over all dimensions except `dim`.

    Args:
        x: Tensor to reduce.
        dim: Dimension to exclude.

    """
    rank: int = len(x.shape)
    if rank == 0:
        return x

    if dim >= rank:
        raise ValueError(f'[-] given dim is bigger than rank. {dim} >= {rank}')

    dims = tuple(d for d in range(rank) if d != dim)

    return x.amax(dim=dims, keepdim=True) if dims else x


class SM3(BaseOptimizer):
    """Adaptive updates with memory efficient per dimension accumulators.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        momentum: Momentum factor.
        beta: Coefficient used for exponential moving averages.
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Batch dense FP32/FP64 updates. `None` enables batching on CUDA.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-1,
        momentum: float = 0.0,
        beta: float = 0.0,
        eps: float = 1e-30,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0)
        self.validate_range(beta, 'beta', 0.0, 1.0, range_type='[]')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        defaults: Defaults = {'lr': lr, 'momentum': momentum, 'beta': beta, 'eps': eps, 'foreach': foreach}

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SM3'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

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

    @staticmethod
    def _get_update(group: ParamGroup, grad: torch.Tensor, state: dict) -> torch.Tensor:
        rank = grad.ndim
        momentum, beta = group['momentum'], group['beta']
        update = state['accumulator_0'].clone()

        for i in range(1, rank):
            update = torch.min(update, state[f'accumulator_{i}'])

        if beta > 0.0:
            update.mul_(beta)

        update.addcmul_(grad, grad, value=1.0 - beta)

        if rank == 0:
            state['accumulator_0'].copy_(update)

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
            m.lerp_(update, weight=1.0 - momentum)
            update = m

        return update

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        states: list[dict],
    ) -> None:
        if self.maximize:
            torch._foreach_neg_(grads)

        updates = [self._get_update(group, grad, state) for grad, state in zip(grads, states)]
        foreach_add_(params, updates, alpha=-group['lr'])

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            beta = group['beta']

            if self.can_use_foreach(
                group, group.get('foreach'), dtypes=(torch.float32, torch.float64), default_on_cuda=True
            ):
                params, grads, _ = self.collect_trainable_params(group, self.state)

                for batch in group_tensors_by_device_and_dtype(params, grads):
                    states = [self.state[p] for p in batch['params']]
                    self._step_foreach(group, batch['params'], batch['grads'], states)

                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                if grad.is_sparse:
                    grad = grad.coalesce()

                    acc = state['accumulator_0']
                    update_values = torch.gather(acc, 0, grad._indices()[0])
                    update_values = update_values.reshape([-1] + [1] * (grad._values().ndim - 1))
                    if update_values.shape != grad._values().shape:
                        update_values = update_values.expand_as(grad._values()).clone()
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
                    update = self._get_update(group, grad, state)

                p.add_(update, alpha=-group['lr'])

        return loss
