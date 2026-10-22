import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.foreach import foreach_add_, foreach_lerp_, group_tensors_by_device_and_dtype


class Gravity(BaseOptimizer):
    """Kinematic optimization with a gradient dependent velocity update.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        alpha: Alpha controls the V initialization.
        beta: Beta will be used to compute running average of V.
        maximize: Maximize the objective instead of minimizing it.
        foreach: Use batched tensor operations. `None` enables them for supported parameter groups.

    """

    _supports_compiled_foreach = True

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-2,
        alpha: float = 0.01,
        beta: float = 0.9,
        maximize: bool = False,
        foreach: bool | None = None,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(alpha, 'alpha', 0.0, 1.0)
        self.validate_range(beta, 'beta', 0.0, 1.0, range_type='[]')

        self.maximize = maximize
        self.foreach = foreach

        defaults: Defaults = {'lr': lr, 'alpha': alpha, 'beta': beta, 'foreach': foreach}

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Gravity'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            state = self.state[p]

            if len(state) == 0:
                state['v'] = torch.empty_like(p).normal_(mean=0.0, std=group['alpha'] / group['lr'])

    def _step_foreach(
        self,
        group: ParamGroup,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        vs: list[torch.Tensor],
        beta_t: float | torch.Tensor,
    ) -> None:
        if self.maximize:
            torch._foreach_neg_(grads)

        ms = torch._foreach_reciprocal(torch._foreach_norm(grads, ord=float('inf')))
        de_noms = torch._foreach_div(grads, ms)
        torch._foreach_pow_(de_noms, 2)
        torch._foreach_add_(de_noms, 1.0)
        zetas = torch._foreach_div(grads, de_noms)

        foreach_lerp_(vs, zetas, weight=1.0 - beta_t)
        foreach_add_(params, vs, alpha=-group['lr'])

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            beta_t: float = (group['beta'] * group['step'] + 1) / (group['step'] + 2)

            if self.can_use_foreach(group, group.get('foreach', self.foreach)):
                params, grads, state = self.collect_trainable_params(group, self.state, state_keys=['v'])
                for batch in group_tensors_by_device_and_dtype(params, grads, state):
                    self._step_foreach(group, batch['params'], batch['grads'], batch['v'], beta_t)
                continue

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                v = state['v']

                p, grad, v = self.view_as_real(p, grad, v)

                m = 1.0 / grad.abs().max()
                zeta = grad / (1.0 + (grad / m) ** 2)

                v.lerp_(zeta, weight=1.0 - beta_t)

                p.add_(v, alpha=-group['lr'])

        return loss
