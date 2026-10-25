from collections import OrderedDict
from collections.abc import Callable

import torch
from torch.optim import Optimizer

from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, OptimizerInstanceOrClass, ParamGroup, State
from pytorch_optimizer.optimizer.utils.foreach import group_tensors_by_device_and_dtype


class OrthoGrad(BaseOptimizer):
    """Wrap an optimizer with gradients orthogonal to the current parameters.

    Args:
        optimizer: Base optimizer.
        foreach: Batch gradient projections with additional temporary memory. `None` enables batching on CUDA.

    """

    def __init__(self, optimizer: OptimizerInstanceOrClass, foreach: bool | None = None, **kwargs) -> None:
        self._optimizer_step_pre_hooks: dict[int, Callable] = OrderedDict()
        self._optimizer_step_post_hooks: dict[int, Callable] = OrderedDict()
        self._patch_step_function()
        self.eps: float = 1e-30
        self.foreach = foreach

        self.optimizer: Optimizer = self.load_optimizer(optimizer, **kwargs)

        self.defaults: Defaults = self.optimizer.defaults

    def __str__(self) -> str:
        return 'OrthoGrad'

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self) -> State:
        return self.optimizer.state

    def state_dict(self) -> State:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: State) -> None:
        self.optimizer.load_state_dict(state_dict)

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

    @torch.no_grad()
    def apply_orthogonal_gradients(self, params) -> None:
        params = list(params)

        if not self.can_use_foreach({'params': params}, self.foreach, default_on_cuda=True):
            super().apply_orthogonal_gradients(params, eps=self.eps)
            return

        params = [p for p in params if p.grad is not None]

        for batch in group_tensors_by_device_and_dtype(params, [p.grad for p in params]):
            dtype = torch.float64 if batch['params'][0].dtype == torch.float64 else torch.float32
            weights = [p.reshape(-1).to(dtype=dtype) for p in batch['params']]
            grads = [grad.reshape(-1).to(dtype=dtype) for grad in batch['grads']]

            dots = torch.stack([product.sum() for product in torch._foreach_mul(weights, grads)])
            weight_norms = torch.stack(torch._foreach_norm(weights))
            projections = dots / weight_norms.square_().add_(self.eps)
            orthogonal = torch._foreach_mul(weights, projections.unbind())
            torch._foreach_neg_(orthogonal)
            torch._foreach_add_(orthogonal, grads)

            grad_norms = torch.stack(torch._foreach_norm(grads))
            orthogonal_norms = torch.stack(torch._foreach_norm(orthogonal))
            scales = torch.where(
                orthogonal_norms <= 4.0 * torch.finfo(dtype).eps * grad_norms,
                0.0,
                grad_norms / orthogonal_norms.add(self.eps),
            )
            torch._foreach_mul_(orthogonal, scales.unbind())
            torch._foreach_copy_(
                batch['grads'], [update.reshape_as(grad) for update, grad in zip(orthogonal, batch['grads'])]
            )

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        if closure is None:
            for group in self.param_groups:
                self.apply_orthogonal_gradients(group['params'])
            return self.optimizer.step()

        def orthogonal_closure():
            with torch.enable_grad():
                loss = closure()
            for group in self.param_groups:
                self.apply_orthogonal_gradients(group['params'])
            return loss

        return self.optimizer.step(orthogonal_closure)
