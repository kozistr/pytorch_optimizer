from typing import Callable, Dict

import torch
from torch.optim import Optimizer

from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, OptimizerInstanceOrClass, ParamGroup, State


class OrthoGrad(BaseOptimizer):
    """Grokking at the Edge of Numerical Stability.

    A wrapper optimizer that projects gradients to be orthogonal to the current parameters before performing an update.

    Args:
        optimizer (OptimizerInstanceOrClass): Base optimizer.

    """

    def __init__(self, optimizer: OptimizerInstanceOrClass, **kwargs) -> None:
        self._optimizer_step_pre_hooks: Dict[int, Callable] = {}
        self._optimizer_step_post_hooks: Dict[int, Callable] = {}
        self.eps: float = 1e-30

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
        super().apply_orthogonal_gradients(params, eps=self.eps)

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
