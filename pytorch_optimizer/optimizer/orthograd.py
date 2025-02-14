from typing import Callable, Dict

import torch
from torch.optim import Optimizer

from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import CLOSURE, DEFAULTS, LOSS, OPTIMIZER_INSTANCE_OR_CLASS, STATE


class OrthoGrad(BaseOptimizer):
    r"""Grokking at the Edge of Numerical Stability.

    A wrapper optimizer that projects gradients to be orthogonal to the current parameters before performing an update.

    :param optimizer: OPTIMIZER_INSTANCE_OR_CLASS. base optimizer.
    """

    def __init__(self, optimizer: OPTIMIZER_INSTANCE_OR_CLASS, **kwargs) -> None:
        self._optimizer_step_pre_hooks: Dict[int, Callable] = {}
        self._optimizer_step_post_hooks: Dict[int, Callable] = {}
        self.eps: float = 1e-30

        self.optimizer: Optimizer = self.load_optimizer(optimizer, **kwargs)

        self.defaults: DEFAULTS = self.optimizer.defaults

    def __str__(self) -> str:
        return 'OrthoGrad'

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self) -> STATE:
        return self.optimizer.state

    def state_dict(self) -> STATE:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: STATE) -> None:
        self.optimizer.load_state_dict(state_dict)

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def reset(self):
        pass

    @torch.no_grad()
    def apply_orthogonal_gradients(self, params) -> None:
        for p in params:
            if p.grad is None or p.grad.is_sparse:
                continue

            w = p.view(-1)
            g = p.grad.view(-1)

            proj = torch.dot(w, g).div_(torch.dot(w, w).add_(self.eps))
            g_ortho = g.to(dtype=torch.float32, copy=True).sub_(w, alpha=proj)
            g_ortho_scaled = g_ortho.mul_(g.norm(2).div_(g_ortho.norm(2).add_(self.eps)))

            p.grad.copy_(g_ortho_scaled.view_as(p.grad))

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        for group in self.param_groups:
            self.apply_orthogonal_gradients(group['params'])
        return self.optimizer.step(closure)
