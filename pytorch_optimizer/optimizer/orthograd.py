import torch

from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import CLOSURE, LOSS, OPTIMIZER, PARAMETERS


class OrthoGrad(BaseOptimizer):
    r"""Grokking at the Edge of Numerical Stability.

    A wrapper optimizer that projects gradients to be orthogonal to the current parameters before performing an update.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param optimizer: OPTIMIZER. base optimizer.
    """

    def __init__(self, params: PARAMETERS, optimizer: OPTIMIZER = torch.optim.AdamW, **kwargs):
        self.eps: float = 1e-30

        super().__init__(params, {})
        self.base_optimizer = optimizer(self.param_groups, **kwargs)

    def __str__(self) -> str:
        return 'OrthoGrad'

    @torch.no_grad()
    def reset(self):
        pass

    @torch.no_grad()
    def orthogonalize_gradients(self, params) -> None:
        for p in params:
            if p.grad is None:
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
            self.orthogonalize_gradients(group['params'])
        return self.base_optimizer.step(closure)
