import random
from collections.abc import Iterable
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from pytorch_optimizer.base.optimizer import BaseOptimizer


def flatten_grad(grads: list[torch.Tensor]) -> torch.Tensor:
    """Flatten and concatenate a list of gradient tensors."""
    return torch.cat([grad.flatten() for grad in grads])


def un_flatten_grad(grads: torch.Tensor, shapes: list[int]) -> list[torch.Tensor]:
    """Restore a flat gradient to tensors with the supplied shapes."""
    idx: int = 0
    un_flatten_grads: list[torch.Tensor] = []
    for shape in shapes:
        length = np.prod(shape)
        un_flatten_grads.append(grads[idx:idx + length].view(shape).clone())  # fmt: skip
        idx += int(length)
    return un_flatten_grads


class PCGrad(BaseOptimizer):
    """Wrap an optimizer with gradient projection for conflicting task objectives.

    Args:
        optimizer: Optimizer instance.
        reduction: Reduction method for gradients.

    """

    def __init__(self, optimizer: Optimizer, reduction: str = 'mean'):
        self.validate_options(reduction, 'reduction', ['mean', 'sum'])

        self.optimizer = optimizer
        self.reduction = reduction

    @torch.no_grad()
    def init_group(self):
        self.zero_grad()

    def zero_grad(self):
        return self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        return self.optimizer.step()

    def set_grad(self, grads: list[torch.Tensor], has_grads: list[torch.Tensor] | None = None) -> None:
        idx: int = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                p.grad = grads[idx] if has_grads is None or torch.any(has_grads[idx]) else None
                idx += 1

    def retrieve_grad(self) -> tuple[list[torch.Tensor], list[int], list[torch.Tensor]]:
        """Collect gradients, shapes, and masks for parameters with gradients."""
        grad, shape, has_grad = [], [], []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p, device=p.device))
                    has_grad.append(torch.zeros_like(p, device=p.device))
                    continue

                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p, device=p.device))

        return grad, shape, has_grad

    def pack_grad(self, objectives: Iterable) -> tuple[list[torch.Tensor], list[list[int]], list[torch.Tensor]]:
        """Compute and flatten gradients for each task loss.

        Args:
            objectives: Scalar task loss tensors to backpropagate.

        """
        grads, shapes, has_grads = [], [], []
        for objective in objectives:
            self.optimizer.zero_grad(set_to_none=True)
            objective.backward(retain_graph=True)

            grad, shape, has_grad = self.retrieve_grad()

            grads.append(flatten_grad(grad))
            has_grads.append(flatten_grad(has_grad))
            shapes.append(shape)

        return grads, shapes, has_grads

    def project_conflicting(self, grads: list[torch.Tensor], has_grads: list[torch.Tensor]) -> torch.Tensor:
        """Remove conflicting task gradient components and combine task gradients.

        Args:
            grads: A list of the gradient of the parameters.
            has_grads: A list of masks representing whether the parameter has gradient.

        """
        shared: torch.Tensor = torch.stack(has_grads).prod(0).bool()

        pc_grad: list[torch.Tensor] = deepcopy(grads)
        for i, g_i in enumerate(pc_grad):
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j: torch.Tensor = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    pc_grad[i] -= g_i_g_j * g_j / (g_j.norm() ** 2)

        merged_grad: torch.Tensor = torch.zeros_like(grads[0])

        shared_pc_gradients: torch.Tensor = torch.stack([g[shared] for g in pc_grad])
        if self.reduction == 'mean':
            merged_grad[shared] = shared_pc_gradients.mean(dim=0)
        else:
            merged_grad[shared] = shared_pc_gradients.sum(dim=0)

        merged_grad[~shared] = torch.stack([g[~shared] for g in pc_grad]).sum(dim=0)

        return merged_grad

    def pc_backward(self, objectives: Iterable[nn.Module]) -> None:
        """Set parameter gradients after projecting conflicting task gradients.

        Args:
            objectives: Scalar task loss tensors to backpropagate.

        """
        grads, shapes, has_grads = self.pack_grad(objectives)

        pc_grad = self.project_conflicting(grads, has_grads)
        pc_grad = un_flatten_grad(pc_grad, shapes[0])
        has_grad = un_flatten_grad(torch.stack(has_grads).sum(dim=0), shapes[0])

        self.set_grad(pc_grad, has_grad)
