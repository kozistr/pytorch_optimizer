import random
from copy import deepcopy
from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from pytorch_optimizer.base.optimizer import BaseOptimizer


def flatten_grad(grads: List[torch.Tensor]) -> torch.Tensor:
    """Flatten the gradient."""
    return torch.cat([grad.flatten() for grad in grads])


def un_flatten_grad(grads: torch.Tensor, shapes: List[int]) -> List[torch.Tensor]:
    """Unflatten the gradient."""
    idx: int = 0
    un_flatten_grads: List[torch.Tensor] = []
    for shape in shapes:
        length = np.prod(shape)
        un_flatten_grads.append(grads[idx:idx + length].view(shape).clone())  # fmt: skip
        idx += length
    return un_flatten_grads


class PCGrad(BaseOptimizer):
    """Gradient Surgery for Multi-Task Learning.

    Args:
        optimizer (Optimizer): Optimizer instance.
        reduction (str): Reduction method for gradients.
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

    def set_grad(self, grads: List[torch.Tensor]) -> None:
        idx: int = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                p.grad = grads[idx]
                idx += 1

    def retrieve_grad(self) -> Tuple[List[torch.Tensor], List[int], List[torch.Tensor]]:
        """Get the gradient of the parameters of the network with specific objective."""
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

    def pack_grad(self, objectives: Iterable) -> Tuple[List[torch.Tensor], List[List[int]], List[torch.Tensor]]:
        """Pack the gradient of the parameters of the network for each objective.

        Args:
            objectives (Iterable[nn.Module]): A list of objectives.
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

    def project_conflicting(self, grads: List[torch.Tensor], has_grads: List[torch.Tensor]) -> torch.Tensor:
        """Project conflicting.

        Args:
            grads (List[torch.Tensor]): A list of the gradient of the parameters.
            has_grads (List[torch.Tensor]): A list of masks representing whether the parameter has gradient.
        """
        shared: torch.Tensor = torch.stack(has_grads).prod(0).bool()

        pc_grad: List[torch.Tensor] = deepcopy(grads)
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
        """Calculate the gradient of the parameters.

        Args:
            objectives (Iterable[nn.Module]): A list of objectives.
        """
        grads, shapes, has_grads = self.pack_grad(objectives)

        pc_grad = self.project_conflicting(grads, has_grads)
        pc_grad = un_flatten_grad(pc_grad, shapes[0])

        self.set_grad(pc_grad)
