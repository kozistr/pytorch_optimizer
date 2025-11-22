import math
from typing import Literal, Optional, Tuple, Union

import torch

PROJECTION_TYPE = Literal['std', 'reverse_std', 'right', 'left', 'full', 'random']


class GaLoreProjector:
    """Memory-Efficient LLM Training by Gradient Low-Rank Projection.

    Args:
        rank (Optional[int]): Low rank to project. If None, the full matrix is used.
        update_proj_gap (int): Number of steps between projection updates.
        scale (float): Scale factor applied during projection.
        projection_type (PROJECTION_TYPE): Type of projection. Supported types include 'std', 'reverse_std',
            'right', 'left', 'full', and 'random'.
    """

    def __init__(
        self,
        rank: Optional[int] = 128,
        update_proj_gap: int = 50,
        scale: float = 1.0,
        projection_type: PROJECTION_TYPE = 'std',
        **kwargs,
    ) -> None:
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.projection_type = projection_type

        self.ortho_matrix: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None
        self.last_svd_step: int = -1

    @staticmethod
    def get_orthogonal_matrix(
        weights: torch.Tensor, rank: Optional[int], projection_type: str, from_random_matrix: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if projection_type not in ('right', 'left', 'full'):
            raise ValueError('`projection_type` should be one of left, right or full')

        original_type = weights.data.dtype
        original_device = weights.data.device
        is_float: bool = original_type == torch.float

        if not from_random_matrix:
            u, _, vh = torch.linalg.svd(weights if is_float else weights.float(), full_matrices=False)
        elif isinstance(rank, int):
            u = torch.randn((weights.size(0), rank), device=original_device, dtype=original_type) / math.sqrt(rank)
            vh = torch.randn((rank, weights.size(1)), device=original_device, dtype=original_type) / math.sqrt(rank)
        else:
            raise TypeError('`rank` should be int when `from_random_matrix` is True')

        if projection_type == 'right':
            b = vh[:rank, :] if isinstance(rank, int) else vh
            return b if is_float else b.to(original_device).type(original_type)
        if projection_type == 'left':
            a = u[:, :rank] if isinstance(rank, int) else u
            return a if is_float else a.to(original_device).type(original_type)

        a = u[:, :rank] if isinstance(rank, int) else u
        b = vh[:rank, :] if isinstance(rank, int) else vh

        return (
            (a, b)
            if is_float
            else (a.to(original_device).type(original_type), b.to(original_device).type(original_type))
        )

    def get_low_rank_grad_std(self, grad: torch.Tensor) -> torch.Tensor:
        if grad.shape[0] >= grad.shape[1]:
            return torch.matmul(grad, self.ortho_matrix.t())
        return torch.matmul(self.ortho_matrix.t(), grad)

    def get_low_rank_grad_reverse_std(self, grad: torch.Tensor) -> torch.Tensor:
        if grad.shape[0] >= grad.shape[1]:
            return torch.matmul(self.ortho_matrix.t(), grad)
        return torch.matmul(grad, self.ortho_matrix.t())

    def get_low_rank_grad_right(self, grad: torch.Tensor) -> torch.Tensor:
        return torch.matmul(grad, self.ortho_matrix.t())

    def get_low_rank_grad_left(self, grad: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.ortho_matrix.t(), grad)

    def get_low_rank_grad_full(self, grad: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.ortho_matrix[0].t(), grad) @ self.ortho_matrix[1].t()

    def get_low_rank_grad_random(self, grad: torch.Tensor) -> torch.Tensor:
        is_right: bool = grad.size(0) >= grad.size(1)
        return torch.matmul(grad, self.ortho_matrix.t()) if is_right else torch.matmul(self.ortho_matrix.t(), grad)

    def update_ortho_matrix(self, x: torch.Tensor, from_random_matrix: bool) -> None:
        is_right: bool = x.size(0) >= x.size(1)

        if self.projection_type == 'std':
            self.ortho_matrix = self.get_orthogonal_matrix(
                x, self.rank, projection_type='right' if is_right else 'left', from_random_matrix=from_random_matrix
            )
        elif self.projection_type == 'reverse_std':
            self.ortho_matrix = self.get_orthogonal_matrix(
                x, self.rank, projection_type='left' if is_right else 'right', from_random_matrix=from_random_matrix
            )
        elif self.projection_type == 'right':
            self.ortho_matrix = self.get_orthogonal_matrix(
                x, self.rank, projection_type='right', from_random_matrix=from_random_matrix
            )
        elif self.projection_type == 'left':
            self.ortho_matrix = self.get_orthogonal_matrix(
                x, self.rank, projection_type='left', from_random_matrix=from_random_matrix
            )
        elif self.projection_type == 'full':
            self.ortho_matrix = self.get_orthogonal_matrix(
                x, self.rank, projection_type='full', from_random_matrix=from_random_matrix
            )
        elif self.projection_type == 'random':
            self.ortho_matrix = self.get_orthogonal_matrix(
                x,
                self.rank,
                projection_type='right' if is_right else 'left',
                from_random_matrix=from_random_matrix,
            )
        else:
            raise NotImplementedError(f'unsupported projection_type: {self.projection_type}')

    def project(
        self,
        grad: torch.Tensor,
        num_steps: int,
        svd_basis_matrix: Optional[torch.Tensor] = None,
        from_random_matrix: bool = False,
    ) -> torch.Tensor:
        update_ortho_matrix: bool = self.ortho_matrix is None or num_steps % self.update_proj_gap == 0
        already_updated_this_step: bool = num_steps == self.last_svd_step

        if update_ortho_matrix and not already_updated_this_step:
            self.update_ortho_matrix(
                x=grad if svd_basis_matrix is None else svd_basis_matrix,
                from_random_matrix=from_random_matrix,
            )
            self.last_svd_step = num_steps

        if self.projection_type == 'std':
            return self.get_low_rank_grad_std(grad)
        if self.projection_type == 'reverse_std':
            return self.get_low_rank_grad_reverse_std(grad)
        if self.projection_type == 'right':
            return self.get_low_rank_grad_right(grad)
        if self.projection_type == 'left':
            return self.get_low_rank_grad_left(grad)
        if self.projection_type == 'full':
            return self.get_low_rank_grad_full(grad)
        if self.projection_type == 'random':
            return self.get_low_rank_grad_random(grad)

        raise NotImplementedError

    def project_back(self, low_rank_grad: torch.Tensor) -> torch.Tensor:
        if self.projection_type == 'std':
            return (
                torch.matmul(low_rank_grad, self.ortho_matrix)
                if low_rank_grad.shape[0] >= low_rank_grad.shape[1]
                else torch.matmul(self.ortho_matrix, low_rank_grad)
            ) * self.scale
        if self.projection_type == 'reverse_std':
            return (
                torch.matmul(self.ortho_matrix, low_rank_grad)
                if low_rank_grad.shape[0] > low_rank_grad.shape[1]
                else torch.matmul(low_rank_grad, self.ortho_matrix)
            ) * self.scale
        if self.projection_type == 'right':
            return torch.matmul(low_rank_grad, self.ortho_matrix) * self.scale
        if self.projection_type == 'left':
            return torch.matmul(self.ortho_matrix, low_rank_grad) * self.scale
        if self.projection_type == 'full':
            return torch.matmul(self.ortho_matrix[0], low_rank_grad) @ self.ortho_matrix[1] * self.scale
        if self.projection_type == 'random':
            return (
                torch.matmul(low_rank_grad, self.ortho_matrix)
                if low_rank_grad.shape[0] >= low_rank_grad.shape[1]
                else torch.matmul(self.ortho_matrix, low_rank_grad)
            ) * self.scale

        raise NotImplementedError
