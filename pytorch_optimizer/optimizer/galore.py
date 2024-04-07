from typing import Literal, Optional, Tuple, Union

import torch

PROJECTION_TYPE = Literal['std', 'reverse_std', 'right', 'left', 'full']


class GaLoreProjector:
    r"""Memory-Efficient LLM Training by Gradient Low-Rank Projection.

    :param rank: int. matrix rank.
    :param update_proj_gap: int. num steps to update the projection.
    :param scale: float. scale factor.
    :param projection_type: PROJECTION_TYPE. type of projection. 'std', 'reverse_std', 'right', 'left', 'full' are
        supported.
    """

    def __init__(
        self, rank: int, update_proj_gap: int = 200, scale: float = 1.0, projection_type: PROJECTION_TYPE = 'std'
    ):
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.projection_type = projection_type

        self.ortho_matrix: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None

    @staticmethod
    def get_orthogonal_matrix(
        weights: torch.Tensor, rank: int, projection_type: str
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if projection_type not in {'right', 'left', 'full'}:
            raise ValueError('projection_type should be one of left, right or full')

        original_type = weights.data.dtype
        original_device = weights.data.device
        is_float: bool = original_type == torch.float

        u, s, vh = torch.linalg.svd(weights if is_float else weights.float(), full_matrices=False)

        if projection_type == 'right':
            b = vh[:rank, :]
            return b if is_float else b.to(original_device).type(original_type)
        if projection_type == 'left':
            a = u[:, :rank]
            return a if is_float else a.to(original_device).type(original_type)

        a = u[:, :rank]
        b = vh[:rank, :]

        return (
            (a, b)
            if is_float
            else (a.to(original_device).type(original_type), b.to(original_device).type(original_type))
        )

    def get_low_rank_grad_std(self, grad: torch.Tensor, steps: int) -> torch.Tensor:
        if grad.shape[0] >= grad.shape[1]:
            if self.ortho_matrix is None or steps % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(grad, self.rank, projection_type='right')
            return torch.matmul(grad, self.ortho_matrix.t())

        if self.ortho_matrix is None or steps % self.update_proj_gap == 0:
            self.ortho_matrix = self.get_orthogonal_matrix(grad, self.rank, projection_type='left')

        return torch.matmul(self.ortho_matrix.t(), grad)

    def get_low_rank_grad_reverse_std(self, grad: torch.Tensor, steps: int) -> torch.Tensor:
        if grad.shape[0] >= grad.shape[1]:
            if self.ortho_matrix is None or steps % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(grad, self.rank, projection_type='left')
            return torch.matmul(self.ortho_matrix.t(), grad)

        if self.ortho_matrix is None or steps % self.update_proj_gap == 0:
            self.ortho_matrix = self.get_orthogonal_matrix(grad, self.rank, projection_type='right')

        return torch.matmul(grad, self.ortho_matrix.t())

    def get_low_rank_grad_right(self, grad: torch.Tensor, steps: int) -> torch.Tensor:
        if self.ortho_matrix is None or steps % self.update_proj_gap == 0:
            self.ortho_matrix = self.get_orthogonal_matrix(grad, self.rank, projection_type='right')
        return torch.matmul(grad, self.ortho_matrix.t())

    def get_low_rank_grad_left(self, grad: torch.Tensor, steps: int) -> torch.Tensor:
        if self.ortho_matrix is None or steps % self.update_proj_gap == 0:
            self.ortho_matrix = self.get_orthogonal_matrix(grad, self.rank, projection_type='left')
        return torch.matmul(self.ortho_matrix.t(), grad)

    def get_low_rank_grad_full(self, grad: torch.Tensor, steps: int) -> torch.Tensor:
        if self.ortho_matrix is None or steps % self.update_proj_gap == 0:
            self.ortho_matrix = self.get_orthogonal_matrix(grad, self.rank, projection_type='full')
        return torch.matmul(self.ortho_matrix[0].t(), grad) @ self.ortho_matrix[1].t()

    def project(self, full_rank_grad: torch.Tensor, steps: int) -> torch.Tensor:
        if self.projection_type == 'std':
            return self.get_low_rank_grad_std(full_rank_grad, steps)
        if self.projection_type == 'reverse_std':
            return self.get_low_rank_grad_reverse_std(full_rank_grad, steps)
        if self.projection_type == 'right':
            return self.get_low_rank_grad_right(full_rank_grad, steps)
        if self.projection_type == 'left':
            return self.get_low_rank_grad_right(full_rank_grad, steps)
        if self.projection_type == 'full':
            return self.get_low_rank_grad_full(full_rank_grad, steps)
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
                if low_rank_grad.shape[0] <= low_rank_grad.shape[1]
                else torch.matmul(low_rank_grad, self.ortho_matrix)
            ) * self.scale
        if self.projection_type == 'right':
            return torch.matmul(low_rank_grad, self.ortho_matrix) * self.scale
        if self.projection_type == 'left':
            return torch.matmul(self.ortho_matrix, low_rank_grad) * self.scale
        if self.projection_type == 'full':
            return torch.matmul(self.ortho_matrix[0], low_rank_grad) @ self.ortho_matrix[1] * self.scale

        raise NotImplementedError
