import math
from typing import Literal, Optional, Tuple, Union

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS

PROJECTION_TYPE = Literal['std', 'reverse_std', 'right', 'left', 'full']


class GaLoreProjector:
    r"""Memory-Efficient LLM Training by Gradient Low-Rank Projection.

    :param rank: int. low rank to project.
    :param update_proj_gap: int. num steps to update the projection.
    :param scale: float. scale factor.
    :param projection_type: PROJECTION_TYPE. type of projection. 'std', 'reverse_std', 'right', 'left', 'full' are
        supported.
    """

    def __init__(
        self,
        rank: int = 128,
        update_proj_gap: int = 50,
        scale: float = 1.0,
        projection_type: PROJECTION_TYPE = 'std',
        **kwargs,
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
            return self.get_low_rank_grad_left(full_rank_grad, steps)
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
                torch.matmul(self.ortho_matrix, low_rank_grad.t())
                if low_rank_grad.shape[0] <= low_rank_grad.shape[1]
                else torch.matmul(low_rank_grad, self.ortho_matrix.t())
            ) * self.scale
        if self.projection_type == 'right':
            return torch.matmul(low_rank_grad, self.ortho_matrix.t()) * self.scale
        if self.projection_type == 'left':
            return torch.matmul(self.ortho_matrix, low_rank_grad) * self.scale
        if self.projection_type == 'full':
            return torch.matmul(self.ortho_matrix[0], low_rank_grad) @ self.ortho_matrix[1].t() * self.scale

        raise NotImplementedError


class GaLore(BaseOptimizer):
    r"""AdamW optimizer with GaLore projector.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace.
    :param weight_decay: float. weight decay (L2 penalty).
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        betas: BETAS = (0.9, 0.999),
        weight_decay: float = 0.0,
        eps: float = 1e-6,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(eps, 'eps')

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'eps': eps,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'GaLore'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            beta1, beta2 = group['betas']

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

            step_size: float = group['lr'] * bias_correction2_sq / bias_correction1

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                if 'rank' in group and p.dim() > 1:
                    if 'projector' not in state:
                        state['projector'] = GaLoreProjector(
                            rank=group['rank'],
                            update_proj_gap=group['update_proj_gap'],
                            scale=group['scale'],
                            projection_type=group['projection_type'],
                        )

                    grad = state['projector'].project(grad, group['step'])

                self.apply_weight_decay(
                    p=p,
                    grad=None,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=True,
                    fixed_decay=False,
                )

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                de_nom = exp_avg_sq.sqrt().add_(group['eps'])

                norm_grad = exp_avg / de_nom

                if 'rank' in group and p.dim() > 1:
                    norm_grad = state['projector'].project_back(norm_grad)

                p.add_(norm_grad, alpha=-step_size)

        return loss
