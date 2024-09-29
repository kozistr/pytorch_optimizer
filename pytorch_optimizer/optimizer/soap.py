import math
from itertools import chain
from typing import List, Optional

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DATA_FORMAT, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.optimizer.shampoo_utils import merge_small_dims


class SOAP(BaseOptimizer):
    r"""Improving and Stabilizing Shampoo using Adam.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param betas: BETAS. coefficients used for computing running averages of gradient and the squared hessian trace
    :param shampoo_beta: Optional[float]. if not None, use this beta for the pre-conditioner (L and R in paper,
        state['GG'] below) moving average instead of betas[1].
    :param weight_decay: float. weight decay (L2 penalty).
    :param precondition_frequency: int. how often to update the pre-conditioner.
    :param max_precondition_dim: int. maximum dimension of the pre-conditioner. Set to 10000, so that we exclude most
        common vocab sizes while including layers.
    :param merge_dims: bool. whether to merge dimensions of the pre-conditioner
    :param precondition_1d: bool. whether to precondition 1D gradients.
    :param correct_bias: bool. whether to correct bias in Adam.
    :param normalize_gradient: bool. whether to normalize the gradients.
    :param eps: float. term added to the denominator to improve numerical stability.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 3e-3,
        betas: BETAS = (0.95, 0.95),
        shampoo_beta: Optional[float] = None,
        weight_decay: float = 1e-2,
        precondition_frequency: int = 10,
        max_precondition_dim: int = 10000,
        merge_dims: bool = False,
        precondition_1d: bool = False,
        correct_bias: bool = True,
        normalize_gradient: bool = False,
        data_format: DATA_FORMAT = 'channels_first',
        eps: float = 1e-8,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(shampoo_beta, 'shampoo_beta')
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_positive(precondition_frequency, 'precondition_frequency')
        self.validate_positive(max_precondition_dim, 'max_precondition_dim')
        self.validate_non_negative(eps, 'eps')

        self.data_format = data_format

        defaults: DEFAULTS = {
            'lr': lr,
            'betas': betas,
            'shampoo_beta': shampoo_beta,
            'weight_decay': weight_decay,
            'precondition_frequency': precondition_frequency,
            'max_precondition_dim': max_precondition_dim,
            'merge_dims': merge_dims,
            'precondition_1d': precondition_1d,
            'correct_bias': correct_bias,
            'normalize_gradient': normalize_gradient,
            'eps': eps,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SOAP'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

    def project(
        self,
        grad: torch.Tensor,
        state,
        merge_dims: bool = False,
        max_precondition_dim: int = 10000,
        project_type: str = 'forward',
    ) -> torch.Tensor:
        original_shape = grad.shape

        if merge_dims:
            if self.data_format == 'channels_last' and grad.dim() == 4:
                permuted_shape = grad.permute(0, 3, 1, 2).shape

            grad = grad.reshape(merge_small_dims(grad.size(), max_precondition_dim))

        for mat in state['Q']:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat, dims=[[0], [0 if project_type == 'forward' else 1]])
            else:
                grad = grad.permute([*list(range(1, len(grad.shape))), 0])

        if merge_dims:
            if self.data_format == 'channels_last' and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)

        return grad

    @staticmethod
    def get_orthogonal_matrix(mat: torch.Tensor) -> List[torch.Tensor]:
        matrices: List = []
        for m in mat:
            if len(m) == 0:
                matrices.append([])
                continue

            try:
                _, q = torch.linalg.eigh(m + 1e-30 * torch.eye(m.shape[0], device=m.device))
            except Exception:  # pragma: no cover
                _, q = torch.linalg.eigh(m.to(torch.float64) + 1e-30 * torch.eye(m.shape[0], device=m.device))
                q = q.to(m.dtype)

            q = torch.flip(q, dims=[1])

            matrices.append(q)

        return matrices

    def get_orthogonal_matrix_qr(self, state, max_precondition_dim: int = 10000, merge_dims: bool = False):
        r"""Compute the eigen-bases of the pre-conditioner using one round of power iteration."""
        orig_shape = state['exp_avg_sq'].shape
        if self.data_format == 'channels_last' and len(orig_shape) == 4:
            permuted_shape = state['exp_avg_sq'].permute(0, 3, 1, 2).shape

        exp_avg_sq = state['exp_avg_sq']
        if merge_dims:
            exp_avg_sq = exp_avg_sq.reshape(merge_small_dims(exp_avg_sq.size(), max_precondition_dim))

        matrices = []
        for ind, (m, o) in enumerate(zip(state['GG'], state['Q'])):
            if len(m) == 0:
                matrices.append([])
                continue

            est_eig = torch.diag(o.T @ m @ o)
            sort_idx = torch.argsort(est_eig, descending=True)
            exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)

            power_iter = m @ o[:, sort_idx]

            q, _ = torch.linalg.qr(power_iter)

            matrices.append(q)

        if merge_dims:
            if self.data_format == 'channels_last' and len(orig_shape) == 4:
                exp_avg_sq = exp_avg_sq.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                exp_avg_sq = exp_avg_sq.reshape(orig_shape)

        state['exp_avg_sq'] = exp_avg_sq

        return matrices

    @staticmethod
    def init_pre_conditioner(
        grad,
        state,
        precondition_frequency: int = 10,
        shampoo_beta: float = 0.95,
        max_precondition_dim: int = 10000,
        precondition_1d: bool = False,
        merge_dims: bool = False,
    ) -> None:
        state['GG'] = []
        if grad.dim() == 1:
            if not precondition_1d or grad.shape[0] > max_precondition_dim:
                state['GG'].append([])
            else:
                state['GG'].append(torch.zeros(grad.shape[0], grad.shape[0], device=grad.device))
        else:
            if merge_dims:
                grad = grad.reshape(merge_small_dims(grad.size(), max_precondition_dim))

            for sh in grad.shape:
                if sh > max_precondition_dim:
                    state['GG'].append([])
                else:
                    state['GG'].append(torch.zeros(sh, sh, device=grad.device))

        state['Q'] = None
        state['precondition_frequency'] = precondition_frequency
        state['shampoo_beta'] = shampoo_beta

    def update_pre_conditioner(
        self,
        grad,
        state,
        step: int,
        max_precondition_dim: int = 10000,
        precondition_1d: bool = False,
        merge_dims: bool = False,
    ) -> None:
        if grad.dim() == 1:
            if precondition_1d and grad.shape[0] <= max_precondition_dim:
                state['GG'][0].lerp_(
                    (grad.unsqueeze(1) @ grad.unsqueeze(0)).to(state['GG'][0].dtype),
                    weight=1.0 - state['shampoo_beta'],
                )
        else:
            if merge_dims:
                grad = grad.reshape(merge_small_dims(grad.size(), max_precondition_dim))

            for idx, dim in enumerate(grad.shape):
                if dim <= max_precondition_dim:
                    outer_product = torch.tensordot(
                        grad,
                        grad,
                        dims=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2,
                    )

                    state['GG'][idx].lerp_(
                        outer_product.to(state['GG'][idx].dtype), weight=1.0 - state['shampoo_beta']
                    )

        if state['Q'] is None:
            state['Q'] = self.get_orthogonal_matrix(state['GG'])

        if step > 0 and step % state['precondition_frequency'] == 0:
            state['Q'] = self.get_orthogonal_matrix_qr(state, max_precondition_dim, merge_dims)

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
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_avg_sq'] = torch.zeros_like(grad)

                    self.init_pre_conditioner(
                        grad,
                        state,
                        precondition_frequency=group['precondition_frequency'],
                        shampoo_beta=group['shampoo_beta'] if group['shampoo_beta'] is not None else beta2,
                        max_precondition_dim=group['max_precondition_dim'],
                        precondition_1d=group['precondition_1d'],
                        merge_dims=group['merge_dims'],
                    )

                    self.update_pre_conditioner(
                        grad,
                        state,
                        step=group['step'],
                        max_precondition_dim=group['max_precondition_dim'],
                        precondition_1d=group['precondition_1d'],
                        merge_dims=group['merge_dims'],
                    )

                    continue

                grad_projected = self.project(
                    grad, state, merge_dims=group['merge_dims'], max_precondition_dim=group['max_precondition_dim']
                )

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad_projected.square(), alpha=1.0 - beta2)

                de_nom = exp_avg_sq.sqrt().add_(group['eps'])

                exp_avg_projected = self.project(
                    exp_avg, state, merge_dims=group['merge_dims'], max_precondition_dim=group['max_precondition_dim']
                )

                step_size = group['lr']
                if group['correct_bias']:
                    bias_correction1: float = self.debias(beta1, group['step'])
                    bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

                    step_size *= bias_correction2_sq / bias_correction1

                norm_grad = self.project(
                    exp_avg_projected / de_nom,
                    state,
                    merge_dims=group['merge_dims'],
                    max_precondition_dim=group['max_precondition_dim'],
                    project_type='backward',
                )

                if group['normalize_gradient']:
                    norm_grad.div_(torch.mean(norm_grad.square()).sqrt_().add_(group['eps']))

                p.add_(norm_grad, alpha=-step_size)

                self.apply_weight_decay(
                    p,
                    grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=True,
                    fixed_decay=False,
                )

                self.update_pre_conditioner(
                    grad,
                    state,
                    step=group['step'],
                    max_precondition_dim=group['max_precondition_dim'],
                    merge_dims=group['merge_dims'],
                    precondition_1d=group['precondition_1d'],
                )

        return loss
