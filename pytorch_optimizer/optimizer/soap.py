import math
from collections import defaultdict
from itertools import chain

import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, DataFormat, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils.shape import merge_small_dims


class SOAP(BaseOptimizer):
    """Adam updates in Shampoo preconditioner eigenbases.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        betas: Decay rates for the first and second moments.
        shampoo_beta: Decay rate for preconditioner statistics. `None` uses `beta2`.
        weight_decay: Weight decay coefficient.
        precondition_frequency: Number of steps between eigenbasis updates.
        max_precondition_dim: Largest dimension to precondition. Larger dimensions use an identity transform.
        merge_dims: Whether to merge dimensions of the preconditioner.
        precondition_1d: Whether to precondition 1D gradients.
        correct_bias: Whether to correct bias in Adam.
        normalize_gradient: Whether to normalize the gradients.
        data_format: Tensor layout for dimension merging: `'channels_first'` (NCHW) or `'channels_last'` (NHWC).
        eps: Term added to the denominator to improve numerical stability.
        maximize: Maximize the objective instead of minimizing it.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-3,
        betas: Betas = (0.95, 0.95),
        shampoo_beta: float | None = None,
        weight_decay: float = 1e-2,
        precondition_frequency: int = 10,
        max_precondition_dim: int = 10000,
        merge_dims: bool = False,
        precondition_1d: bool = False,
        correct_bias: bool = True,
        normalize_gradient: bool = False,
        data_format: DataFormat = 'channels_first',
        eps: float = 1e-8,
        maximize: bool = False,
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
        self.maximize = maximize

        defaults: Defaults = {
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

    def merge_dims(self, grad: torch.Tensor, max_precondition_dim: int) -> torch.Tensor:
        """Merge dimensions after converting channels-last tensors to channels-first."""
        if self.data_format == 'channels_last' and grad.dim() == 4:
            grad = grad.permute(0, 3, 1, 2)

        return grad.reshape(merge_small_dims(grad.size(), max_precondition_dim))

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

        _, beta2 = group['betas']
        initialized: list[dict] = []

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

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
                    update_basis=False,
                )
                initialized.append(state)

        matrices = iter(self.get_orthogonal_matrix(
            [m for state in initialized for m in state['GG']],
            batched=[
                group['step'] == 0 and (len(m) == 0 or state['exp_avg_sq'].numel() >= m.numel())
                for state in initialized for m in state['GG']
            ],
        ))
        for state in initialized:
            state['Q'] = [next(matrices) for _ in state['GG']]
        self.get_orthogonal_matrix_qr(
            [
                state for state in initialized
                if group['step'] > 0 and group['step'] % state['precondition_frequency'] == 0
            ],
            group['max_precondition_dim'], group['merge_dims'], batched=False,
        )

    def project(
        self,
        grad: torch.Tensor,
        state,
        merge_dims: bool = False,
        max_precondition_dim: int = 10000,
        project_type: str = 'forward',
    ) -> torch.Tensor:
        original_shape = grad.shape
        permuted_shape = original_shape

        do_permute: bool = self.data_format == 'channels_last' and len(original_shape) == 4

        if merge_dims:
            if do_permute:
                permuted_shape = grad.permute(0, 3, 1, 2).shape

            grad = self.merge_dims(grad, max_precondition_dim)

        for mat in state['Q']:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat, dims=[[0], [0 if project_type == 'forward' else 1]])
            else:
                grad = grad.permute([*list(range(1, len(grad.shape))), 0])

        if merge_dims:
            grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1) if do_permute else grad.reshape(original_shape)

        return grad

    @staticmethod
    def get_orthogonal_matrix(mat: list, use_qr: bool = False, batched: bool | list[bool] = True) -> list:
        matrices: list = [[] for _ in mat]
        groups: defaultdict[tuple, list[int]] = defaultdict(list)
        for index, m in enumerate(mat):
            if len(m) == 0:
                continue
            batchable = batched if isinstance(batched, bool) else batched[index]
            groups[m.device, m.dtype, m.shape, None if batchable else index].append(index)

        for indices in groups.values():
            batch = mat[indices[0]] if len(indices) == 1 else torch.stack([mat[index] for index in indices])
            if use_qr:
                q, _ = torch.linalg.qr(batch.float())
                q = q.to(batch.dtype)
            else:
                try:
                    _, q = torch.linalg.eigh(
                        batch + 1e-30 * torch.eye(batch.size(-1), device=batch.device, dtype=batch.dtype)
                    )
                except Exception:  # pragma: no cover
                    _, q = torch.linalg.eigh(
                        batch.double() + 1e-30 * torch.eye(batch.size(-1), device=batch.device, dtype=torch.float64)
                    )
                    q = q.to(batch.dtype)
                q = q.flip(-1)

            for index, basis in zip(indices, [q] if q.ndim == 2 else q.unbind()):
                matrices[index] = basis

        return matrices

    def get_orthogonal_matrix_qr(
        self,
        state: dict | list[dict],
        max_precondition_dim: int = 10000,
        merge_dims: bool = False,
        batched: bool = True,
    ):
        """Compute the eigenbases of the preconditioner using one round of power iteration."""
        states = [state] if isinstance(state, dict) else state
        power_iters: list = []
        batchable: list[bool] = []
        for current_state in states:
            original_shape = current_state['exp_avg_sq'].shape
            exp_avg_sq = current_state['exp_avg_sq']
            permuted_shape = original_shape
            if self.data_format == 'channels_last' and len(original_shape) == 4:
                permuted_shape = exp_avg_sq.permute(0, 3, 1, 2).shape
            if merge_dims:
                exp_avg_sq = self.merge_dims(exp_avg_sq, max_precondition_dim)

            for ind, (m, o) in enumerate(zip(current_state['GG'], current_state['Q'])):
                if len(m) == 0:
                    power_iters.append([])
                    batchable.append(False)
                    continue

                matrix_product = m @ o
                sort_idx = torch.argsort((o * matrix_product).sum(dim=0), descending=True)
                exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
                power_iters.append(matrix_product.index_select(1, sort_idx))
                batchable.append(batched and exp_avg_sq.numel() >= m.numel())

            if merge_dims:
                exp_avg_sq = (
                    exp_avg_sq.reshape(permuted_shape).permute(0, 2, 3, 1)
                    if self.data_format == 'channels_last' and len(original_shape) == 4
                    else exp_avg_sq.reshape(original_shape)
                )
            current_state['exp_avg_sq'] = exp_avg_sq

        matrices = self.get_orthogonal_matrix(power_iters, use_qr=True, batched=batchable)
        bases = iter(matrices)
        for current_state in states:
            current_state['Q'] = [next(bases) for _ in current_state['GG']]
        return matrices

    def init_pre_conditioner(
        self,
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
                state['GG'].append(torch.zeros(grad.shape[0], grad.shape[0], device=grad.device, dtype=grad.dtype))
        else:
            if merge_dims:
                grad = self.merge_dims(grad, max_precondition_dim)

            for sh in grad.shape:
                if sh > max_precondition_dim:
                    state['GG'].append([])
                else:
                    state['GG'].append(torch.zeros(sh, sh, device=grad.device, dtype=grad.dtype))

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
        update_basis: bool = True,
    ) -> None:
        if grad.dim() == 1:
            if precondition_1d and grad.shape[0] <= max_precondition_dim:
                state['GG'][0].lerp_(
                    (grad.unsqueeze(1) @ grad.unsqueeze(0)).to(state['GG'][0].dtype),
                    weight=1.0 - state['shampoo_beta'],
                )
        else:
            if merge_dims:
                grad = self.merge_dims(grad, max_precondition_dim)

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

        if update_basis:
            if state['Q'] is None:
                state['Q'] = self.get_orthogonal_matrix(state['GG'], batched=False)
            if step > 0 and step % state['precondition_frequency'] == 0:
                self.get_orthogonal_matrix_qr(state, max_precondition_dim, merge_dims, batched=False)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.init_group(group)
            group['step'] += 1

            if group['step'] == 1:
                continue

            beta1, beta2 = group['betas']

            step: int = group['step'] - 1
            refresh_states: list[dict] = []
            step_size: float = group['lr']
            if group['correct_bias']:
                bias_correction1: float = self.debias(beta1, step)
                bias_correction2_sq: float = math.sqrt(self.debias(beta2, step))

                step_size *= bias_correction2_sq / bias_correction1

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                grad_projected = self.project(
                    grad, state, merge_dims=group['merge_dims'], max_precondition_dim=group['max_precondition_dim']
                )

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                exp_avg.lerp_(grad, weight=1.0 - beta1)
                exp_avg_sq.lerp_(grad_projected.square(), weight=1.0 - beta2)

                de_nom = exp_avg_sq.sqrt().add_(group['eps'])

                exp_avg_projected = self.project(
                    exp_avg, state, merge_dims=group['merge_dims'], max_precondition_dim=group['max_precondition_dim']
                )

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
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=True,
                    fixed_decay=False,
                )

                self.update_pre_conditioner(
                    grad,
                    state,
                    step=step,
                    max_precondition_dim=group['max_precondition_dim'],
                    merge_dims=group['merge_dims'],
                    precondition_1d=group['precondition_1d'],
                    update_basis=False,
                )
                if step % state['precondition_frequency'] == 0:
                    refresh_states.append(state)

            if refresh_states:
                self.get_orthogonal_matrix_qr(
                    refresh_states, group['max_precondition_dim'], group['merge_dims'],
                )

        return loss
