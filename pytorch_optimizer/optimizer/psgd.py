import math
from collections.abc import Callable
from string import ascii_lowercase, ascii_uppercase
from typing import Literal, cast

import numpy as np
import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.psgd_utils import norm_lower_bound

MEMORY_SAVE_MODE_TYPE = Literal['one_diag', 'smart_one_diag', 'all_diag']


def precondition_update_prob_schedule(
    max_prob: float = 1.0, min_prob: float = 0.03, decay: float = 0.001, flat_start: int = 500
) -> Callable[[int], torch.Tensor]:
    """Create an exponential decay schedule for preconditioner update frequency.

    Args:
        max_prob: Initial update frequency as a fraction of steps.
        min_prob: Minimum update frequency.
        decay: Exponential decay rate after the flat initial period.
        flat_start: Number of steps to keep the initial frequency.

    Returns:
        Callable: Function mapping the step index to an update frequency tensor.

    """

    def _schedule(n: int) -> torch.Tensor:
        """Compute the update probability with exponential decay after the initial constant period."""
        prob = max_prob * torch.exp(-decay * (torch.tensor(n, dtype=torch.float32) - flat_start))
        prob.clamp_(min=min_prob, max=max_prob)
        return prob

    return _schedule


class Kron(BaseOptimizer):
    """Preconditioned SGD with Kronecker factored preconditioners.

    Args:
        params: Parameters to optimize or dictionaries defining parameter groups.
        lr: Learning rate.
        momentum: Momentum factor.
        weight_decay: Weight decay coefficient.
        weight_decouple: Apply weight decay to parameters instead of adding it to the gradient.
        pre_conditioner_update_probability: Update frequency as a fraction or a callable of the step index. `None`
            uses the default decay schedule.
        max_size_triangular: Largest dimension that can use a triangular preconditioner.
        min_ndim_triangular: Minimum tensor dimensionality for triangular preconditioners.
        memory_save_mode: Diagonal storage policy: `None`, `'one_diag'`, `'smart_one_diag'`, or `'all_diag'`.
        momentum_into_precondition_update: Use momentum instead of raw gradients when updating preconditioners.
        mu_dtype: Dtype of the momentum accumulator.
        precondition_dtype: Dtype of the preconditioner.
        balance_prob: Probability of performing balancing.
        maximize: Maximize the objective instead of minimizing it.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        pre_conditioner_update_probability: float | Callable[[int], torch.Tensor] | None = None,
        max_size_triangular: int = 8192,
        min_ndim_triangular: int = 2,
        memory_save_mode: MEMORY_SAVE_MODE_TYPE | None = None,
        momentum_into_precondition_update: bool = True,
        mu_dtype: torch.dtype | None = None,
        precondition_dtype: torch.dtype | None = torch.float32,
        balance_prob: float = 0.01,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')

        self.balance_prob: float = balance_prob
        self.eps: float = torch.finfo(torch.bfloat16).tiny
        self.maximize = maximize

        defaults = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'pre_conditioner_update_probability': pre_conditioner_update_probability,
            'max_size_triangular': max_size_triangular,
            'min_ndim_triangular': min_ndim_triangular,
            'memory_save_mode': memory_save_mode,
            'momentum_into_precondition_update': momentum_into_precondition_update,
            'precondition_lr': 1e-1,
            'precondition_init_scale': 1.0,
            'mu_dtype': mu_dtype,
            'precondition_dtype': precondition_dtype,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Kron'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        if 'step' not in group:
            group['step'] = 0

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        first_group = self.param_groups[0]
        update_prob = first_group['pre_conditioner_update_probability']
        if update_prob is None:
            update_prob = precondition_update_prob_schedule()(first_group.get('step', 0))
        if callable(update_prob):
            update_prob = cast(torch.Tensor, update_prob(first_group.get('step', 0)))

        update_counter = first_group.get('update_counter', 0) + 1
        do_update: bool = update_counter >= 1 / update_prob
        first_group['update_counter'] = 0 if do_update else update_counter

        balance: bool = np.random.random() < self.balance_prob and do_update

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            bias_correction1: float = self.debias(group['momentum'], group['step'])

            mu_dtype, precondition_dtype = group['mu_dtype'], group['precondition_dtype']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                if torch.is_complex(p):
                    raise NoComplexParameterError(str(self))

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p, dtype=mu_dtype or p.dtype)
                    state['Q'], state['expressions'] = initialize_q_expressions(
                        p,
                        group['precondition_init_scale'],
                        group['max_size_triangular'],
                        group['min_ndim_triangular'],
                        group['memory_save_mode'],
                        dtype=precondition_dtype,
                    )

                momentum_buffer = state['momentum_buffer']
                momentum_buffer.mul_(group['momentum']).add_(grad, alpha=1.0 - group['momentum'])

                if mu_dtype is not None:
                    momentum_buffer = momentum_buffer.to(dtype=mu_dtype, non_blocking=True)

                de_biased_momentum = (momentum_buffer / bias_correction1).to(
                    dtype=precondition_dtype, non_blocking=True
                )

                if grad.dim() > 1 and balance:
                    balance_q(state['Q'])

                if do_update:
                    update_precondition(
                        state['Q'],
                        state['expressions'],
                        torch.randn_like(de_biased_momentum, dtype=precondition_dtype),
                        de_biased_momentum if group['momentum_into_precondition_update'] else grad,
                        group['precondition_lr'],
                        self.eps,
                    )

                precondition_grad = get_precondition_grad(state['Q'], state['expressions'], de_biased_momentum).to(
                    dtype=p.dtype, non_blocking=True
                )

                precondition_grad.mul_(torch.clamp(1.1 / (precondition_grad.square().mean().sqrt() + 1e-6), max=1.0))

                if group['weight_decay'] != 0 and p.dim() >= 2:
                    precondition_grad.add_(p, alpha=group['weight_decay'])

                p.add_(precondition_grad, alpha=-group['lr'])

        return loss


def initialize_q_expressions(
    t: torch.Tensor,
    scale: float,
    max_size: int,
    min_ndim_triangular: int,
    memory_save_mode: MEMORY_SAVE_MODE_TYPE | None,
    dtype: torch.dtype | None = None,
) -> tuple[list[torch.Tensor], tuple[str, list[str], str]]:
    """Initialize Kronecker preconditioner factors and reusable einsum expressions.

    For a scalar or tensor t, we initialize its preconditioner Q and reusable einsum expressions for updating Q and
    preconditioning gradient.
    """
    letters: str = ascii_lowercase + ascii_uppercase

    t_dtype: torch.dtype = dtype if dtype is not None else t.dtype
    shape = t.shape
    if len(shape) == 0:
        qs: list[torch.Tensor] = [scale * torch.ones_like(t, dtype=t_dtype)]
        expressions_a: str = ',->'
        expression_gr: list[str] = [',->']
        expression_r: str = ',,->'

        return qs, (expressions_a, expression_gr, expression_r)

    if len(shape) > 13:
        raise ValueError(f'got tensor with dim {len(t.shape)}. Einstein runs out of letters!')

    scale = math.pow(scale, 1.0 / len(shape))

    if memory_save_mode is None:
        dim_diag = [False for _ in shape]
    elif memory_save_mode == 'one_diag':
        dim_diag = [False for _ in shape]
        dim_diag[np.argsort(shape)[::-1][0]] = True
    elif memory_save_mode == 'smart_one_diag':
        dim_diag = [False for _ in shape]
        sorted_shape = sorted(shape)
        if len(shape) >= 2 and sorted_shape[-1] > sorted_shape[-2]:
            dim_diag[np.argsort(shape)[::-1][0]] = True
    elif memory_save_mode == 'all_diag':
        dim_diag = [True for _ in shape]
    else:
        raise NotImplementedError(
            f'invalid memory_save_mode {memory_save_mode}. '
            'it must be one of [None, `one_diag`, `smart_one_diag`, `all_diag`]'
        )

    qs: list[torch.Tensor] = []
    expr_gr = []
    piece_1a, piece_2a, piece_3a = [], '', ''
    piece_1p, piece_2p, piece_3p, piece_4p = [], [], '', ''
    for i, (size, dim_d) in enumerate(zip(shape, dim_diag)):
        if size == 1 or size > max_size or len(shape) < min_ndim_triangular or dim_d:
            qs.append(scale * torch.ones(size, dtype=t_dtype, device=t.device))

            piece_1a.append(letters[i])
            piece_2a += letters[i]
            piece_3a += letters[i]

            piece1: str = ''.join([(letters[i + 13] if j == i else letters[j]) for j in range(len(shape))])
            expr_gr.append(f'{piece1},{piece1}->{letters[i + 13]}')

            piece_1p.append(letters[i + 13])
            piece_2p.append(letters[i + 13])
            piece_3p += letters[i + 13]
            piece_4p += letters[i + 13]
        else:
            qs.append(scale * torch.eye(size, dtype=t_dtype, device=t.device))

            piece_1a.append(letters[i] + letters[i + 13])
            piece_2a += letters[i + 13]
            piece_3a += letters[i]

            piece1: str = ''.join([(letters[i + 13] if j == i else letters[j]) for j in range(len(shape))])
            piece2: str = ''.join([(letters[i + 26] if j == i else letters[j]) for j in range(len(shape))])
            expr_gr.append(f'{piece1},{piece2}->{letters[i + 13]}{letters[i + 26]}')

            a, b, c = letters[i], letters[i + 13], letters[i + 26]
            piece_1p.append(a + b)
            piece_2p.append(a + c)
            piece_3p += c
            piece_4p += b

    expr_a: str = ','.join(piece_1a) + f',{piece_2a}->{piece_3a}'
    expr_r: str = ','.join(piece_1p) + ',' + ','.join(piece_2p) + f',{piece_3p}->{piece_4p}'

    return qs, (expr_a, expr_gr, expr_r)


def balance_q(q_in: list[torch.Tensor]) -> None:
    """Balance the norms of Kronecker preconditioner factors in place."""
    norms = torch.stack([q.norm(float('inf')) for q in q_in])
    geometric_mean = norms.prod() ** (1 / len(q_in))
    norms = geometric_mean / norms
    for i, q in enumerate(q_in):
        q.mul_(norms[i])


def solve_triangular_right(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Compute `X @ inv(A)` using a triangular solve."""
    orig_dtype: torch.dtype = x.dtype
    x = x.to(dtype=torch.float32, non_blocking=True)
    a = a.to(dtype=torch.float32, non_blocking=True)
    out = torch.linalg.solve_triangular(a, x.reshape(-1, x.size(-1)), upper=True, left=False).reshape_as(x)
    return out.to(dtype=orig_dtype, non_blocking=True)


def get_a_and_conj_b(
    expr_a: str,
    g: torch.Tensor,
    qs: list[torch.Tensor],
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute transformed gradient and noise terms for preconditioner updates."""
    a = torch.einsum(expr_a, *qs, g)

    order: int = g.dim()
    p = list(range(order))

    conj_b = torch.permute(v.conj(), p[1:] + p[:1])
    for i, q in enumerate(qs):
        conj_b = conj_b / q if q.dim() < 2 else solve_triangular_right(conj_b, q)
        if i < order - 1:
            conj_b = torch.transpose(conj_b, i, order - 1)

    return a, conj_b


def get_q_terms(expr_gs: list[str], a: torch.Tensor, conj_b: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Compute factor wise terms for a Kronecker preconditioner update."""
    terms: list = []
    for expr_g in expr_gs:
        term1 = torch.einsum(expr_g, a, a.conj())
        term2 = torch.einsum(expr_g, conj_b.conj(), conj_b)
        terms.append((term1, term2))
    return terms


def update_precondition(
    qs: list[torch.Tensor],
    expressions: tuple[str, list[str], str],
    v: torch.Tensor,
    g: torch.Tensor,
    step: int,
    eps: float,
) -> None:
    """Update Kronecker preconditioner factors from a noise gradient pair."""
    expr_a, expr_gs, _ = expressions

    a, conj_b = get_a_and_conj_b(expr_a, g, qs, v)

    q_terms: list[tuple[torch.Tensor, torch.Tensor]] = get_q_terms(expr_gs, a, conj_b)

    for q, (term1, term2) in zip(qs, q_terms):
        tmp = term1 - term2
        tmp *= step

        if q.dim() < 2:
            tmp *= q
            tmp.div_((term1 + term2).norm(float('inf')).add_(eps))
        else:
            tmp = torch.triu(tmp)
            tmp.div_(norm_lower_bound(term1 + term2).add_(eps))
            tmp @= q

        q.sub_(tmp)


def get_precondition_grad(qs: list[torch.Tensor], expressions: list[str], g: torch.Tensor) -> torch.Tensor:
    """Apply Kronecker preconditioner factors to a gradient."""
    return torch.einsum(expressions[-1], *[x.conj() for x in qs], *qs, g)
