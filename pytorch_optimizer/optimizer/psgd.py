import math
from string import ascii_lowercase, ascii_uppercase
from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import CLOSURE, LOSS, PARAMETERS
from pytorch_optimizer.optimizer.psgd_utils import norm_lower_bound

MEMORY_SAVE_MODE_TYPE = Literal['one_diag', 'smart_one_diag', 'all_diag']


def precondition_update_prob_schedule(
    max_prob: float = 1.0, min_prob: float = 0.03, decay: float = 0.001, flat_start: int = 500
) -> Callable[[int], torch.Tensor]:
    """Anneal pre-conditioner update probability during beginning of training.

    PSGD benefits from more pre-conditioner updates at the beginning of training, but once the pre-conditioner is
    learned the update probability can drop low.

    This schedule is an exponential anneal with a flat start. Default settings keep update probability at 1.0 for 200
    steps then exponentially anneal down to `min_prob` by 4000 steps. Default settings work very well for most models
    and training regimes.
    """

    def _schedule(n: int) -> torch.Tensor:
        """Exponential anneal with flat start."""
        n = torch.tensor(n, dtype=torch.float32)
        prob = max_prob * torch.exp(-decay * (n - flat_start))
        prob.clamp_(min=min_prob, max=max_prob)
        return prob

    return _schedule


class Kron(BaseOptimizer):
    """PSGD with the Kronecker product pre-conditioner.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param momentum: float. momentum factor.
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    :param pre_conditioner_update_probability: Optional[Tuple[Callable, float]]. Probability of updating the
        pre-conditioner. If None, defaults to a schedule that anneals from 1.0 to 0.03 by 4000 steps.
    :param max_size_triangular: int. max size for dim's pre-conditioner to be triangular.
    :param min_ndim_triangular: int. minimum number of dimensions a layer needs to have triangular pre-conditioners.
    :param memory_save_mode: Optional[str]. None, 'one_diag', or 'all_diag', None is default to set all
        pre-conditioners to be triangular, 'one_diag' sets the largest or last dim to be diagonal per layer, and
        'all_diag' sets all pre-conditioners to be diagonal.
    :param momentum_into_precondition_update: bool. whether to send momentum into pre-conditioner update instead of
        raw gradients.
    :param mu_dtype: Optional[torch.dtype]. dtype of the momentum accumulator.
    :param precondition_dtype: torch.dtype. dtype of the pre-conditioner.
    :param balance_prob: float. probability of performing balancing.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        pre_conditioner_update_probability: Optional[Tuple[Callable, float]] = None,
        max_size_triangular: int = 8192,
        min_ndim_triangular: int = 2,
        memory_save_mode: Optional[MEMORY_SAVE_MODE_TYPE] = None,
        momentum_into_precondition_update: bool = True,
        mu_dtype: Optional[torch.dtype] = None,
        precondition_dtype: Optional[torch.dtype] = torch.float32,
        balance_prob: float = 0.01,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')

        if pre_conditioner_update_probability is None:
            pre_conditioner_update_probability = precondition_update_prob_schedule()

        self.balance_prob: float = balance_prob
        self.eps: float = torch.finfo(torch.bfloat16).tiny
        self.prob_step: int = 0
        self.update_counter: int = 0

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

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['momentum_buffer'] = p.grad.clone()

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        update_prob: Union[float, Callable] = self.param_groups[0]['pre_conditioner_update_probability']
        if callable(update_prob):
            update_prob = update_prob(self.prob_step)

        self.update_counter += 1
        do_update: bool = self.update_counter >= 1 / update_prob
        if do_update:
            self.update_counter = 0
        self.prob_step += 1

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
    memory_save_mode: Optional[MEMORY_SAVE_MODE_TYPE],
    dtype: Optional[torch.dtype] = None,
) -> Tuple[List[torch.Tensor], Tuple[str, List[str], str]]:
    r"""Initialize Q expressions.

    For a scalar or tensor t, we initialize its pre-conditioner Q and reusable einsum expressions for updating Q and
    pre-conditioning gradient.
    """
    letters: str = ascii_lowercase + ascii_uppercase

    dtype: torch.dtype = dtype if dtype is not None else t.dtype
    shape = t.shape
    if len(shape) == 0:
        qs: list[torch.Tensor] = [scale * torch.ones_like(t, dtype=dtype)]
        expressions_a: str = ',->'
        expression_gr: List[str] = [',->']
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
            'it must be one of [None, \'one_diag\', \'smart_one_diag\', \'all_diag\']'
        )

    qs: List[torch.Tensor] = []
    expr_gr = []
    piece_1a, piece_2a, piece_3a = [], '', ''
    piece_1p, piece_2p, piece_3p, piece_4p = [], [], '', ''
    for i, (size, dim_d) in enumerate(zip(shape, dim_diag)):
        if size == 1 or size > max_size or len(shape) < min_ndim_triangular or dim_d:
            qs.append(scale * torch.ones(size, dtype=dtype, device=t.device))

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
            qs.append(scale * torch.eye(size, dtype=dtype, device=t.device))

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


def balance_q(q_in: List[torch.Tensor]) -> None:
    r"""Balance Q."""
    norms = torch.stack([q.norm(float('inf')) for q in q_in])
    geometric_mean = norms.prod() ** (1 / len(q_in))
    norms = geometric_mean / norms
    for i, q in enumerate(q_in):
        q.mul_(norms[i])


def solve_triangular_right(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    r"""Calculate X @ inv(A)."""
    orig_dtype: torch.dtype = x.dtype
    x = x.to(dtype=torch.float32, non_blocking=True)
    a = a.to(dtype=torch.float32, non_blocking=True)
    out = torch.linalg.solve_triangular(a, x.reshape(-1, x.size(-1)), upper=True, left=False).reshape_as(x)
    return out.to(dtype=orig_dtype, non_blocking=True)


def get_a_and_conj_b(
    expr_a: List[str], g: torch.Tensor, qs: List[torch.Tensor], v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Get A and b.conj."""
    a = torch.einsum(expr_a, *qs, g)

    order: int = g.dim()
    p = list(range(order))

    conj_b = torch.permute(v.conj(), p[1:] + p[:1])
    for i, q in enumerate(qs):
        conj_b = conj_b / q if q.dim() < 2 else solve_triangular_right(conj_b, q)
        if i < order - 1:
            conj_b = torch.transpose(conj_b, i, order - 1)

    return a, conj_b


def get_q_terms(expr_gs: List[str], a: torch.Tensor, conj_b: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    r"""Get Q terms."""
    terms: List = []
    for expr_g in expr_gs:
        term1 = torch.einsum(expr_g, a, a.conj())
        term2 = torch.einsum(expr_g, conj_b.conj(), conj_b)
        terms.append((term1, term2))
    return terms


def update_precondition(
    qs: List[torch.Tensor],
    expressions: List[Tuple[str, List[str], str]],
    v: torch.Tensor,
    g: torch.Tensor,
    step: int,
    eps: float,
) -> None:
    r"""Update Kronecker product pre-conditioner Q with pair (V, G)."""
    expr_a, expr_gs, _ = expressions

    a, conj_b = get_a_and_conj_b(expr_a, g, qs, v)

    q_terms: List[Tuple[torch.Tensor, torch.Tensor]] = get_q_terms(expr_gs, a, conj_b)

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
    r"""Precondition gradient G with pre-conditioner Q."""
    return torch.einsum(expressions[-1], *[x.conj() for x in qs], *qs, g)
