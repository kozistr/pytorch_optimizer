from typing import List, Tuple

import torch
from torch.linalg import vector_norm


def damped_pair_vg(g: torch.Tensor, damp: float = 2 ** -13) -> Tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
    r"""Get damped pair v and g.

    Instead of return (v, g), it returns pair (v, g + sqrt(eps)*mean(abs(g))*v)
    such that the covariance matrix of the modified g is lower bound by eps * (mean(abs(g)))**2 * I
    This should damp the pre-conditioner to encourage numerical stability.
    The default amount of damping is 2**(-13), slightly smaller than sqrt(eps('single')).

    If v is integrated out, let's just use the modified g;
    If hvp is used, recommend to use L2 regularization to lower bound the Hessian, although this method also works.

    Please check example
        https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_with_finite_precision_arithmetic.py
    for the rationale to set default damping level to 2**(-13).
    """
    v = torch.randn_like(g)
    return v, g + damp * torch.mean(torch.abs(g)) * v


def norm_lower_bound(a: torch.Tensor) -> torch.Tensor:
    r"""Get a cheap lower bound for the spectral norm of A.

    Numerical results on random matrices with a wide range of distributions and sizes suggest,
    norm(A) <= sqrt(2) * norm_lower_bound(A)
    Looks to be a very tight lower bound.
    """
    max_abs = torch.max(torch.abs(a))
    if max_abs <= 0:
        return max_abs

    a.div_(max_abs)

    aa = torch.real(a * a.conj())
    value0, i = torch.max(torch.sum(aa, dim=0), 0)
    value1, j = torch.max(torch.sum(aa, dim=1), 0)

    if value0 > value1:
        x = a[:, i].conj() @ a
        return max_abs * vector_norm((x / vector_norm(x)) @ a.H)

    x = a @ a[j].conj()
    return max_abs * vector_norm(a.H @ (x / vector_norm(x)))


def woodbury_identity(inv_a: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> None:
    r"""Get the Woodbury identity.

    inv(A + U * V) = inv(A) - inv(A) * U * inv(I + V * inv(A) * U) * V * inv(A)

    with inplace update of inv_a.

    Note that using the Woodbury identity multiple times could accumulate numerical errors.
    """
    inv_au = inv_a @ u
    v_inv_au = v @ inv_au

    ident = torch.eye(v_inv_au.shape[0], dtype=v_inv_au.dtype, device=v_inv_au.device)
    inv_a.sub_(inv_au @ torch.linalg.solve(ident + v_inv_au, v @ inv_a))


def triu_with_diagonal_and_above(a: torch.Tensor) -> torch.Tensor:
    r"""Get triu with diagonal and above.

    It is useful as for a small A, the R of QR decomposition qr(I + A) is about I + triu(A, 0) + triu(A, 1)
    """
    return torch.triu(a, diagonal=0) + torch.triu(a, diagonal=1)


def update_precondition_dense(
    q: torch.Tensor, dxs: List[torch.Tensor], dgs: List[torch.Tensor], step: float = 0.01, eps: float = 1.2e-38
) -> torch.Tensor:
    r"""Update dense pre-conditioner P = Q^T * Q.

    :param q: torch.Tensor. Cholesky factor of pre-conditioner with positive diagonal entries.
    :param dxs: List[torch.Tensor]. list of perturbations of parameters.
    :param dgs: List[torch.Tensor]. list of perturbations of gradients.
    :param step: float. update step size normalized to range [0, 1].
    :param eps: float. an offset to avoid division by zero.
    """
    dx = torch.cat([torch.reshape(x, [-1, 1]) for x in dxs])
    dg = torch.cat([torch.reshape(g, [-1, 1]) for g in dgs])

    a = q.mm(dg)
    b = torch.linalg.solve_triangular(q.t(), dx, upper=False)

    grad = torch.triu(a.mm(a.t()) - b.mm(b.t()))

    return q - (step / norm_lower_bound(grad).add_(eps)) * grad.mm(q)
