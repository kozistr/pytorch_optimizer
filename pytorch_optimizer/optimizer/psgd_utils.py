
import torch
from torch.linalg import vector_norm


def damped_pair_vg(g: torch.Tensor, damp: float = 2 ** -13) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
    """Sample a noise vector and pair it with a damped gradient.

    Adds `damp * mean(abs(g)) * v` to the gradient to stabilize preconditioner updates.

    Args:
        g: Gradient tensor.
        damp: Noise damping coefficient.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Noise vector and damped gradient.

    Reference: https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_with_finite_precision_arithmetic.py

    """
    v = torch.randn_like(g)
    return v, g + damp * torch.mean(torch.abs(g)) * v


def norm_lower_bound(a: torch.Tensor) -> torch.Tensor:
    """Estimate a lower bound for the spectral norm of a matrix.

    Args:
        a: Matrix to inspect. The function rescales it in place.

    Returns:
        torch.Tensor: Lower bound estimate of the spectral norm.

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
    """Update `inv_a` in place to the inverse of `A + U @ V`.

    Args:
        inv_a: Inverse of `A`, overwritten with the updated inverse.
        u: Left factor of the low rank update.
        v: Right factor of the low rank update.

    Note:
        Repeated updates can accumulate numerical error.

    """
    inv_au = inv_a @ u
    v_inv_au = v @ inv_au

    ident = torch.eye(v_inv_au.shape[0], dtype=v_inv_au.dtype, device=v_inv_au.device)
    inv_a.sub_(inv_au @ torch.linalg.solve(ident + v_inv_au, v @ inv_a))


def triu_with_diagonal_and_above(a: torch.Tensor) -> torch.Tensor:
    """Return the diagonal plus twice the strictly upper triangular entries.

    Approximates the triangular correction in a QR decomposition of `I + A` for small `A`.

    Args:
        a: Matrix to transform.

    Returns:
        torch.Tensor: `triu(a, 0) + triu(a, 1)`.

    """
    return torch.triu(a, diagonal=0) + torch.triu(a, diagonal=1)


def update_precondition_dense(
    q: torch.Tensor, dxs: list[torch.Tensor], dgs: list[torch.Tensor], step: float = 0.01, eps: float = 1.2e-38
) -> torch.Tensor:
    """Update the Cholesky factor of a dense preconditioner from parameter gradient perturbations.

    Args:
        q: Cholesky factor of preconditioner with positive diagonal entries.
        dxs: List of perturbations of parameters.
        dgs: List of perturbations of gradients.
        step: Update step size normalized to range [0, 1].
        eps: An offset to avoid division by zero.

    """
    dx = torch.cat([torch.reshape(x, [-1, 1]) for x in dxs])
    dg = torch.cat([torch.reshape(g, [-1, 1]) for g in dgs])

    a = q.mm(dg)
    b = torch.linalg.solve_triangular(q.t(), dx, upper=False)

    grad = torch.triu(a.mm(a.t()) - b.mm(b.t()))

    return q - (step / norm_lower_bound(grad).add_(eps)) * grad.mm(q)
