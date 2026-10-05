import torch

DTensor = torch.Tensor
has_dtensor: bool = False

try:  # pragma: no cover
    from torch.distributed.tensor import DTensor

    has_dtensor = True
except ImportError:  # pragma: no cover
    try:
        from torch.distributed._tensor import DTensor  # type: ignore[attr-defined]

        has_dtensor = True
    except ImportError:
        pass

NewtonSchulzWeight = tuple[float, float, float]
NewtonSchulzWeights = str | NewtonSchulzWeight | list[NewtonSchulzWeight] | tuple[NewtonSchulzWeight, ...]

NS_COEFFICIENTS = {
    'original': [
        # Keller Jordan's Muon.
        (3.4445, -4.7750, 2.0315),
    ],
    'quintic': [
        # Optimized quintic schedules from modded-nanogpt.
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ],
    'polar_express': [
        # Polar Express schedule with safety 1e-2.
        (8.237312490495555, -23.157747414558198, 16.680568411445915),
        (4.082441999064835, -2.893047735332586, 0.5252849256975648),
        (3.9263479922546582, -2.8547468034765298, 0.5318022422894988),
        (3.2982187133085143, -2.424541981026706, 0.48632008358844075),
        (2.2970369434552573, -1.63662558125903, 0.4002628455953627),
        (1.8763805351440397, -1.2347896577722228, 0.35891887501668385),
        (1.8564423485617974, -1.2132449880935525, 0.3568003487825883),
        (1.8749994008682747, -1.2499988017229169, 0.3749994008546422),
    ],
    'polar_express_safer': [
        # Polar Express safer schedule with safety 2e-2.
        (8.156554524902461, -22.48329292557795, 15.878769915207462),
        (4.0429299351667245, -2.808917465908704, 0.5000178451051299),
        (3.8916678022926563, -2.7724841532176825, 0.5060648178503389),
        (3.285753657755658, -2.3681294933425394, 0.46449024233003117),
        (2.3005307116270983, -1.6111665557258408, 0.3833374427545273),
        (1.8631210546382593, -1.2042160621002727, 0.3421879560523383),
        (1.8382572152247512, -1.1779263289537742, 0.3396513038637379),
        (1.8749999923301852, -1.2499999836060613, 0.374999991275876),
    ],
}


def get_newton_schulz_weights(weights: NewtonSchulzWeights) -> list[NewtonSchulzWeight]:
    """Resolve a coefficient preset or sequence into Newton-Schulz quintic weights."""
    if isinstance(weights, str):
        key = weights.lower()
        if key in NS_COEFFICIENTS:
            return list(NS_COEFFICIENTS[key])

        raise ValueError(f'Invalid `weights` string choice. expected one of {tuple(NS_COEFFICIENTS.keys())}.')

    # Single coefficient tuple, e.g. (3.4445, -4.7750, 2.0315).
    if isinstance(weights, tuple) and len(weights) == 3:
        w0, w1, w2 = weights
        if isinstance(w0, (int, float)) and isinstance(w1, (int, float)) and isinstance(w2, (int, float)):
            return [(float(w0), float(w1), float(w2))]

    if len(weights) == 0:
        raise ValueError('`weights` schedule must not be empty.')

    normalized: list[NewtonSchulzWeight] = []
    for coeff in weights:
        if not isinstance(coeff, tuple) or len(coeff) != 3:
            raise ValueError('`weights` must be a preset name, a coefficient tuple, or a list of coefficient tuples.')

        c0, c1, c2 = coeff
        normalized.append((float(c0), float(c1), float(c2)))

    return normalized


@torch.no_grad()
def power_iteration(mat_g: torch.Tensor, num_iters: int = 100) -> torch.Tensor:
    """Estimate the largest eigenvalue of a positive semidefinite matrix.

    Args:
        mat_g: Symmetric positive semidefinite matrix.
        num_iters: Number of power iterations.

    Returns:
        torch.Tensor: Estimated largest eigenvalue.

    """
    v = torch.randn(mat_g.shape[0], dtype=mat_g.dtype, device=mat_g.device)
    mat_v = torch.empty_like(v)

    for _ in range(num_iters):
        torch.mv(mat_g, v, out=mat_v)
        v.copy_(mat_v)
        v.div_(torch.linalg.norm(v))

    return (v.t() @ mat_g @ v).clamp_min_(1e-16)


@torch.inference_mode()
def compute_power_schur_newton(
    mat_g: torch.Tensor,
    p: int,
    max_iters: int = 100,
    error_tolerance: float = 1e-3,
    ridge_epsilon: float = 1e-6,
    max_error_ratio: float = 1.2,
) -> torch.Tensor:
    """Compute a regularized matrix inverse root with coupled Schur-Newton iteration.

    Reference:
        Guo and Higham, A Schur-Newton Method for the Matrix p-th Root and its Inverse (2006).
        https://pdfs.semanticscholar.org/0abe/7f77433cf5908bfe2b79aa91af881da83858.pdf

    Args:
        mat_g: Square positive semidefinite matrix.
        p: Positive integer root order.
        max_iters: Maximum number of iterations.
        error_tolerance: Residual threshold for stopping the iteration.
        ridge_epsilon: Diagonal regularization, scaled by the estimated largest eigenvalue.
        max_error_ratio: Maximum allowed ratio between successive residual errors.

    Returns:
        torch.Tensor: Approximation to the regularized matrix raised to `-1 / p`.

    """
    shape: torch.Size = mat_g.shape
    if len(shape) == 1:
        return torch.pow(mat_g + ridge_epsilon, -1.0 / p)

    identity = torch.eye(shape[0], dtype=mat_g.dtype, device=mat_g.device)
    if shape[0] == 1:
        return identity

    mat_g += power_iteration(mat_g) * identity * ridge_epsilon

    z = (1 + p) / (2 * torch.linalg.norm(mat_g))

    mat_root = identity * torch.pow(z, 1.0 / p)

    mat_m = mat_g * z

    alpha: float = -1.0 / p
    alpha_identity = (1.0 - alpha) * identity

    prev_error = torch.dist(mat_m, identity, p=torch.inf)

    mat_m_i = torch.empty_like(mat_m)
    new_mat_m = torch.empty_like(mat_m)
    new_mat_root = torch.empty_like(mat_root)

    for _ in range(max_iters):
        torch.add(alpha_identity, alpha * mat_m, out=mat_m_i)
        torch.matmul(mat_root, mat_m_i, out=new_mat_root)

        torch.matmul(torch.linalg.matrix_power(mat_m_i, p), mat_m, out=new_mat_m)

        error = torch.dist(new_mat_m, identity, p=torch.inf)

        # NOTE
        # This is the main bottleneck that slows Scalable Shampoo.
        # Because it is handled on the Python side so values need to be on the CPU
        # while XLA devices (e.g. TPU) don't seem to be affected.
        if torch.logical_or(error > prev_error * max_error_ratio, error <= error_tolerance):
            break

        mat_root.copy_(new_mat_root)
        mat_m, new_mat_m = new_mat_m, mat_m
        prev_error = error

    return mat_root


@torch.no_grad()
def compute_power_svd(matrix: torch.Tensor, power: float) -> torch.Tensor:
    """Compute a matrix inverse root with singular value decomposition.

    Args:
        matrix: Positive semidefinite matrix or batch of matrices.
        power: Root order. The matrix exponent is `-1 / power`.

    Returns:
        torch.Tensor: Matrix inverse root in the input data type.

    """
    u, s, vh = torch.linalg.svd(matrix.to(torch.float32), full_matrices=False)
    s.pow_(-1.0 / power)
    return (u @ (s.diag() if len(matrix.shape) == 2 else s.diag_embed()) @ vh).to(matrix.dtype)


def zero_power_via_newton_schulz_5(
    g: torch.Tensor,
    num_steps: int = 5,
    eps: float = 1e-7,
    safety_factor: float = 1.0,
    weights: NewtonSchulzWeights = (3.4445, -4.7750, 2.0315),
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Approximate a matrix's polar factor with quintic Newton-Schulz iterations.

    The coefficient schedule controls the singular value transformation. A finite
    number of iterations need not produce an exactly orthogonal matrix.

    Args:
        g: Matrix or batch of matrices with at least two dimensions.
        num_steps: Number of Newton-Schulz iterations.
        eps: Lower bound for the input normalization denominator.
        safety_factor: Multiplier for the input norm before iteration.
        weights: Preset name, coefficient tuple, or sequence of coefficient tuples. Reuse the final tuple if the
            sequence has fewer entries than `num_steps`.
        dtype: Data type for the iteration and output.

    Returns:
        torch.Tensor: Transformed matrix with the same shape as `g`.

    Raises:
        ValueError: The input has fewer than two dimensions or the coefficients are invalid.

    """
    if g.ndim < 2:
        raise ValueError(f'input must be over 2-dimensional. got {g.ndim}D.')

    is_dtensor: bool = has_dtensor and isinstance(g, DTensor)
    weight_schedule = get_newton_schulz_weights(weights)

    coeff_sequence = [weight_schedule[min(i, len(weight_schedule) - 1)] for i in range(num_steps)]

    transpose: bool = g.size(-2) > g.size(-1)
    x = g.mT if transpose else g
    x = (
        x.to(dtype=dtype, copy=True)
        if is_dtensor
        else x.to(dtype=dtype, copy=True, memory_format=torch.contiguous_format)
    )

    x.div_(x.norm(2, dim=(-2, -1), keepdim=True).mul_(safety_factor).clamp_min_(eps))

    if is_dtensor:  # pragma: no cover
        for w0, w1, w2 in coeff_sequence:
            a = x @ x.mT
            b = w1 * a + w2 * (a @ a)
            x = w0 * x + (b @ x)

        return x.mT if transpose else x

    mm_fn = torch.baddbmm if x.ndim > 2 else torch.addmm
    gram_fn = torch.bmm if x.ndim > 2 else torch.mm

    a = torch.empty((*x.shape[:-1], x.size(-2)), device=x.device, dtype=x.dtype)
    b = torch.empty_like(a)
    c = torch.empty_like(x)

    for w0, w1, w2 in coeff_sequence:
        gram_fn(x, x.mT, out=a)
        mm_fn(a, a, a, beta=w1, alpha=w2, out=b)
        mm_fn(x, b, x, beta=w0, alpha=1.0, out=c)
        x, c = c, x

    return x.mT if transpose else x
