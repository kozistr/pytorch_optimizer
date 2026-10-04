import itertools
from enum import IntEnum

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


class LayerWiseGrafting(IntEnum):
    """Layer wise update scale references for Shampoo.

    Grafting combines the Shampoo update direction with the magnitude of an SGD,
    AdaGrad, RMSProp, or sign based update.

    Reference: https://arxiv.org/abs/2002.11803

    """

    NONE = 0
    SGD = 1
    ADAGRAD = 2
    RMSPROP = 3
    SQRTN = 4


class Graft:
    """Identity graft that leaves Shampoo updates unchanged."""

    def __init__(self, *args):
        pass

    def add_statistics(self, grad: torch.Tensor, beta2: float) -> None:
        """Accept gradient statistics without accumulating them."""

    def precondition_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """Return the gradient unchanged."""
        return grad

    def update_momentum(self, update: torch.Tensor, beta1: float) -> torch.Tensor:
        """Return the update unchanged."""
        return update


class SGDGraft(Graft):
    """SGD momentum as a layer wise update scale reference."""

    def __init__(self, var: torch.Tensor):
        super().__init__(var)
        self.momentum: torch.Tensor = torch.zeros_like(var)

    def update_momentum(self, update: torch.Tensor, beta1: float) -> torch.Tensor:
        """Accumulate and return the momentum update."""
        self.momentum.mul_(beta1).add_(update)
        return self.momentum


class SQRTNGraft(Graft):
    """Sign based layer wise update scale reference."""

    def __init__(self, var: torch.Tensor):
        super().__init__(var)

    def precondition_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """Return the elementwise gradient sign."""
        return grad.sign()


class AdaGradGraft(SGDGraft):
    """Graft using AdaGrad with momentum.

    Args:
        var: Parameter tensor that determines the accumulator shape.
        diagonal_eps: Small epsilon added to diagonal for numerical stability.

    """

    def __init__(self, var: torch.Tensor, diagonal_eps: float):
        super().__init__(var)
        self.diagonal_eps = diagonal_eps
        self.statistics: torch.Tensor = torch.zeros_like(var)

    def add_statistics(self, grad: torch.Tensor, _) -> None:
        """Accumulate squared gradients for AdaGrad scaling."""
        self.statistics.add_(grad.pow(2))

    def precondition_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """Scale gradients by the inverse root of accumulated squared gradients."""
        return grad.div(self.statistics.sqrt().add_(self.diagonal_eps))


class RMSPropGraft(SGDGraft):
    """Graft using RMSProp with momentum.

    Args:
        var: Parameter tensor that determines the accumulator shape.
        diagonal_eps: Small epsilon added to diagonal for numerical stability.

    """

    def __init__(self, var: torch.Tensor, diagonal_eps: float):
        super().__init__(var)
        self.diagonal_eps = diagonal_eps
        self.statistics: torch.Tensor = torch.zeros_like(var)

    def add_statistics(self, grad: torch.Tensor, beta2: float) -> None:
        """Update the exponential moving average of squared gradients."""
        self.statistics.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    def precondition_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """Scale gradients by the inverse root of the squared gradient moving average."""
        return grad.div(self.statistics.sqrt().add_(self.diagonal_eps))


class BlockPartitioner:
    """Partition a tensor into smaller tensors for preconditioning.

    For example, if a variable has shape (4096, 512), splitting the 4096 dimension into 4 blocks,
    results in 4 smaller tensors each of shape (1024, 512).

    Args:
        var: Tensor variable.
        rank: Number of tensor dimensions, used to determine preconditioner shapes.
        block_size: Size of each block to partition.
        pre_conditioner_type: Type of preconditioner used.

    """

    def __init__(self, var: torch.Tensor, rank: int, block_size: int, pre_conditioner_type: int):
        self.shape: torch.Size = var.shape

        self.splits: list[tuple[int, torch.Tensor]] = []
        self.split_sizes: list[tuple[int, torch.Tensor]] = []

        split_sizes: list[torch.Tensor] = []

        # We split var into smaller blocks. Here we store the metadata to make that split.
        for i, d in enumerate(self.shape):
            if block_size <= 0 or block_size >= d:
                split_sizes.append(torch.tensor([d], dtype=torch.int32))
                continue

            # d - 1, otherwise split appends a 0-size array.
            num_split: int = (d - 1) // block_size
            indices = (torch.arange(num_split, dtype=torch.int32) + 1) * block_size

            sizes: torch.Tensor = torch.full((num_split + 1,), block_size, dtype=torch.int32)
            sizes[-1] = d - indices[-1]

            self.splits.append((i, indices))
            self.split_sizes.append((i, sizes))
            split_sizes.append(sizes)

        self.num_splits: int = len(split_sizes)
        self.pre_conditioner_shapes: list[list[torch.Tensor] | None] = self.build_pre_conditioner_shapes(
            split_sizes,
            pre_conditioner_type,
            rank,
        )

    @staticmethod
    def build_pre_conditioner_shapes(
        split_sizes: list[torch.Tensor],
        pre_conditioner_type: int,
        rank: int,
    ) -> list[list[torch.Tensor] | None]:
        """Build matrix shapes for each block preconditioner."""
        pre_conditioner_shapes: list[list[torch.Tensor] | None] = []
        for t in itertools.product(*split_sizes):
            t_shape: list[list[torch.Tensor] | None] = [[d, d] for d in t]
            if pre_conditioner_type == PreConditionerType.INPUT:
                t_shape[-1] = None
            elif pre_conditioner_type == PreConditionerType.OUTPUT:
                t_shape = [None] * (rank - 1) + t_shape[-1:]
            pre_conditioner_shapes.extend(t_shape)
        return pre_conditioner_shapes

    def shapes_for_pre_conditioners(self) -> list[list[torch.Tensor] | None]:
        """Return the matrix shapes of the block preconditioners."""
        return self.pre_conditioner_shapes

    @torch.no_grad()
    def partition(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Partition tensor into blocks."""
        if x.shape != self.shape:
            raise ValueError(f'self.shape != x.shape ({self.shape} vs {x.shape})')

        tensors = [x]
        for i, sizes in self.split_sizes:
            tensors = [torch.split(t, list(sizes), dim=i) for t in tensors]
            tensors = [t for tensor in tensors for t in tensor]
        return tensors

    def merge_partitions(self, partitions: list[torch.Tensor]) -> torch.Tensor:
        """Merge partitions back to original shape."""
        merged_partitions = partitions
        for i, indices in reversed(self.splits):
            n: int = len(indices) + 1

            # fmt: off
            merged_partitions: list[torch.Tensor] = [
                torch.cat(merged_partitions[idx:idx + n], dim=i) for idx in range(0, len(merged_partitions), n)
            ]
            # fmt: on

        return merged_partitions[0]


class PreConditionerType(IntEnum):
    """Dimensions to precondition with Shampoo.

    `ALL` preconditions every dimension. `INPUT` and `OUTPUT` use one sided
    preconditioning, treating the last dimension as output and the others as input.

    """

    ALL = 0
    INPUT = 1
    OUTPUT = 2


class PreConditioner:
    """Shampoo gradient statistics and matrix preconditioners.

    Args:
        var: Tensor variable corresponding to model parameters.
        beta2: Decay rate for second moment estimates.
        inverse_exponent_override: Override for inverse exponent used in preconditioning.
        block_size: Size of blocks for partitioning large tensors.
        skip_preconditioning_rank_lt: Skip preconditioning for tensors with rank less than this.
        no_preconditioning_for_layers_with_dim_gt: Skip preconditioning for layers with dimension size greater than
            this.
        shape_interpretation: Whether to apply automatic shape interpretation for tensor dimensions.
        pre_conditioner_type: Type of preconditioner to use.
        matrix_eps: Epsilon term added for numerical stability in matrix operations.
        use_svd: Use SVD method instead of Schur-Newton method for matrix inverse powers calculation.

    """

    def __init__(
        self,
        var: torch.Tensor,
        beta2: float,
        inverse_exponent_override: int,
        block_size: int,
        skip_preconditioning_rank_lt: int,
        no_preconditioning_for_layers_with_dim_gt: int,
        shape_interpretation: bool,
        pre_conditioner_type: int,
        matrix_eps: float = 1e-6,
        use_svd: bool = False,
    ):
        self.beta2 = beta2
        self.inverse_exponent_override = inverse_exponent_override
        self.skip_preconditioning_rank_lt = skip_preconditioning_rank_lt
        self.no_preconditioning_for_layers_with_dim_gt = no_preconditioning_for_layers_with_dim_gt
        self.pre_conditioner_type = pre_conditioner_type
        self.matrix_eps = matrix_eps
        self.use_svd = use_svd

        self.w2: float = 1.0 if self.beta2 == 1.0 else (1.0 - self.beta2)

        self.original_shape: torch.Size = var.shape
        self.transformed_shape: list[int] | torch.Size = (
            merge_small_dims(self.original_shape, block_size) if shape_interpretation else var.shape
        )

        self.should_precondition_dims: list[bool] = self.get_should_precondition_dims()
        self.rank: int = sum(self.should_precondition_dims)
        self.exponent_for_pre_conditioner: int = (
            self.inverse_exponent_override if self.inverse_exponent_override > 0 else 2 * self.rank
        )

        self.statistics: list[torch.Tensor] | torch.Tensor = []
        self.pre_conditioners: list[torch.Tensor] | torch.Tensor = []

        self.is_same_shapes: bool = False
        if len(self.transformed_shape) > 1 and not self.skip_precondition(var):
            self.partitioner = BlockPartitioner(
                var=torch.reshape(var, self.transformed_shape),
                rank=self.rank,
                block_size=block_size,
                pre_conditioner_type=self.pre_conditioner_type,
            )

            shapes = self.partitioner.shapes_for_pre_conditioners()
            self.statistics = [self.matrix_eps * torch.eye(shape[0], device=var.device) for shape in shapes if shape]
            self.pre_conditioners = [torch.eye(shape[0], device=var.device) for shape in shapes if shape]

            filtered_shape: list[tuple] = [tuple(shape) for shape in shapes if shape is not None]
            self.is_same_shapes = bool(filtered_shape) and len(set(filtered_shape)) == 1

        if self.is_same_shapes:
            self.statistics = torch.stack(self.statistics, dim=0)
            self.pre_conditioners = torch.stack(self.pre_conditioners, dim=0)

    def get_should_precondition_dims(self) -> list[bool]:
        """Select dimensions to precondition from the preconditioner type."""
        if self.pre_conditioner_type == PreConditionerType.ALL or len(self.transformed_shape) <= 1:
            return [True] * len(self.transformed_shape)
        if self.pre_conditioner_type == PreConditionerType.INPUT:
            return [True] * (len(self.transformed_shape) - 1) + [False]
        if self.pre_conditioner_type == PreConditionerType.OUTPUT:
            return [False] * (len(self.transformed_shape) - 1) + [True]
        raise ValueError

    def skip_precondition(self, x: torch.Tensor) -> bool:
        return (len(x.shape) < self.skip_preconditioning_rank_lt) or any(
            dim > self.no_preconditioning_for_layers_with_dim_gt for dim in x.shape
        )

    def add_statistics(self, grad: torch.Tensor) -> None:
        """Compute statistics from gradients and add to state entries.

        Args:
            grad: Gradient tensor from which to compute statistics.

        """
        if len(self.statistics) == 0:
            return

        reshaped_grad: torch.Tensor = torch.reshape(grad, self.transformed_shape)
        partitioned_grads: list[torch.Tensor] = self.partitioner.partition(reshaped_grad)

        for j, partitioned_grad in enumerate(partitioned_grads):
            for i, axis in enumerate(ax for ax, selected in enumerate(self.should_precondition_dims) if selected):
                axes: list[int] = [ax for ax in range(partitioned_grad.ndim) if ax != axis]
                stat: torch.Tensor = torch.tensordot(partitioned_grad, partitioned_grad, dims=[axes, axes])
                self.statistics[j * self.rank + i].mul_(self.beta2).add_(stat, alpha=self.w2)

    def compute_pre_conditioners(self) -> None:
        """Compute inverse roots of the accumulated statistics matrices.

        Use batched SVD for compatible shapes when `use_svd=True`, otherwise compute
        each inverse root with SVD or coupled Schur-Newton iteration.
        """
        if self.use_svd and self.is_same_shapes:
            self.pre_conditioners = compute_power_svd(matrix=self.statistics, power=self.exponent_for_pre_conditioner)
            return

        for i, statistic in enumerate(self.statistics):
            self.pre_conditioners[i] = (
                compute_power_svd(matrix=statistic, power=self.exponent_for_pre_conditioner)
                if self.use_svd
                else compute_power_schur_newton(
                    mat_g=statistic, p=self.exponent_for_pre_conditioner, ridge_epsilon=self.matrix_eps
                )
            )

    @staticmethod
    def precondition_block(
        partitioned_grad: torch.Tensor,
        should_preconditioned_dims: list[bool],
        pre_conditioners_for_grad: list[torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        """Perform a preconditioning operation on a single gradient block.

        Loop invariant: the dimension to be preconditioned is first
        We keep all axes in the same cyclic order they were originally.
        """
        rank: int = len(partitioned_grad.shape)
        roll: tuple[int, ...] = (*range(1, rank), 0)

        i: int = 0
        for should_precondition_dim in should_preconditioned_dims:
            if not should_precondition_dim:
                partitioned_grad = torch.permute(partitioned_grad, roll)
                continue

            partitioned_grad = torch.tensordot(partitioned_grad, pre_conditioners_for_grad[i], dims=[[0], [0]])
            i += 1

        return partitioned_grad

    def preconditioned_grad(self, grad: torch.Tensor) -> torch.Tensor:
        """Precondition the gradient.

        Args:
            grad: Gradient tensor to precondition.

        """
        if len(self.pre_conditioners) == 0:
            return grad

        reshaped_grad = torch.reshape(grad, self.transformed_shape)
        partitioned_grads = self.partitioner.partition(reshaped_grad)

        # fmt: off
        pre_cond_partitioned_grads: list[torch.Tensor] = [
            self.precondition_block(
                partitioned_grad,
                self.should_precondition_dims,
                self.pre_conditioners[i * self.rank:(i + 1) * self.rank],
            )
            for i, partitioned_grad in enumerate(partitioned_grads)
        ]
        # fmt: on

        merged_grad = self.partitioner.merge_partitions(pre_cond_partitioned_grads)

        return merged_grad.reshape(self.original_shape)


def build_graft(p: torch.Tensor, graft_type: int, diagonal_eps: float = 1e-10):
    """Construct a Shampoo graft from a `LayerWiseGrafting` value."""
    if graft_type == LayerWiseGrafting.ADAGRAD:
        return AdaGradGraft(p, diagonal_eps)
    if graft_type == LayerWiseGrafting.RMSPROP:
        return RMSPropGraft(p, diagonal_eps)
    if graft_type == LayerWiseGrafting.SGD:
        return SGDGraft(p)
    if graft_type == LayerWiseGrafting.SQRTN:
        return SQRTNGraft(p)
    return Graft(p)


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


def merge_small_dims(shape_to_merge: list[int] | torch.Size, max_dim: int) -> list[int]:
    """Merge small dimensions in a tensor shape.

    If a tensor shape has small dimensions, merge them into larger combined dimensions without exceeding max_dim.

    Examples:
        [1, 2, 512, 1, 2048, 1, 3, 4] with max_dim=1024 becomes [1024, 2048, 12],
        and [1, 2, 768, 1, 2048] becomes [2, 768, 2048].

    Args:
        shape_to_merge: The original shape to merge.
        max_dim: Maximum allowed dimension for merging.

    """
    merged_shape: list[int] = []

    product: int = 1
    for dim in shape_to_merge:
        product *= dim
        if product > max_dim:
            merged_shape.append(product // dim)
            product = dim

    merged_shape.append(product)

    return merged_shape if len(merged_shape) > 1 else [1]


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

    x = g.to(dtype=dtype, copy=True)

    transpose: bool = x.size(-2) > x.size(-1)
    if transpose:
        x = x.mT

    x.div_(x.norm(2, dim=(-2, -1), keepdim=True).mul_(safety_factor).clamp_min_(eps))

    if is_dtensor:  # pragma: no cover
        for w0, w1, w2 in coeff_sequence:
            a = x @ x.mT
            b = w1 * a + w2 * (a @ a)
            x = w0 * x + (b @ x)

        return x.mT if transpose else x

    mm_fn = torch.baddbmm if x.ndim > 2 else torch.addmm

    x = x.contiguous()
    a = torch.empty((*x.shape[:-1], x.size(-2)), device=x.device, dtype=x.dtype)
    b = torch.empty_like(a)
    c = torch.empty_like(x)

    for w0, w1, w2 in coeff_sequence:
        mm_fn(a, x, x.mT, beta=0.0, alpha=1.0, out=a)
        mm_fn(a, a, a, beta=w1, alpha=w2, out=b)
        mm_fn(x, b, x, beta=w0, alpha=1.0, out=c)
        x, c = c, x

    return x.mT if transpose else x
