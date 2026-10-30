import itertools
from enum import IntEnum

import torch

from pytorch_optimizer.optimizer.utils.matrix import compute_power_schur_newton, compute_power_svd
from pytorch_optimizer.optimizer.utils.shape import merge_small_dims


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
    """Identity graft with momentum for Shampoo warm-up updates."""

    def __init__(self, var: torch.Tensor):
        self.momentum: torch.Tensor = torch.zeros_like(var)

    def add_statistics(self, grad: torch.Tensor, beta2: float) -> None:
        """Accept gradient statistics without accumulating them."""

    def precondition_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """Return the gradient unchanged."""
        return grad

    def update_momentum(self, update: torch.Tensor, beta1: float, weight: float = 1.0) -> torch.Tensor:
        """Accumulate and return the momentum update."""
        self.momentum.mul_(beta1).add_(update, alpha=weight)
        return self.momentum


class SGDGraft(Graft):
    """SGD momentum as a layer wise update scale reference."""


class SQRTNGraft(Graft):
    """Sign based layer wise update scale reference."""

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
        self.statistics.addcmul_(grad, grad)

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


class PreConditionerType(IntEnum):
    """Dimensions to precondition with Shampoo.

    `ALL` preconditions every dimension. `INPUT` and `OUTPUT` use one sided
    preconditioning, treating the last dimension as output and the others as input.

    """

    ALL = 0
    INPUT = 1
    OUTPUT = 2


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

        self.splits: list[tuple[int, list[int]]] = []
        self.split_sizes: list[tuple[int, list[int]]] = []

        split_sizes: list[list[int]] = []

        # We split var into smaller blocks. Here we store the metadata to make that split.
        for i, d in enumerate(self.shape):
            if block_size <= 0 or block_size >= d:
                split_sizes.append([d])
                continue

            # d - 1, otherwise split appends a 0-size array.
            num_split: int = (d - 1) // block_size
            indices = list(range(block_size, d, block_size))
            sizes = [block_size] * num_split + [d - num_split * block_size]

            self.splits.append((i, indices))
            self.split_sizes.append((i, sizes))
            split_sizes.append(sizes)

        self.num_splits: int = len(split_sizes)
        self.pre_conditioner_shapes: list[list[int] | None] = self.build_pre_conditioner_shapes(
            split_sizes,
            pre_conditioner_type,
            rank,
        )

    @staticmethod
    def build_pre_conditioner_shapes(
        split_sizes: list[list[int]],
        pre_conditioner_type: int,
        rank: int,
    ) -> list[list[int] | None]:
        """Build matrix shapes for each block preconditioner."""
        pre_conditioner_shapes: list[list[int] | None] = []

        for t in itertools.product(*split_sizes):
            t_shape: list[list[int] | None] = [[d, d] for d in t]

            if pre_conditioner_type == PreConditionerType.INPUT:
                t_shape[-1] = None
            elif pre_conditioner_type == PreConditionerType.OUTPUT:
                t_shape = [None] * (rank - 1) + t_shape[-1:]
            pre_conditioner_shapes.extend(t_shape)
        return pre_conditioner_shapes

    def shapes_for_pre_conditioners(self) -> list[list[int] | None]:
        """Return the matrix shapes of the block preconditioners."""
        return self.pre_conditioner_shapes

    @torch.no_grad()
    def partition(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Partition tensor into blocks."""
        if x.shape != self.shape:
            raise ValueError(f'self.shape != x.shape ({self.shape} vs {x.shape})')

        tensors = [x]
        for i, sizes in self.split_sizes:
            tensors = [torch.split(t, sizes, dim=i) for t in tensors]
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
            dtype = torch.float64 if var.dtype == torch.float64 else torch.float32
            self.statistics = [
                self.matrix_eps * torch.eye(shape[0], device=var.device, dtype=dtype) for shape in shapes if shape
            ]
            self.pre_conditioners = [torch.eye(shape[0], device=var.device, dtype=dtype) for shape in shapes if shape]

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

        reshaped_grad = grad.to(dtype=self.statistics[0].dtype).reshape(self.transformed_shape)
        partitioned_grads: list[torch.Tensor] = self.partitioner.partition(reshaped_grad)

        if self.is_same_shapes:
            unfolded = torch.stack([
                block.movedim(axis, 0).reshape(block.size(axis), -1)
                for block in partitioned_grads
                for axis, selected in enumerate(self.should_precondition_dims)
                if selected
            ])
            self.statistics.baddbmm_(unfolded, unfolded.mT, beta=self.beta2, alpha=self.w2)
            return

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

        reshaped_grad = grad.to(dtype=self.pre_conditioners[0].dtype).reshape(self.transformed_shape)
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

        return merged_grad.reshape(self.original_shape).to(dtype=grad.dtype)
