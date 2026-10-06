import torch

from pytorch_optimizer.optimizer.utils.matrix import compute_power_schur_newton, compute_power_svd
from pytorch_optimizer.optimizer.utils.partition import BlockPartitioner, PreConditionerType
from pytorch_optimizer.optimizer.utils.shape import merge_small_dims


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
