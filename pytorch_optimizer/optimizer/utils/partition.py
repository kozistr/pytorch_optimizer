import itertools
from enum import IntEnum

import torch


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
