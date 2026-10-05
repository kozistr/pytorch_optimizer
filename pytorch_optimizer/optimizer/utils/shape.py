import torch


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
