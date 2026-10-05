import torch
from torch import nn


def has_overflow(grad_norm: torch.Tensor) -> bool:
    """Return whether a tensor contains NaN or infinite values."""
    return bool(torch.logical_or(torch.isnan(grad_norm), torch.isinf(grad_norm)).any())


def to_real(x: torch.Tensor) -> torch.Tensor:
    """Return the real part of a complex tensor, or the original real tensor."""
    return x.real if torch.is_complex(x) else x


@torch.no_grad()
def copy_stochastic(target: torch.Tensor, source: torch.Tensor) -> None:
    """Copy float32 values to bfloat16 with stochastic rounding.

    reference: https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905

    Args:
        target: A tensor in bfloat16 format to copy to.
        source: A tensor in float32 format to copy from.

    """
    result = torch.randint_like(
        source,
        dtype=torch.int32,
        low=0,
        high=1 << 16,
    )

    result.add_(source.view(dtype=torch.int32))

    result.bitwise_and_(-65536)

    target.copy_(result.view(dtype=torch.float32))


class StochasticAccumulator:
    """Accumulate bfloat16 gradients with stochastic rounding.

    Attach hooks once, then restore the accumulated gradient buffers before each optimizer step.

    Examples:
        ```python
        hooks = StochasticAccumulator.assign_hooks(model)
        optimizer.zero_grad()
        for inputs, targets in microbatches:
            loss = loss_fn(model(inputs), targets) / len(microbatches)
            loss.backward()
        StochasticAccumulator.reassign_grad_buffer(model)
        optimizer.step()
        optimizer.zero_grad()
        ```

    """

    @staticmethod
    def stochastic_grad_accum(p: torch.Tensor) -> None:
        if hasattr(p, 'acc_grad'):
            acc_grad_fp32 = p.acc_grad.clone().to(torch.float32)
            acc_grad_fp32.add_(p.grad.to(torch.float32))

            copy_stochastic(p.acc_grad, acc_grad_fp32)

            del acc_grad_fp32
        else:
            p.acc_grad = p.grad.clone().to(torch.bfloat16)  # ty: ignore[invalid-assignment]

        del p.grad

    @staticmethod
    def reassign_grad_buffer(model: nn.Module) -> None:
        for _, p in model.named_parameters():
            if p.requires_grad and hasattr(p, 'acc_grad'):
                p.grad = p.acc_grad  # ty: ignore[invalid-assignment]
                del p.acc_grad

    @staticmethod
    def assign_hooks(model: nn.Module) -> list:
        return [
            p.register_post_accumulate_grad_hook(StochasticAccumulator.stochastic_grad_accum)
            for _, p in model.named_parameters()
            if p.requires_grad
        ]
