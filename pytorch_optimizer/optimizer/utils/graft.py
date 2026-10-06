from enum import IntEnum

import torch


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
