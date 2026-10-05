import math

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from pytorch_optimizer.base.type import ParamsT


def is_valid_parameters(parameters: ParamsT) -> bool:
    """Check for a nonempty list or tuple whose first entry is a parameter group dictionary."""
    return isinstance(parameters, (list, tuple)) and len(parameters) > 0 and isinstance(parameters[0], dict)


def disable_running_stats(model: nn.Module):
    """Pause BatchNorm running statistic updates by setting momentum to zero."""

    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model: nn.Module):
    """Restore BatchNorm momentum after pausing running statistic updates."""

    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, 'backup_momentum'):
            module.momentum = module.backup_momentum

    model.apply(_enable)


def reg_noise(
    network1: nn.Module,
    network2: nn.Module,
    num_data: int,
    lr: float,
    eta: float = 8e-3,
    temperature: float = 1e-4,
) -> torch.Tensor:
    """Compute the Entropy-MCMC coupling and noise term for two networks.

    Usage example and detailed implementation can be found at:
    https://github.com/lblaoke/EMCMC/blob/master/exp/cifar10_emcmc.py

    Args:
        network1: First neural network.
        network2: Second neural network.
        num_data: Number of training data points.
        lr: Learning rate.
        eta: Eta parameter controlling auxiliary guiding variable.
        temperature: Temperature parameter for sampling.

    """
    reg_coef: float = 0.5 / (eta * num_data)
    noise_coef: float = math.sqrt(2.0 / lr / num_data * temperature)

    loss = torch.tensor(0.0, device=next(network1.parameters()).device)

    for param1, param2 in zip(network1.parameters(), network2.parameters()):
        reg = (param1 - param2).pow_(2).mul_(reg_coef).sum()

        noise = param1 * torch.randn_like(param1)
        noise.add_(param2 * torch.randn_like(param2))

        loss.add_(reg - noise.mul_(noise_coef).sum())

    return loss
