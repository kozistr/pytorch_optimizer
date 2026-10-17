from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.optim import Optimizer

from pytorch_optimizer.optimizer.muon.adago import AdaGO
from pytorch_optimizer.optimizer.muon.adamuon import AdaMuon
from pytorch_optimizer.optimizer.muon.muon import Muon
from pytorch_optimizer.optimizer.muon.muon_clip import MuonClip
from pytorch_optimizer.optimizer.muon.normuon import NorMuon

if TYPE_CHECKING:
    from pytorch_optimizer.base.type import ParamsT


def prepare_muon_parameters(
    model: nn.Module,
    optimizer_name: str,
    lr: float | torch.Tensor,
    weight_decay: float,
    adamw_lr: float = 3e-4,
    adamw_wd: float = 0.0,
    **kwargs,
) -> Optimizer:
    """Create a Muon family optimizer by grouping model parameters.

    Classifies weights by parameter name and dimensionality. Review the resulting groups,
    or construct them yourself using the optimizer's example for model specific control.

    Args:
        model: Model whose parameters to group.
        optimizer_name: Muon family optimizer name.
        lr: Learning rate for the orthogonal update groups.
        weight_decay: Weight decay for the orthogonal update groups.
        adamw_lr: Learning rate for the AdamW groups.
        adamw_wd: Weight decay for the AdamW groups.
        **kwargs (dict): Options for the selected optimizer.

    Returns:
        Optimizer: Optimizer with orthogonal update and AdamW parameter groups.

    """
    muon_parameters: dict[int, torch.Tensor] = {}
    non_muon_params: dict[int, torch.Tensor] = {}

    for module_name, module in model.named_modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue

            full_name = f'{module_name}.{name}' if module_name else name
            if (
                isinstance(module, (nn.Linear, nn.Conv1d, nn.LSTM, nn.Conv2d))
                and param.ndim >= 2
                and 'head' not in full_name
            ):
                muon_parameters[id(param)] = param
            else:
                non_muon_params[id(param)] = param

    param_groups: ParamsT = [
        {
            'params': [p for key, p in muon_parameters.items() if key not in non_muon_params],
            'lr': lr,
            'weight_decay': weight_decay,
            'use_muon': True,
        },
        {'params': list(non_muon_params.values()), 'lr': adamw_lr, 'weight_decay': adamw_wd, 'use_muon': False},
    ]

    optimizer_name = optimizer_name.lower()

    if optimizer_name == 'muonclip':
        return MuonClip(param_groups, **kwargs)
    if optimizer_name == 'adamuon':
        return AdaMuon(param_groups, **kwargs)
    if optimizer_name == 'adago':
        return AdaGO(param_groups, **kwargs)
    if optimizer_name == 'normuon':
        return NorMuon(param_groups, **kwargs)

    return Muon(param_groups, **kwargs)
