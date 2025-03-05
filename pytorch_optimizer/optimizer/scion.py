import math
from typing import Literal

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.optimizer.shampoo_utils import zero_power_via_newton_schulz_5

LMO_TYPE = Literal['spectral', 'sign', 'col_norm', 'row_norm']


class SCION(BaseOptimizer):
    r"""Training Deep Learning Models with Norm-Constrained LMOs.

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups.
    :param lr: float. learning rate.
    :param momentum: float. momentum factor.
    :param constraint: bool. whether to use a constraint SCG or not.
    :param lmo_type: LMO_TYPE. supported LMO types.
    :param scale: float. based on the usage of the original intend, 50.0 is used for Transformer block, and 3000.0 is
        used for others (e.g. Embedding, LM head)
    :param weight_decay: float. weight decay (L2 penalty).
    :param weight_decouple: bool. the optimizer uses decoupled weight decay as in AdamW.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-3,
        momentum: float = 0.1,
        constraint: bool = False,
        lmo_type: LMO_TYPE = 'spectral',
        scale: float = 1.0,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0, '(]')
        self.validate_positive(scale, 'scale')
        self.validate_options(lmo_type, 'lmo_type', ['spectral', 'sign', 'col_norm', 'row_norm'])

        defaults: DEFAULTS = {
            'lr': lr,
            'momentum': momentum,
            'constraint': constraint,
            'lmo_type': lmo_type,
            'scale': scale,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
        }
        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SCION'

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['d'] = torch.zeros_like(p)

    @staticmethod
    def get_lmo_direction(grad: torch.Tensor, lmo_type: LMO_TYPE) -> torch.Tensor:
        r"""Get LMO direction.

        fallback to `sign`
        """
        d_out, d_in, *_ = grad.shape if grad.ndim > 1 else (grad.size(0), grad.size(0))

        if lmo_type == 'spectral':
            return (
                zero_power_via_newton_schulz_5(grad.reshape(len(grad), -1))
                .view(grad.shape)
                .mul_(max(1.0, math.sqrt(d_out / d_in)))
            )
        if lmo_type == 'sign':
            return torch.sign(grad).div_(d_in)
        if lmo_type == 'col_norm':
            return grad / torch.norm(grad, dim=0, keepdim=True).add_(1e-6)
        if lmo_type == 'row_norm' and grad.ndim == 2:
            return grad / torch.norm(grad, dim=1, keepdim=True).add_(1e-6)
        return torch.sign(grad).div_(d_in)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            step_size: float = -group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                if 'd' not in state:
                    state['d'] = torch.zeros_like(grad)

                d = state['d']
                d.mul_(1.0 - group['momentum']).add_(grad, alpha=group['momentum'])

                update = self.get_lmo_direction(d, group['lmo_type'])
                update.mul_(group['scale'])

                if not group['constraint']:
                    self.apply_weight_decay(
                        p,
                        grad,
                        lr=group['lr'],
                        weight_decay=group['weight_decay'],
                        weight_decouple=group['weight_decouple'],
                        fixed_decay=False,
                    )

                    p.add_(update, alpha=step_size)
                else:
                    p.mul_(1.0 - step_size).add_(update, alpha=step_size)

        return loss
