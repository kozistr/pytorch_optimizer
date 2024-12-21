import os
from typing import List, Optional

import torch
from torch.distributed import ReduceOp, all_reduce

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS
from pytorch_optimizer.optimizer.shampoo_utils import zero_power_via_newton_schulz_5


class Muon(BaseOptimizer):
    r"""MomentUm Orthogonalized by Newton-schulz.

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-processing step, in which
    each 2D parameter's update is replaced with the nearest orthogonal matrix. To efficiently orthogonalize each
    update, we use a Newton-Schulz iteration, which has the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for fine-tuning pretrained models, but we haven't tested this.

    :param params: PARAMETERS. the parameters to be optimized by Muon.
    :param lr: float. learning rate.
    :param momentum: float. the momentum used by the internal SGD.
    :param betas: The betas for the internal AdamW.
    :param nesterov: bool. whether to use nesterov momentum.
    :param ns_steps: int. the number of Newton-Schulz iterations to run. (6 is probably always enough)
    :param adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are {0, 1}-D or
        are detected as being the embed or lm_head will be optimized by AdamW as well.
    :param adamw_lr: The learning rate for the internal AdamW.
    :param adamw_wd: The weight decay for the internal AdamW.
    :param adamw_eps: The epsilon for the internal AdamW.
    """

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 2e-2,
        momentum: float = 0.95,
        betas: BETAS = (0.95, 0.95),
        nesterov: bool = True,
        ns_steps: int = 6,
        adamw_params: Optional[PARAMETERS] = None,
        adamw_lr: float = 3e-4,
        adamw_wd: float = 0,
        adamw_eps: float = 1e-8,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_learning_rate(adamw_lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0, range_type='[)')
        self.validate_positive(ns_steps, 'ns_steps')
        self.validate_betas(betas)
        self.validate_non_negative(adamw_wd, 'adamw_wd')
        self.validate_non_negative(adamw_eps, 'adamw_eps')

        params = self.get_parameters(params)
        adamw_params = self.get_parameters(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)

        self.world_size: int = int(os.environ.get('WORLD_SIZE', 1))
        self.rank: int = int(os.environ.get('RANK', 0))

        defaults: DEFAULTS = {
            'lr': lr,
            'momentum': momentum,
            'nesterov': nesterov,
            'ns_steps': ns_steps,
            'adamw_lr': adamw_lr,
            'adamw_lr_ratio': adamw_lr / lr,
            'adamw_betas': betas,
            'adamw_wd': adamw_wd,
            'adamw_eps': adamw_eps,
        }
        super().__init__(params, defaults)

        self.set_muon_state(params, adamw_params)

    def __str__(self) -> str:
        return 'Muon'

    @staticmethod
    def get_parameters(params: PARAMETERS) -> List[torch.Tensor]:
        if isinstance(params, list) and isinstance(params[0], torch.Tensor):
            return params

        new_params = []
        for group in params:
            if isinstance(group, dict) and 'params' in group:
                new_params.extend(list(group['params']))
            else:
                new_params.append(group)

        return new_params

    def set_muon_state(self, params: PARAMETERS, adamw_params: PARAMETERS, threshold: int = 8192) -> None:
        r"""Set use_muon flag."""
        for p in params:
            self.state[p]['use_muon'] = p.ndim >= 2 and p.size(0) < threshold

        for p in adamw_params:
            self.state[p]['use_muon'] = False

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                state = self.state[p]

                state['momentum_buffer'] = torch.zeros_like(p)
                state['moment1'] = torch.zeros_like(p)
                state['moment2'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            params = []
            for p in group['params']:
                if p.grad is not None and self.state[p]['use_muon']:
                    if p.grad.is_sparse:
                        raise NoSparseGradientError(str(self))
                    params.append(p)

            if len(params) == 0:
                continue

            lr = group['lr']
            momentum = group['momentum']

            total_params: int = sum(p.numel() for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr_idx: int = 0

            for i, p in enumerate(params):
                if i % self.world_size != self.rank:
                    curr_idx += p.numel()
                    continue

                g = p.grad
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)

                if group['nesterov']:
                    g.add_(buf, alpha=momentum)
                else:
                    g = buf

                g = zero_power_via_newton_schulz_5(g, num_steps=group['ns_steps'])
                g.mul_(max(1.0, g.size(0) / g.size(1)) ** 0.5)

                updates_flat[curr_idx:curr_idx + p.numel()] = g.flatten()  # fmt: skip

            if self.world_size > 1:  # pragma: no cover
                all_reduce(updates_flat, op=ReduceOp.SUM)

            curr_idx: int = 0
            for p in params:
                g = updates_flat[curr_idx:curr_idx + p.numel()].view_as(p).type_as(p)  # fmt: skip
                p.add_(g, alpha=-lr)
                curr_idx += p.numel()

            params = [p for p in group['params'] if p.grad is not None and not self.state[p]['use_muon']]

            lr: float = group['adamw_lr_ratio'] * group['lr']
            beta1, beta2 = group['adamw_betas']

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2: float = self.debias(beta2, group['step'])
            scale: float = bias_correction1 / bias_correction2 ** 0.5  # fmt: skip
            step_size: float = lr / scale

            for p in params:
                grad = p.grad
                state = self.state[p]
                if 'moment1' not in state:
                    state['moment1'] = torch.zeros_like(grad)
                    state['moment2'] = torch.zeros_like(grad)

                buf1, buf2 = state['moment1'], state['moment2']
                buf1.lerp_(grad, weight=1.0 - beta1)
                buf2.lerp_(grad.square(), weight=1.0 - beta2)

                update = buf1 / buf2.sqrt().add_(group['adamw_eps'])

                self.apply_weight_decay(
                    p,
                    grad,
                    lr=lr,
                    weight_decay=group['adamw_wd'],
                    weight_decouple=True,
                    fixed_decay=False,
                )

                p.add_(update, alpha=-step_size)

        return loss
