from typing import Dict

import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.types import (
    CLOSURE,
    DEFAULT_PARAMETERS,
    PARAM_GROUPS,
    PARAMS,
)


class SAM(Optimizer):
    def __init__(
        self,
        params: PARAMS,
        base_optimizer,
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs,
    ):
        self.rho = rho

        self.check_valid_parameters()

        defaults: DEFAULT_PARAMETERS = dict(
            rho=rho, adaptive=adaptive, **kwargs
        )
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups: PARAM_GROUPS = self.base_optimizer.param_groups

    def check_valid_parameters(self):
        if 0.0 > self.rho:
            raise ValueError(f'Invalid rho : {self.rho}')

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self.grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)

            for p in group['params']:
                if p.grad is None:
                    continue
                self.state[p]['old_p'] = p.data.clone()
                e_w = (
                    (torch.pow(p, 2) if group['adaptive'] else 1.0)
                    * p.grad
                    * scale.to(p)
                )
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data = self.state[p][
                    'old_p'
                ]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure: CLOSURE = None):
        if closure is None:
            raise RuntimeError(
                'Sharpness Aware Minimization requires closure, but it was not provided'
            )

        # the closure should do a full forward-backward pass
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]['params'][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group['adaptive'] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group['params']
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict: Dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
