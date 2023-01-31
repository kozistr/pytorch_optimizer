from typing import Dict

import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base.exception import NoClosureError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.types import CLOSURE, DEFAULTS, OPTIMIZER, PARAMETERS


class SAM(Optimizer, BaseOptimizer):
    r"""Sharpness-Aware Minimization for Efficiently Improving Generalization.

    Example:
    -------
        Here's an example::

            model = YourModel()
            base_optimizer = Ranger21
            optimizer = SAM(model.parameters(), base_optimizer)

            for input, output in data:
                # first forward-backward pass

                loss = loss_function(output, model(input))
                loss.backward()
                optimizer.first_step(zero_grad=True)

                # second forward-backward pass
                # make sure to do a full forward pass
                loss_function(output, model(input)).backward()
                optimizer.second_step(zero_grad=True)

        Alternative example with a single closure-based step function::

            model = YourModel()
            base_optimizer = Ranger21
            optimizer = SAM(model.parameters(), base_optimizer)

            def closure():
                loss = loss_function(output, model(input))
                loss.backward()
                return loss

            for input, output in data:
                loss = loss_function(output, model(input))
                loss.backward()
                optimizer.step(closure)
                optimizer.zero_grad()

    :param params: PARAMETERS. iterable of parameters to optimize or dicts defining parameter groups
    :param base_optimizer: Optimizer. base optimizer
    :param rho: float. size of the neighborhood for computing the max loss
    :param adaptive: bool. element-wise Adaptive SAM
    :param kwargs: Dict. parameters for optimizer.
    """

    def __init__(
        self,
        params: PARAMETERS,
        base_optimizer: OPTIMIZER,
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs,
    ):
        self.rho = rho

        self.validate_parameters()

        defaults: DEFAULTS = {'rho': rho, 'adaptive': adaptive}
        defaults.update(kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    def validate_parameters(self):
        self.validate_rho(self.rho)

    @property
    def __str__(self) -> str:
        return 'SAM'

    @torch.no_grad()
    def reset(self):
        pass

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self.grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)

            for p in group['params']:
                if p.grad is None:
                    continue

                self.state[p]['old_p'] = p.clone()
                e_w = (torch.pow(p, 2) if group['adaptive'] else 1.0) * p.grad * scale.to(p)

                # climb to the local maximum "w + e(w)"
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # get back to "w" from "w + e(w)"
                p.data = self.state[p]['old_p']

        # do the actual "sharpness-aware" update
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure: CLOSURE = None):
        if closure is None:
            raise NoClosureError(self.__str__)

        self.first_step(zero_grad=True)

        # the closure should do a full forward-backward pass
        with torch.enable_grad():
            closure()

        self.second_step()

    def grad_norm(self) -> torch.Tensor:
        # put everything on the same device, in case of model parallelism
        shared_device = self.param_groups[0]['params'][0].device
        return torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group['adaptive'] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    for group in self.param_groups
                    for p in group['params']
                    if p.grad is not None
                ]
            ),
            p=2,
        )

    def load_state_dict(self, state_dict: Dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
